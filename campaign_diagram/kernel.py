from campaign_diagram.fusion import *
from campaign_diagram.kernel_color import *
from fractions import Fraction
from copy import deepcopy
from deprecated import deprecated


import pandas as pd
from campaign_diagram.resource import Resource, ResourceType, ResourceRegistry
from math import isclose

default_comp = "Compute Pool"
default_mem = "Memory Pool"

# Class to hold the parameters for Kernel
class Kernel:
    def __init__(self,
                 name,
                 start=0,
                 duration=0,
                 compute_util=0,
                 bw_util=0,
                 origin=None,
                 bw_util_limit=1.0,
                 throttled_duration=0,
                 fusion_group=None,
                 fusion_type=FusionType.NONE,
                 windup=0.0,
                 compute_type=None,
                 memory_type=None,
                 compute_subutils=None,
                 memory_subutils=None):

        self.name = name
        self.start = start
        self.duration = duration
        self.orig_duration = duration
        self.throttled_duration = throttled_duration

        # relative to the TOTAL available resource
        self.compute_util = compute_util
        self.bw_util = bw_util

        # Relative to its resource subtype
        self.compute_util_raw = compute_util
        self.bw_util_raw = bw_util
        self.compute_type = compute_type
        self.memory_type = memory_type

        ########################################################################
        # Optional per-sub-resource raw-util maps (opt-in PATH c).             #
        #                                                                      #
        #   compute_subutils = {"Tensor": t, "FMA": f}   -- accumulative pool  #
        #   memory_subutils  = {"L2": l, "DRAM": d}      -- non-accum hierarchy#
        #                                                                      #
        # Each value is the raw utilization of that ONE sub-resource (0..1,    #
        # relative to its own peak / its own level gauge). When both are None  #
        # the kernel is the historical SCALAR kernel and every derived field   #
        # below stays byte-identical. When present, update_global_util_view    #
        # collapses each map into the phase aggregate PER the sub-resource's   #
        # `accumulative` flag (resolved from the registry):                    #
        #   accumulative resources POOL  -> SUM_r util_r * p_of_tot_r           #
        #   non-accumulative resources   -> MAX_r util_r  (independent gauges)  #
        # a mixed map takes max(pooled-sum, gauge-max). With the registry      #
        # defaults (compute accumulative, memory non-accumulative) this is the #
        # historical compute-sum / memory-max. The per-sub-resource pool fills #
        # are stashed for the renderer. The p_of_tot weights + flags come from #
        # the registry passed to update_global_util_view; a single-entry map   #
        # over a p_of_tot==1.0 pool reproduces the scalar kernel exactly.      #
        ########################################################################
        self.compute_subutils = compute_subutils
        self.memory_subutils = memory_subutils
        # Per-sub-resource pool fills, populated by update_global_util_view
        # when the maps are present: name -> (raw_util, p_of_tot, fill) where
        # fill == raw_util * p_of_tot. Consumed READ-ONLY by the renderer.
        self.compute_subfills = {}
        self.memory_subfills = {}
        # Resolved per-sub-resource `accumulative` flag (name -> bool), pinned
        # alongside the fills so the overlap/throttle math in intervals.py can
        # partition pooled vs independent sub-resources WITHOUT a registry.
        self.compute_subaccum = {}
        self.memory_subaccum = {}
      
        if origin is None:
            self.origin = self
        else:
            self.origin = origin
          
        self.bw_util_limit = bw_util_limit
        self.compute_color = None
        self.bw_color = None
        self.fusion_group = fusion_group
        self.fusion_type = fusion_type
        self.windup = windup

        ########################################################################
        # was_transformed: True once .dilate() or cascade-level .throttle() has #
        # touched this kernel. Used by the renderer's `show_ideal_dashes`       #
        # kwarg to gate the "extra time over ideal" dashed lines (Family 1).   #
        # Vanilla kernels stay False; tile()/split() do NOT flip this (they    #
        # are mechanical reshape operations, not user-facing transforms).      #
        ########################################################################
        self.was_transformed = False

        self.update_global_util_view()
      

    def update_global_util_view(
        self,
        registry: "ResourceRegistry" = None,
        mem_resource: "Resource" = None,
        comp_resource: "Resource" = None,
        comp_resource_name: str = default_comp,
        mem_resource_name: str = default_mem):
        """
        Recompute the *pool-relative* utilization view for this kernel.

            util_pool = util_raw x resource.attrs['p_of_tot']   (default 1.0)

        Source of truth (set ONCE in __init__, NEVER rescaled here):
          - self.compute_util_raw / self.bw_util_raw : the per-sub-resource
            utilizations as the constructor received them.
          - self.compute_type / self.memory_type     : the kernel's chosen
            sub-resource NAMES (may be None for single-resource kernels).

        This method is IDEMPOTENT: every derived field is recomputed from the
        `_raw` values on each call, so calling it repeatedly with the same
        registry yields identical results.

        Primary API:
            kernel.update_global_util_view(registry=reg)
        resolves the pool weights from self.compute_type / self.memory_type
        via the supplied registry. Explicit `comp_resource` / `mem_resource`
        (Resource object OR name string) override that resolution.
        """

        #######################################################################
        # Step 1: pin the kernel's chosen sub-resource NAMES                  #
        #                                                                     #
        # We must NOT clobber a user-supplied compute_type / memory_type.     #
        # Only when the name is None do we fall back to the default pool      #
        # name. This keeps e.g. compute_type="Special1D" intact even when    #
        # update_global_util_view() is called with no arguments from         #
        # __init__.                                                           #
        #######################################################################

        # Compute sub-resource name: keep what the user gave us; only when it
        # is None do we substitute the default compute pool name.
        if self.compute_type is None:
            self.compute_type = default_comp

        # Memory sub-resource name: same rule. This also fixes the historical
        # self.mem_type vs self.memory_type inconsistency -- we now write
        # self.memory_type ONLY (the renderer reads kernel.memory_type).
        if self.memory_type is None:
            self.memory_type = default_mem

        #######################################################################
        # Step 2: resolve the COMPUTE pool weight                             #
        #                                                                     #
        # Precedence:                                                         #
        #   (a) an explicitly passed comp_resource -- Resource OR name str    #
        #   (b) a registry lookup of self.compute_type                        #
        #   (c) no registry at all -> single-resource default weight 1.0      #
        #######################################################################

        # comp_w is the fraction of the global pool this kernel's compute
        # sub-resource represents. Defaults to 1.0 (single-resource case).
        comp_w = 1.0

        if comp_resource is not None:
            # The caller handed us an explicit compute resource. It may be a
            # Resource object directly, or just a NAME string we resolve via
            # the registry (if one is available).
            if isinstance(comp_resource, Resource):
                resolved_comp = comp_resource
            elif registry is not None:
                resolved_comp = registry.get(comp_resource)
            else:
                resolved_comp = None
            if resolved_comp is not None:
                comp_w = resolved_comp.attrs.get("p_of_tot", 1.0)

        elif registry is not None:
            # No explicit resource, but we have a registry: look up this
            # kernel's compute sub-resource by name (self.compute_type was
            # pinned in Step 1, so it is never None here).
            resolved_comp = registry.get(self.compute_type)
            comp_w = resolved_comp.attrs.get("p_of_tot", 1.0)

        # else: no registry and no explicit resource -> comp_w stays 1.0,
        # which is exactly the single-resource behavior we must preserve.

        #######################################################################
        # Step 3: resolve the MEMORY pool weight (mirror of Step 2)           #
        #######################################################################

        # mem_w is the fraction of the global pool this kernel's memory
        # sub-resource represents. Defaults to 1.0 (single-resource case).
        mem_w = 1.0

        if mem_resource is not None:
            # Explicit memory resource: Resource object, or name to resolve.
            if isinstance(mem_resource, Resource):
                resolved_mem = mem_resource
            elif registry is not None:
                resolved_mem = registry.get(mem_resource)
            else:
                resolved_mem = None
            if resolved_mem is not None:
                mem_w = resolved_mem.attrs.get("p_of_tot", 1.0)

        elif registry is not None:
            # Registry lookup of this kernel's memory sub-resource by name.
            resolved_mem = registry.get(self.memory_type)
            mem_w = resolved_mem.attrs.get("p_of_tot", 1.0)

        # else: no registry / no explicit resource -> mem_w stays 1.0.

        #######################################################################
        # Step 4: write the pool-relative view (idempotent)                   #
        #                                                                     #
        # Always derive from the *_raw values so repeated calls with the     #
        # same weights are stable. With single-resource defaults (weight     #
        # 1.0) these are numerically identical to the raw inputs.            #
        #######################################################################

        # Pool-relative utilizations = raw utilization scaled by pool share.
        self.compute_util = self.compute_util_raw * comp_w
        self.bw_util      = self.bw_util_raw * mem_w

        # Keep the resolved weights around for the tooltip layer to display.
        self.comp_perc = comp_w
        self.bw_perc   = mem_w

        #######################################################################
        # Step 4b: per-sub-resource maps override (opt-in PATH c)             #
        #                                                                     #
        # When a sub-util map is present the phase aggregate is derived from  #
        # the map instead of the scalar *_raw value, per side independently.  #
        # _aggregate_submap partitions each map by the sub-resource's         #
        # `accumulative` flag (resolved from the registry):                   #
        #                                                                     #
        #   ACCUMULATIVE resources POOL onto one budget -> the aggregate is   #
        #     the SUM of their fills (util_r * p_of_tot_r). That sum is       #
        #     ALREADY pool-relative, so *_util == *_util_raw here and the     #
        #     *_perc weight collapses to 1.0.                                 #
        #                                                                     #
        #   NON-ACCUMULATIVE resources are independent 0..1 gauges (never     #
        #     summed against each other) -> the aggregate is the MAX raw util.#
        #                                                                     #
        # A MIXED map takes max(pooled-sum, gauge-max). With the registry     #
        # defaults (compute accumulative, memory non-accumulative) this is    #
        # the historical compute-sum / memory-max, byte-for-byte. Per-level   #
        # fills + resolved flags are stashed for the renderer / overlap math. #
        #                                                                     #
        # Step 5 below then computes ONE cb_ideal / mb_ideal / ideal /        #
        # throttled per phase from these aggregates, exactly as for a scalar  #
        # kernel -- the formulas are untouched.                              #
        #######################################################################
        if self.compute_subutils is not None:
            fills, accum_flags, agg = self._aggregate_submap(
                self.compute_subutils, registry)
            self.compute_subfills = fills
            self.compute_subaccum = accum_flags
            self.compute_util_raw = agg
            self.compute_util = agg
            self.comp_perc = 1.0

        if self.memory_subutils is not None:
            fills, accum_flags, agg = self._aggregate_submap(
                self.memory_subutils, registry)
            self.memory_subfills = fills
            self.memory_subaccum = accum_flags
            self.bw_util_raw = agg
            self.bw_util = agg
            self.bw_perc = 1.0

        #######################################################################
        # Step 5: recompute ideal / throttled durations from the _raw values #
        #                                                                     #
        # Formulas intentionally UNCHANGED:                                   #
        #   cb_ideal = duration * compute_util_raw                            #
        #   mb_ideal = duration * bw_util_raw                                 #
        #   ideal    = max(cb_ideal, mb_ideal)                                #
        #   throttled= duration - ideal                                       #
        #######################################################################

        # Ideal duration if the kernel were purely compute-bound.
        self.cb_ideal_duration = self.duration * self.compute_util_raw

        # Ideal duration if the kernel were purely memory(bandwidth)-bound.
        self.mb_ideal_duration = self.duration * self.bw_util_raw

        # The achievable ideal is bounded by the more-demanding of the two.
        self.ideal_duration = max(
            self.cb_ideal_duration,
            self.mb_ideal_duration
        )

        # Throttled time = wall-clock duration minus the relevant ideal.
        self.cb_throttled_duration = self.duration - self.cb_ideal_duration
        self.mb_throttled_duration = self.duration - self.mb_ideal_duration
        self.throttled_duration = self.duration - self.ideal_duration

    def _sub_p_of_tot(self, name, registry, default=1.0):
        """Resolve a sub-resource's pool share (p_of_tot) from the registry.

        Used only on the maps (PATH c) path. Falls back to ``default`` (1.0)
        when no registry is supplied -- e.g. the transient __init__ call that
        runs before the caller re-resolves with update_global_util_view(
        registry=reg) -- or when the name is not registered. With p_of_tot 1.0
        a single-entry map reproduces the scalar kernel exactly.
        """
        if registry is None:
            return default
        try:
            return float(registry.get(name).attrs.get("p_of_tot", default))
        except (KeyError, AttributeError):
            return default

    def _sub_accumulative(self, name, registry, default=True):
        """Resolve a sub-resource's ``accumulative`` flag from the registry.

        Mirrors _sub_p_of_tot. With no registry (the transient __init__ call)
        or an unregistered name we fall back to ``default`` -- True, the
        pooling/compute-pool default -- so an as-yet-unresolved map keeps
        summing until update_global_util_view(registry=reg) re-runs with the
        real flags.
        """
        if registry is None:
            return default
        try:
            return bool(registry.get(name).attrs.get("accumulative", default))
        except (KeyError, AttributeError):
            return default

    def _aggregate_submap(self, submap, registry):
        """Collapse a per-sub-resource raw-util map into one axis aggregate.

        Each entry's ``accumulative`` flag (resolved from the registry) decides
        how it folds into the axis budget:

          * ACCUMULATIVE resources POOL -- they share one budget, so they
            contribute a weighted SUM of ``util * p_of_tot``.
          * NON-ACCUMULATIVE resources are independent 0..1 gauges -- the
            busiest sets the floor, so they contribute a MAX of raw util.

        A homogeneous map collapses to just that sum (all-accumulative) or that
        max (all-non-accumulative); a MIXED map takes max(pooled-sum,
        gauge-max) -- both express "fraction of the axis budget bound", and a
        non-accumulative level is never silently summed into the pool.

        Returns (fills, accum_flags, aggregate) where
        fills[name] == (util, p_of_tot, util * p_of_tot) and
        accum_flags[name] == the resolved bool.
        """
        fills = {}
        accum_flags = {}
        pooled_sum = 0.0
        gauge_max = 0.0
        have_pooled = False
        have_gauge = False
        for name, u in submap.items():
            u = float(u)
            p = self._sub_p_of_tot(name, registry)
            is_accum = self._sub_accumulative(name, registry)
            fills[name] = (u, p, u * p)
            accum_flags[name] = is_accum
            if is_accum:
                pooled_sum += u * p
                have_pooled = True
            else:
                gauge_max = max(gauge_max, u)
                have_gauge = True
        if have_pooled and have_gauge:
            aggregate = max(pooled_sum, gauge_max)
        elif have_gauge:
            aggregate = gauge_max
        else:
            aggregate = pooled_sum
        return fills, accum_flags, aggregate

    @property
    def end(self):
        """ Return end based on start and duration properties """

        return float(self.start) + float(self.duration)
      
    def set_start(self, last_end=0):
        """Sets the start time based on the last end or defaults to 0."""

        self.start = last_end

        return self

    def set_color(self, color):
        """ Set the color of the kernel. """

        self.compute_color = color
        self.bw_color = KernelColor.lightenColor(color,
                                                 amount=0.5)

        return self

    def set_windup(self, windup):
      """ 
      Set the wind up time for this kernel, the amount of time
      before a tile of data is ready for consumption by the next
      Einsum.
      """
      self.windup = windup
         
    def clone(self):
        """ Create a copy of kernel with a new origin """

        k = self.copy()
        k.origin = k
        return k

    def copy(self):
        """Creates a copy of the kernel.

        Note: colors ARE copied now

        """
        
        return deepcopy(self)

        # return Kernel(name=self.name,
        #               start=self.start,
        #               duration=self.duration,
        #               compute_util=self.compute_util,
        #               bw_util=self.bw_util,
        #               origin=self.origin,
        #               bw_util_limit=self.bw_util_limit,
        #               throttled_duration=self.throttled_duration,
        #               fusion_group=self.fusion_group,
        #               fusion_type=self.fusion_type,
        #               windup=self.windup)

    def scale_duration(self, scale):
        #self.duration *= scale
      
        # we usually call this when we're duplicating or splitting.
        orig_duration = self.duration
        self.orig_duration = orig_duration

        #What fraction of the runtime is non-ideal?
        throttled_frac = self.throttled_duration/orig_duration
        cb_throttled_frac = self.cb_throttled_duration/orig_duration
        mb_throttled_frac = self.mb_throttled_duration/orig_duration
      
        self.duration *= scale


        # have the proportions remain the same
        self.throttled_duration = throttled_frac * self.duration
        self.cb_throttled_duration = cb_throttled_frac * self.duration
        self.mb_throttled_duration = mb_throttled_frac  * self.duration   

        # But the resource utilizations stay the same, so no changes
      
        return self


    # We now have throttled_duration as a variable
    # that represents how much time is due to non-ideal activities
    # TODO: ideally, we should know how much of the resource well we have
    #       given this resource well, we can recalculate the ideal duration
    #       as 
    def dilate(self, dilation):

        ########################################################################
        # Mark this kernel as transformed: the renderer's `show_ideal_dashes`  #
        # flag uses this to decide whether to keep the dashed extensions for   #
        # a given kernel. Dilation is a user-facing transform (pipeline(spread #
        # =True) calls it internally too), so the flag belongs here.          #
        ########################################################################
        self.was_transformed = True

        orig_duration = self.duration
        self.orig_duration = orig_duration

        throttled_frac = self.throttled_duration/orig_duration
        cb_throttled_frac = self.cb_throttled_duration/orig_duration
        mb_throttled_frac = self.mb_throttled_duration/orig_duration
      
      
        # dilate the overall runtime
        self.duration *= dilation
        # print(f"IN DILATE: {self.name}: td is {self.throttled_duration}")

        # The region 
        self.throttled_duration = throttled_frac * self.duration
        self.cb_throttled_duration = cb_throttled_frac * self.duration
        self.mb_throttled_duration = mb_throttled_frac  * self.duration          
      
        #self.throttled_duration += self.duration - self.ideal_duration
        #self.throttled_duration += self.duration - orig_duration

        inverse_dilation = 1.0 / dilation

        self.compute_util *= inverse_dilation
        self.bw_util *= inverse_dilation

        #######################################################################
        # STALE-IDEAL FIX                                                     #
        #                                                                     #
        # cb_ideal_duration / mb_ideal_duration / ideal_duration were set by  #
        # update_global_util_view as duration * *_util_raw and were NEVER     #
        # refreshed here, so after a dilation they still described the        #
        # PRE-dilation kernel (their throttled counterparts, in contrast, are #
        # rescaled just above). Anything that reads an ideal duration after a #
        # dilate therefore saw a value inconsistent with duration and with    #
        # throttled_duration. Re-derive them from the freshly-rescaled        #
        # throttled durations so ideal + throttled == duration holds on each  #
        # axis, keeping the whole set self-consistent.                        #
        #######################################################################
        self.cb_ideal_duration = self.duration - self.cb_throttled_duration
        self.mb_ideal_duration = self.duration - self.mb_throttled_duration
        self.ideal_duration = self.duration - self.throttled_duration

        # print(f"We dilated {orig_duration}, {self.throttled_duration}\n")
        return self


    @deprecated
    # This is a bug...
    # If we already dilated, the math for throttled_duration isn't coming out
    # to what we expect, that is -- the original duration, before any dilation, 
    # should actually stay the same.
    # one way to check: 
    # throttled_duration = orig_duration*(dilation1*dilation2*...) - orig_duration
    def old_dilate(self, dilation):

        orig_duration = self.duration
        self.orig_duration = orig_duration
      
        self.duration *= dilation
        # print(f"IN DILATE: {self.name}: td is {self.throttled_duration}")

        self.throttled_duration *= dilation
        self.throttled_duration += self.duration - orig_duration

        inverse_dilation = 1.0 / dilation

        self.compute_util *= inverse_dilation
        self.bw_util *= inverse_dilation

        # print(f"We dilated {orig_duration}, {self.throttled_duration}\n")
        return self


    def split(self, split_time):
        """Splits the kernel into two at split_time.

        Notes:
            - throttled_duration is dropped during a split
            - bw_util_limit - is dropped during a split
        """

        if split_time <= self.start or split_time >= self.end:
            return self.copy(), None

        # First part is from start to split_time
        first_part = Kernel(name=self.name,
                            start=self.start,
                            duration=split_time - self.start,
                            compute_util=self.compute_util,
                            bw_util=self.bw_util,
                            origin=self.origin,
                            fusion_group=self.fusion_group,
                            fusion_type=self.fusion_type,
                            windup=self.windup,
                            compute_type=self.compute_type,
                            memory_type=self.memory_type,
                            compute_subutils=self.compute_subutils,
                            memory_subutils=self.memory_subutils
                            )

        # Second part is from split_time to original end
        second_part = Kernel(name=self.name,
                             start=split_time,
                             duration=self.end - split_time,
                             compute_util=self.compute_util,
                             bw_util=self.bw_util,
                             origin=self.origin,
                             fusion_group=self.fusion_group,
                             fusion_type=self.fusion_type,
                             windup=self.windup,
                             compute_type=self.compute_type,
                             memory_type=self.memory_type,
                             compute_subutils=self.compute_subutils,
                             memory_subutils=self.memory_subutils
                            )

        ########################################################################
        # PRESERVE MULTI-RESOURCE WEIGHTING ACROSS A TIME SPLIT                #
        #                                                                      #
        # Slicing a kernel in TIME changes only WHEN it runs (start) and for   #
        # HOW LONG (duration). It does NOT change which sub-resources it uses, #
        # how hard it drives them (raw util), or what fraction of the global   #
        # pool those sub-resources represent (the p_of_tot weights). But the   #
        # Kernel(...) constructor above re-ran update_global_util_view() with  #
        # NO registry, which collapsed comp_perc/bw_perc to 1.0 and copied the #
        # pool-relative util into the *_raw fields -- destroying the parent's  #
        # per-resource weighting. So we explicitly restore the parent's        #
        # correct state onto each half here, then recompute ONLY the           #
        # duration-dependent ideal/throttled fields from each half's own       #
        # (shorter) duration. For a single-resource kernel raw == pool and     #
        # comp_perc == 1.0, so every line below is a no-op and the recompute   #
        # reproduces exactly what the constructor already produced (byte-      #
        # identical -- guarded by the regression harness).                     #
        ########################################################################
        for p in (first_part, second_part):

            # Per-sub-resource RAW compute util is a physical property of the
            # kernel; a time slice cannot change how hard it drives compute.
            p.compute_util_raw = self.compute_util_raw

            # Same for bandwidth: the raw (sub-resource-relative) bw demand is
            # unchanged by cutting the kernel in time.
            p.bw_util_raw = self.bw_util_raw

            # The resolved compute pool weight (sub-resource p_of_tot) is a
            # property of the resource binding, not of the time window.
            p.comp_perc = self.comp_perc

            # Likewise the resolved memory pool weight must survive the split.
            p.bw_perc = self.bw_perc

            # Pool-relative compute util = raw * p_of_tot; neither factor is
            # affected by a time split, so the parent's value carries over.
            p.compute_util = self.compute_util

            # Pool-relative bandwidth util carries over for the same reason.
            p.bw_util = self.bw_util

            # The compute sub-resource NAME (already passed to the ctor) must
            # remain exactly the parent's -- ensure it, do not let it drift.
            p.compute_type = self.compute_type

            # The memory sub-resource NAME must likewise stay the parent's.
            p.memory_type = self.memory_type

            # PATH c: the per-sub-resource maps + their resolved pool fills are
            # physical properties of the kernel; a time slice cannot change
            # which sub-resources it drives or how hard. Carry the parent's
            # values so each half keeps its multi-sub-resource decomposition
            # (the ctor above already passed the raw maps, but the resolved
            # fills were re-derived with NO registry -- restore the parent's).
            p.compute_subutils = self.compute_subutils
            p.memory_subutils = self.memory_subutils
            p.compute_subfills = dict(self.compute_subfills)
            p.memory_subfills = dict(self.memory_subfills)
            # The resolved accumulative flags are part of the same binding;
            # carry them so a post-split throttle still partitions correctly.
            p.compute_subaccum = dict(self.compute_subaccum)
            p.memory_subaccum = dict(self.memory_subaccum)

            # `was_transformed` is a per-kernel marker for "user-facing
            # transform applied"; slicing in time mustn't drop it. Without
            # this line, a dilated kernel that's later split would lose its
            # transform flag and the renderer's show_ideal_dashes filter
            # would misclassify the halves as vanilla.
            p.was_transformed = self.was_transformed

            # Recompute the duration-dependent fields from THIS half's own
            # (shorter) duration, mirroring update_global_util_view's formulas
            # exactly so behavior is consistent with a freshly-built kernel.

            # Ideal duration if this half were purely compute-bound.
            p.cb_ideal_duration = p.duration * p.compute_util_raw

            # Ideal duration if this half were purely bandwidth-bound.
            p.mb_ideal_duration = p.duration * p.bw_util_raw

            # The achievable ideal is bounded by the more-demanding of the two.
            p.ideal_duration = max(p.cb_ideal_duration, p.mb_ideal_duration)

            # Throttled time = this half's wall-clock duration minus the
            # relevant ideal duration (compute-bound view).
            p.cb_throttled_duration = p.duration - p.cb_ideal_duration

            # Throttled time, bandwidth-bound view.
            p.mb_throttled_duration = p.duration - p.mb_ideal_duration

            # Overall throttled time for this half.
            p.throttled_duration = p.duration - p.ideal_duration

        #print(f"Split: {self.name} -- {self.start}, {self.end}, {split_time}")
        return first_part, second_part


    ###########################################################################
    # Functions for aggregating data
    ###########################################################################
    def __repr__(self):
        """Returns a represention of the Kernel's state """

        return (f"Kernel("
                f"name={self.name}, "
                f"start={self.start:.2f}, "
                f"duration={self.duration:.2f}, "
                f"throttled_duration={self.throttled_duration:.2f}, "
                f"compute={self.compute_util:.2f}, "
                f"bw={self.bw_util:.2f}, "
                f"fusion_group={self.fusion_group},"
                f"fusion_type={str(self.fusion_type)},"
                f"origin_start={self.origin.start},"
                f"windup={str(self.windup)})")

    def __str__(self):
        """Returns a human-readable string representation of the Kernel's state."""

        return (f"Kernel(name={self.name}, "
                f"start={self.start:.2f}, "
                f"duration={self.duration:.2f}, "
                f"throttled_duration={self.throttled_duration:.2f}, "
                f"compute_util={self.compute_util:.2f}, "
                f"bw_util={self.bw_util:.2f},"
                f"fusion_group={self.fusion_group},"
                f"fusion_type={str(self.fusion_type)},"
                f"windup={str(self.windup)}")

    def to_dict(self):
        return {
            "name": self.name,
            "start": self.start,
            "duration": self.duration,
            "end": self.end,
            "compute_util": self.compute_util,
            "bw_util": self.bw_util,
            "throttled_duration": self.throttled_duration,
            "bw_util_limit": self.bw_util_limit,
            "fusion_group": self.fusion_group,
            "fusion_type": str(self.fusion_type),  # assuming it's an Enum
            "windup": self.windup,
            "origin": self.origin.name if self.origin else None
        }

    def to_df(self):
        return pd.DataFrame([self.to_dict()])
      
