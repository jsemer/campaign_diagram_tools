import pandas as pd
import altair as alt

from campaign_diagram.kernel_color import KernelColor
from campaign_diagram.utils import generate_color_scale
from campaign_diagram.utils import gen_fusion_group_boundaries, load_df_as_cascade
from campaign_diagram.fusion import FusionType
from campaign_diagram.roofline import roofline
# READ-ONLY import: we only need the ResourceType enum so the multi-resource
# band ordering can tell COMPUTE resources apart from MEMORY resources when it
# walks the user's ResourceRegistry. resource.py itself is NEVER mutated here.
from campaign_diagram.resource import ResourceType

phase_legend_name = "Einsum"

class CampaignDiagramAltair2:
    def __init__(self, cascade, resource_registry=None):
        """Construct the diagram renderer.

        Parameters
        ----------
        cascade : Cascade
            The cascade of kernels to render (unchanged behavior).
        resource_registry : ResourceRegistry, optional
            The SAME ResourceRegistry the caller used to declare its compute /
            memory sub-resources (the object on which reg.add_compute(...) /
            reg.add_memory(...) were called). It is consumed READ-ONLY and is
            ONLY consulted on the opt-in multi_resource rendering path: there
            the mini-roof band stacking order is taken from the registry's
            DECLARATION (insertion) order rather than from the order resources
            happen to first appear among the cascade's kernels.

            API / precedence (documented once here, see also draw()):
              * Pass it to the constructor to set a default for every draw().
              * draw(..., resource_registry=...) OVERRIDES the constructor one
                for that single call.
              * If neither supplies a registry (None), the renderer FALLS BACK
                to the legacy kernel-first-appearance ordering, so the
                single-resource path and prior multi_resource callers are
                completely unaffected (the hard regression gate stays green).
        """
        cascade.assign_colors()
        self.cascade = cascade
        # Stash the (optional) registry as the instance-level default. May be
        # None -> fall back to the legacy kernel-appearance ordering.
        self.resource_registry = resource_registry
        self.kernels = sorted(
            cascade.kernels,
            key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name)
        )
        self.kernel_color_map = KernelColor()

    def update_color_scale(self, color_scale, tint=0.5):
        self.color_scale = color_scale
        self.tinted_color_scale = tint_scale(self.color_scale, tint=0.5)

    def generate_roofline(
        self,
        compute_roof, bw_roof,
        compute_div_factor, compute_div_unit,
        bw_div_factor, bw_div_unit,
        color_scale,
        legend_name="Phases",
        x_start = 1,
        ext = 100,
        point_size = 200  # 2026-05-20: marker AREA (px^2) for the roofline dots; bumped from the old implicit 80 so the points read at paper scale
    ):
        axis_x_name = "Operational Intensity (FLOPs/byte)"
        axis_y_name = f"Compute Throughput ({compute_div_unit})"

        rf_chart = roofline.gen_cottage_border(
            roof_compute_bw=[(compute_roof, bw_roof)],
            axis_x_name=axis_x_name,
            axis_y_name=axis_y_name,
            compute_div_factor=compute_div_factor,
            compute_div_unit=compute_div_unit,
            bw_div_factor=bw_div_factor,
            bw_div_unit=bw_div_unit,
            x_start = x_start,
            ext = ext
        )

        agg_latency = 0.0
        agg_mem_traffic = 0.0
        agg_comp_volume = 0.0

        for kernel in self.kernels:
            latency = kernel.duration  # seconds (or consistent unit)
            comp_volume = (kernel.compute_util * compute_roof) * latency  # FLOPs
            mem_traffic = (kernel.bw_util * bw_roof) * latency            # Bytes

            # avoid divide-by-zero
            if mem_traffic == 0 or latency == 0:
                continue

            point_ai = comp_volume / mem_traffic                          # FLOPs/byte
            point_tp = (comp_volume / compute_div_factor) / latency        # (scaled FLOPs/sec)

            rf_chart = roofline.plot_roofline_point_color_scale(
                ai = point_ai,
                compute = point_tp,
                name = kernel.name,
                chart = rf_chart,
                axis_x_name = axis_x_name,
                axis_y_name = axis_y_name,
                color_scale = color_scale,
                legend_name = legend_name,
                point_size = point_size,
            )

            agg_latency += latency
            agg_mem_traffic += mem_traffic
            agg_comp_volume += comp_volume

        if agg_mem_traffic != 0 and agg_latency != 0:
            agg_ai = agg_comp_volume / agg_mem_traffic
            agg_tp = (agg_comp_volume / compute_div_factor) / agg_latency

            rf_chart = roofline.plot_roofline_point_color_scale(
                ai=agg_ai,
                compute=agg_tp,
                name="Aggregated",
                chart=rf_chart,
                axis_x_name=axis_x_name,
                axis_y_name=axis_y_name,
                color_scale=color_scale,
                legend_name=legend_name,
                point_size=point_size
            )


        return rf_chart

    
    def generate_roofline_aggregate(
        self,
        compute_roof, bw_roof,
        compute_div_factor, compute_div_unit,
        bw_div_factor, bw_div_unit,
        color_scale,
        legend_name="Phases",
        x_start = 1,
        ext = 100,
        point_name="Aggregated",   # name/label for the plotted aggregate point
        base_chart=None,           # if given, layer the point onto this existing chart instead of drawing a fresh roofline border -> lets a caller stack >1 aggregate point (e.g. unpipelined + pipelined) on ONE roofline.
    ):
        axis_x_name = "Operational Intensity (FLOPs/byte)"
        axis_y_name = f"Compute Throughput ({compute_div_unit})"

        rf_chart = base_chart if base_chart is not None else roofline.gen_cottage_border(
            roof_compute_bw=[(compute_roof, bw_roof)],
            axis_x_name=axis_x_name,
            axis_y_name=axis_y_name,
            compute_div_factor=compute_div_factor,
            compute_div_unit=compute_div_unit,
            bw_div_factor=bw_div_factor,
            bw_div_unit=bw_div_unit,
            x_start = x_start,
            ext = ext
        )

        agg_latency = 0.0
        agg_mem_traffic = 0.0
        agg_comp_volume = 0.0
        current_parallel_start = None
        last_duration = 0

        # For this kernel
        for kernel in self.kernels:
            # We are at a new parallel block of phases.
            if kernel.start != current_parallel_start:
                current_parallel_start = kernel.start
                
                latency = kernel.duration  # seconds (or consistent unit)
                comp_volume = (kernel.compute_util * compute_roof) * latency  # FLOPs
                mem_traffic = (kernel.bw_util * bw_roof) * latency            # Bytes

                agg_latency = kernel.start
                agg_mem_traffic += mem_traffic
                agg_comp_volume += comp_volume
                last_duration = latency
                
            #else we are in some previously seen parallel block
            #we still need the memory traffic and the compute volume
            else:
                comp_volume = (kernel.compute_util * compute_roof) * latency  # FLOPs
                mem_traffic = (kernel.bw_util * bw_roof) * latency            # Bytes

                agg_mem_traffic += mem_traffic
                agg_comp_volume += comp_volume
                
        # We're done! Whatever the last start time we saw was + that parallel block's duration is our final latency
        agg_latency += last_duration
        

        if agg_mem_traffic != 0 and agg_latency != 0:
            agg_ai = agg_comp_volume / agg_mem_traffic
            agg_tp = (agg_comp_volume / compute_div_factor) / agg_latency

            rf_chart = roofline.plot_roofline_point_color_scale(
                ai=agg_ai,
                compute=agg_tp,
                name=point_name,
                chart=rf_chart,
                axis_x_name=axis_x_name,
                axis_y_name=axis_y_name,
                color_scale=color_scale,
                legend_name=legend_name,
                point_size=120
            )


        return rf_chart
         
    def draw(self,
             bw_util_scaling=0.5,  # size of the guard band (memory-bandwidth pipe height as fraction of y-axis at 100% util). 2026-05-20: bumped 0.125 -> 0.25 -> 0.5 so 100% mem bw renders at half the height of full compute util.
             include_boundaries=False, # boundaries between fusion grou[s
             include_tile_boundaries=True, 
             log_scale=False, 
             dual_mode=False, # dual_mode vs. colocated mode
             return_df=False, # return the data frame used to plot the campaign diagram
             guard_position="above", # above, centered, below
             alpha=0.5, 
             x_title='Execution Time (cycles)',
             title=None, 
             color_scale = None, 
             einsum_order=None,
             line_thickness = 2,
             phase_name = "Einsum",
             kernel_sort_preset=None,
             multi_resource=False,  # Phase 2 (grey-band rev): opt-in SOLID GREY per-resource bands + full-width roofs. Default False == byte-identical legacy behavior.
             resource_registry=None,  # Phase 2 (rev3): optional ResourceRegistry; if given (and multi_resource), bands stack in REGISTRY DECLARATION order. Overrides the constructor's. None -> legacy kernel-appearance order.
             show_ideal_dashes=True,  # 2026-05-20: when False, hide the "extra time over ideal" dashed extensions for VANILLA kernels (Kernel.was_transformed == False). Dashed lines from .throttle()/.dilate() kernels still draw. Default True == byte-identical legacy behavior.
             show_x_axis=True,  # 2026-05-20: when False, drop the x-axis entirely (title + labels + ticks). Used to stack two diagrams that share one bottom x-axis via alt.vconcat(...).resolve_scale(x='shared'). Default True == unchanged.
             y_max=1.52,  # 2026-05-21: top of the y-axis (Compute Utilization) domain. Default 1.52 == unchanged. Pass e.g. 1.2 for figures rendered at a smaller bw_util_scaling so the axis is not mostly empty headroom.
             y_min=0.0,   # 2026-05-21: bottom of the y-axis (Compute Utilization) domain. Default 0.0 == unchanged. Pass e.g. 0.4 to crop empty headroom below the bands and render a short, wide (4:1) panel.
             width=600,   # 2026-05-21: chart width in px. Default 600 == unchanged (was hard-coded). Lower/raise with height to set the aspect ratio.
             height=300,  # 2026-05-21: chart height in px. Default 300 == unchanged (was hard-coded). Pass e.g. 150 (with width=600) for a 4:1 panel.
             show_legend=True,  # 2026-05-21: when False, omit the phase-color legend entirely (the legend_only layer is not composed in). Default True == unchanged.
            ):
        """Render an Altair chart. Return (chart, DataFrame) if return_df is True.

        multi_resource (bool, default False):
            When False, the rendering path is EXACTLY the legacy single-resource
            path (no behavioral change whatsoever -- this is the hard regression
            gate). When True, we render the "SOLID GREY BANDS" stacked-band
            visual (dual mode only):

              * Each compute SUB-resource (kernel.compute_type) gets a stacked
                horizontal BAND whose height == that resource's p_of_tot pool
                share (== the per-kernel comp_perc, constant per resource).
                Memory sub-resources stack identically on the negative side
                using bw_perc.
              * Each resource band is filled with an OPAQUE per-resource GREY
                drawn from an evenly-spaced ramp (darkest #7a7a7a -> lightest
                #b0b0b0) in ResourceRegistry DECLARATION order: the FIRST
                resource is darkest, later resources progressively lighter.
                The memory side uses an INDEPENDENT ramp restarted from the
                dark end for the memory resources. This REPLACES the old
                "white gap".
              * The Einsum-coloured fill is drawn OPAQUE ON TOP of the grey
                band, rising by exactly Compute_Utilization (the pool-relative
                value already in the df, == raw_util * p_of_tot). Concurrent
                Einsums on the SAME resource STACK cumulatively WITHIN that
                resource's grey band. The unfilled remainder of the band shows
                the grey.
              * Each resource's roof is a SINGLE solid thin BLACK horizontal
                rule at its fixed cumulative band_top spanning the ENTIRE x
                domain of the cascade, with a BLACK label. These are visually
                distinct from the dashed darkgray +/-1.0 / 0.0 "100% of ALL
                resources" reference lines (solid thin black vs dashed gray).
              * A SEPARATE combined "Resources" pixel-overlay legend maps each
                grey swatch -> resource name + p_of_tot (compute first, then
                memory, in registry order). Black text.

            COLOR stays the Einsum identity (existing solid scale) -- no
            hatch / hash / resource colour channel is added.

            multi_resource now works in BOTH render modes (the dual auto-enable
            was removed): with dual_mode=True it drives the existing mirrored
            dual grey-band visual; with dual_mode=False it drives the NEW
            colocated multi-resource visual -- full-width dark-grey roof rules
            at each compute sub-resource's cumulative p_of_tot boundary, plus a
            persistently partitioned memory "pipe" whose per-resource grey
            segments shade the whole timeline and whose Einsum-coloured used-bw
            fill is offset to start at the bottom of ITS memory sub-resource's
            segment. dual_mode is honoured verbatim (no promotion).

        resource_registry (ResourceRegistry, optional):
            The caller's ResourceRegistry (the object reg.add_compute(...) /
            reg.add_memory(...) were called on). Used READ-ONLY and ONLY on the
            multi_resource path. When supplied, the mini-roof bands stack in
            the registry's DECLARATION order (the order resources were added
            to reg._by_name, an insertion-ordered dict) instead of the order
            they first appear among the cascade's kernels.

            Precedence: this draw() kwarg OVERRIDES the registry passed to the
            constructor; the constructor's value is the fallback default; if
            neither is given (None) we fall back to the legacy kernel-first-
            appearance ordering -- which keeps the single-resource regression
            gate and previous multi_resource callers byte-identical.

            Only COMPUTE-typed registry resources order the compute bands and
            only MEMORY-typed ones order the memory bands. The default single
            resource pools ("Compute Pool" / "Memory Pool") are EXCLUDED from
            the ordering unless a kernel actually uses them, and a registry
            resource is only given a band if some kernel in the cascade
            actually uses it. Each band's height is the registry resource's
            authoritative attrs["p_of_tot"]; bands are FIXED GLOBAL CUMULATIVE
            in that registry order (band_bottom_i = sum of p_of_tot of all
            resources declared before i, band_top_i = band_bottom_i +
            p_of_tot_i). Memory mirrors onto the negative Y axis.
        """
        global phase_legend_name
        phase_legend_name = phase_name

        ########################################################################
        # Phase 2 (rev) / colocated MR: multi-resource opt-in setup            #
        #                                                                      #
        # HISTORY: multi_resource USED to force-promote dual_mode=True because  #
        # the grey-band visual only existed for the mirrored dual view. That    #
        # auto-enable has been REMOVED so multi_resource now works in BOTH      #
        # modes, as a clean two-mode shape:                                     #
        #                                                                      #
        #   * dual_mode=True  + multi_resource=True -> the EXISTING, final/     #
        #     approved dual grey-band path (UNCHANGED below).                   #
        #   * dual_mode=False + multi_resource=True -> the NEW colocated        #
        #     multi-resource path (full-width grey compute roofs + a            #
        #     persistently partitioned, correctly-offset memory pipe).          #
        #                                                                      #
        # WHY removing the promotion is safe for the regression gate: the ONLY  #
        # caller that passes multi_resource is the notebook, and it passes      #
        # dual_mode explicitly; the single-resource regression harness has NO   #
        # multi_resource cases at all (multi_resource defaults to False there). #
        # So with multi_resource=False this block is still inert and every      #
        # single-resource colocated AND dual snapshot stays byte-identical --   #
        # removing the auto-enable cannot change a non-MR figure because the    #
        # `if multi_resource ...` predicate was never True for them anyway.     #
        ########################################################################
        # (Auto-enable of dual_mode intentionally REMOVED -- see banner above:
        #  multi_resource now honours the caller's dual_mode verbatim so the
        #  colocated MR path can exist as its own second mode.)
        # Stash on the instance so BOTH the dual-mode rendering block and the
        # new colocated-mode rendering block can branch on it.
        self.multi_resource = multi_resource

        ########################################################################
        # Phase 2 (rev3): resolve the EFFECTIVE ResourceRegistry               #
        #                                                                      #
        # Precedence (documented in the docstring above):                      #
        #   1. the draw(resource_registry=...) kwarg, if the caller passed one #
        #   2. otherwise the registry stashed on the instance by __init__      #
        #   3. otherwise None -> the multi-resource block falls back to the    #
        #      legacy "order resources by first appearance in cascade.kernels" #
        #      behavior, so single-resource output and older multi_resource    #
        #      callers stay byte-identical (the hard regression gate).         #
        #                                                                      #
        # We only read the registry; resource.py is never mutated.             #
        ########################################################################
        # The draw kwarg wins; fall back to whatever the constructor stored
        # (which itself may be None). getattr guards instances built before
        # this attribute existed.
        if resource_registry is not None:
            self._effective_resource_registry = resource_registry
        else:
            self._effective_resource_registry = getattr(
                self, "resource_registry", None)


        # we may want to modify the sorting:
        if kernel_sort_preset is not None:
          if kernel_sort_preset == "min_compute_first":
              kernel_sort_key = lambda k: (k.start, k.compute_util, -k.bw_util, k.name)
          elif kernel_sort_preset == "max_compute_first":
              kernel_sort_key = lambda k: (k.start, -k.compute_util, -k.bw_util, k.name)
          elif kernel_sort_preset == "max_bw_first":
              kernel_sort_key = lambda k: (k.start, -k.bw_util, k.compute_util, k.name)
          elif kernel_sort_preset == "alphabetical":
              # Stack overlapping phases ALPHABETICALLY by name. Within a
              # parallel/overlap window (shared k.start), the alphabetically
              # FIRST name (e.g. "Phase A") sorts first and is drawn at the
              # BOTTOM of the stack; later names stack on top. Used by the
              # paper notebooks so phase ordering is stable + intuitive
              # regardless of each phase's bw/compute utilization.
              kernel_sort_key = lambda k: (k.start, k.name)
          self.kernels = sorted(self.kernels, key=kernel_sort_key)
        else:
          pass #keep the same sorting in __init__!

        self.guard_position = guard_position
        self.dual_mode = dual_mode
        self.alpha = alpha

        ################################################################
        # Stash show_ideal_dashes so the throttled-line emit code can   #
        # consult it. When False the renderer filters out the dashed    #
        # extensions whose kernels were never .throttle()d/.dilate()d   #
        # (kernel.was_transformed == False). Default True preserves     #
        # the legacy "always draw all dashes" behavior.                 #
        ################################################################
        self.show_ideal_dashes = show_ideal_dashes

        ################################################################
        # Stash the bandwidth-pipe scaling factor on self so that       #
        # downstream renderer helpers (draw_guard_bands, draw_tile_     #
        # boundaries) can read it. Previously this was honored only by  #
        # the multi-resource path (which reads `bw_util_scaling`        #
        # directly into `_pipe_h`) while the single-resource colocated  #
        # path hardcoded 0.125 inside `draw_guard_bands()`. By storing  #
        # it here we let `draw_guard_bands()` interpolate the value     #
        # into its Vega-Lite expressions, so callers passing e.g.       #
        # `bw_util_scaling=0.25` actually get a taller memory pipe.     #
        ################################################################
        self.bw_util_scaling = bw_util_scaling

        drawing_data = []
        throttled_lines = []
        mem_throttled_lines = []
        comp_throttled_lines = []
        current_parallel_start = None
        cumulative_compute_util = 0
        cumulative_bw_util = 0
         
        # calculate the cumulative utilizations for each kernel
        for i, kernel in enumerate(self.kernels):
            if kernel.start != current_parallel_start:
                current_parallel_start = kernel.start
                #cumulative utilization
                cumulative_compute_util = kernel.compute_util
                cumulative_bw_util = kernel.bw_util
                cutil_start = 0.0
                mutil_start = 0.0
            else:
                cutil_start = cumulative_compute_util
                mutil_start = cumulative_bw_util
                cumulative_compute_util += kernel.compute_util
                cumulative_bw_util += kernel.bw_util

            available_bw = max(0.0, 1.0 - (cumulative_bw_util - kernel.bw_util))
            memory_well = available_bw

            kernel_start = float(kernel.start)
            kernel.compute_util = float(kernel.compute_util)
            kernel.bw_util = float(kernel.bw_util)
            kernel.throttled_duration = float(kernel.throttled_duration)

            ################################################################
            # show_ideal_dashes=False + vanilla kernel => extend the SOLID  #
            # lines to span the full kernel duration (don't leave a gap    #
            # where the dashes used to be). We do this at the data layer   #
            # by collapsing End_Time / cb_End_Time / mb_End_Time to        #
            # full_End_Time for vanilla rows when the flag is off. Every   #
            # downstream mark_rule that draws SOLID compute step / memory  #
            # pipe edge pulls these columns, so they automatically extend  #
            # to the kernel's actual end. The dashed-extension code paths  #
            # would have used the same columns to span                     #
            # (cb|mb)_End_Time -> full_End_Time -- those now collapse to   #
            # zero-length and are skipped by the throttled-row filter      #
            # `_gate_throttle_rows` defined further below.                 #
            #                                                              #
            # For transformed kernels (was_transformed=True) we keep the   #
            # original values so the throttle/dilate dashes still show.   #
            # Default (show_ideal_dashes=True) is byte-identical to       #
            # before this branch.                                          #
            ################################################################
            _hide_ideal_for_this_kernel = (
                (not self.show_ideal_dashes)
                and not bool(getattr(kernel, "was_transformed", False))
            )
            if _hide_ideal_for_this_kernel:
                _end_time     = kernel.end
                _cb_end_time  = kernel.end
                _mb_end_time  = kernel.end
            else:
                _end_time     = kernel.end - kernel.throttled_duration
                _cb_end_time  = kernel.end - kernel.cb_throttled_duration
                _mb_end_time  = kernel.end - kernel.mb_throttled_duration

            drawing_data.append({
                phase_legend_name: kernel.name,
                "fusion_group": kernel.fusion_group,
                "fusion_type": getattr(kernel.fusion_type, "name", "NONE"),
                "Starting_Time": kernel_start,
                "Time_Stamp": kernel_start,
                "End_Time": _end_time,
                "cb_End_Time": _cb_end_time,
                "mb_End_Time": _mb_end_time,
                "full_End_Time": kernel.end,
                "runtime": kernel.duration,
                "og_runtime": kernel.duration,
                "Compute_Utilization": kernel.compute_util,
                "Compute_Utilization_plot": cumulative_compute_util,
                "Memory_Utilization": kernel.bw_util,
                "Memory_Utilization_plot": cumulative_bw_util,
                "memory_well": memory_well,
                "cutil_start": cutil_start,
                "mutil_start": cumulative_bw_util - kernel.bw_util,
                "compute_color": kernel.compute_color,
                "bw_color": kernel.bw_color,
                "throttled": kernel.throttled_duration > 0,
                "throttled_duration": kernel.throttled_duration,
                "mem_throttled_duration": kernel.mb_throttled_duration,
                "comp_throttled_duration": kernel.cb_throttled_duration,
                "comp_perc": kernel.comp_perc,
                "bw_perc": kernel.bw_perc,
                "compute_type": kernel.compute_type,
                "memory_type": kernel.memory_type,
            })

          
            tooltip_fields = [phase_legend_name, 'runtime', 'og_runtime', 
                              'Memory_Utilization',
                              'Compute_Utilization', 
                              'Compute_Utilization_plot',
                              'fusion_group', 
                              'memory_well',
                              'full_End_Time',
                              'throttled_duration', 
                              'mem_throttled_duration',
                              'comp_throttled_duration',
                              'comp_perc',
                              'bw_perc',
                              'compute_type',
                              'memory_type'
                             ]
                          
            if kernel.throttled_duration > 0:
                # Base (legacy) throttle-line dict. The "y" here is the
                # pre-band GLOBAL cumulative util -- correct for the legacy
                # single-resource path. On the multi_resource path this y is
                # OVERWRITTEN later (FIX C) with the band-relative y so the
                # dashed remainder is colinear with the solid ideal line.
                _throttle_row = {
                    phase_legend_name: kernel.name,
                    "x": kernel.end - kernel.throttled_duration,
                    "x2": kernel.end,
                    "y": cumulative_compute_util,
                    "color": kernel.compute_color
                }
                # Only on the opt-in multi_resource path do we tag the row
                # with the df row index `i` so FIX C can look this kernel's
                # FIXED band-relative compute fill-top up after the band
                # geometry is computed. The legacy path never adds this key,
                # so its throttle-data spec stays byte-identical.
                if self.multi_resource:
                    _throttle_row["mr_row_idx"] = i
                # Tag with the source kernel's transform status; used by the
                # show_ideal_dashes filter, stripped before the row hits the
                # Altair chart so the spec stays byte-identical by default.
                _throttle_row["_was_transformed"] = bool(getattr(kernel, "was_transformed", False))
                throttled_lines.append(_throttle_row)

            if kernel.mb_throttled_duration > 0 :
                # Memory-side dashed throttle remainder. Legacy y is the
                # negated global cumulative bw util; multi_resource path
                # re-points it to the negated band-relative memory fill-top.
                _mem_throttle_row = {
                    phase_legend_name: kernel.name,
                    "x": kernel.end - kernel.mb_throttled_duration,
                    "x2": kernel.end,
                    "y": -cumulative_bw_util,
                    "color": kernel.compute_color
                }
                if self.multi_resource:
                    _mem_throttle_row["mr_row_idx"] = i
                _mem_throttle_row["_was_transformed"] = bool(getattr(kernel, "was_transformed", False))
                mem_throttled_lines.append(_mem_throttle_row)

            if kernel.cb_throttled_duration > 0 :
                # Compute-side dashed throttle remainder. Legacy y is the
                # global cumulative compute util; multi_resource path
                # re-points it to the band-relative compute fill-top.
                _comp_throttle_row = {
                    phase_legend_name: kernel.name,
                    "x": kernel.end - kernel.cb_throttled_duration,
                    "x2": kernel.end,
                    "y": cumulative_compute_util,
                    "color": kernel.compute_color
                }
                if self.multi_resource:
                    _comp_throttle_row["mr_row_idx"] = i
                _comp_throttle_row["_was_transformed"] = bool(getattr(kernel, "was_transformed", False))
                comp_throttled_lines.append(_comp_throttle_row)

            if cumulative_bw_util > 1.0:
                #print(f"{kernel.start:.2f}: Bandwidth overflow ({cumulative_bw_util:.2f})")
                cumulative_bw_util = 1.0

        df = pd.DataFrame(drawing_data)
        #print(f"Einsum order is {einsum_order}")
        self.einsum_order = einsum_order or df[phase_legend_name].unique().tolist()

        if color_scale is None:
          color_scale = generate_color_scale(self.einsum_order)
        self.update_color_scale(color_scale, tint=alpha)


        ########################################################################
        # Phase 2 (grey-band rev): SOLID GREY BAND geometry (opt-in only)      #
        #                                                                      #
        # Everything in this block is gated behind self.multi_resource so the  #
        # single-resource regression snapshots are byte-identical (the legacy  #
        # path never adds these columns and never reaches the new rendering).  #
        #                                                                      #
        # GOAL (corrected): every distinct compute sub-resource gets ONE       #
        # FIXED GLOBAL band -- the SAME [bottom, top] interval across the      #
        # ENTIRE x-axis (it does NOT re-stack per time interval). The bands    #
        # are stacked CUMULATIVELY.                                            #
        #                                                                      #
        # ORDER (Phase 2 rev3 FIX A): if the caller supplied a ResourceRegistry #
        # (via the constructor or the draw kwarg) the stacking order is the    #
        # registry's DECLARATION order -- the insertion order of the           #
        # registry._by_name dict (i.e. the order reg.add_compute(...) /        #
        # reg.add_memory(...) were called), filtered to COMPUTE-typed          #
        # resources for the compute bands and MEMORY-typed for the memory      #
        # bands, and the band HEIGHTS are the registry resources'              #
        # authoritative attrs["p_of_tot"]. If NO registry is supplied we fall  #
        # back to the legacy behavior: order each sub-resource by the order it #
        # FIRST APPEARS while scanning the cascade's kernels in the user's     #
        # INPUT ORDER (self.cascade.kernels), with the per-kernel comp_perc /  #
        # bw_perc as the height. Either way, only resources actually used by   #
        # some kernel get a band.                                              #
        #                                                                      #
        # For compute sub-resource i in that order:                            #
        #   band_bottom_i = sum(p_of_tot_j for every j BEFORE i in the order)  #
        #   band_top_i    = band_bottom_i + p_of_tot_i      (roof level)       #
        # where p_of_tot_i == the constant comp_perc carried by any kernel     #
        # that uses sub-resource i. The whole band [band_bottom, band_top] is  #
        # painted an OPAQUE per-resource GREY; each kernel ROW's coloured fill #
        # is then drawn OPAQUE ON TOP of the grey, stacking WITHIN the band    #
        # when concurrent on the same resource. (Compute_Utilization is        #
        # already the pool-relative util in the df, == raw_util * p_of_tot.)   #
        # The unfilled remainder [fill_top, band_top] shows the GREY (there is #
        # no white gap anymore).                                               #
        #                                                                      #
        # The memory side is the exact mirror on the NEGATIVE axis using       #
        # bw_perc as the per-resource p_of_tot and Memory_Utilization as the   #
        # fill height (we store positive magnitudes here; the negative sign    #
        # is applied in the encoding via transform_calculate, matching the     #
        # legacy memory area which already negates there).                     #
        #                                                                      #
        # The default single-resource pools ("Compute Pool" / "Memory Pool")  #
        # are EXCLUDED from the ordered resource list unless a kernel actually #
        # uses them, so a real multi-resource cascade is never polluted by an  #
        # unused default band.                                                 #
        ########################################################################
        if self.multi_resource:
            # ----------------------------------------------------------------
            # STEP A: derive the ordered sub-resource lists + their band
            # heights (p_of_tot).
            #
            # Two ordering sources, chosen by whether a ResourceRegistry was
            # resolved above (self._effective_resource_registry):
            #
            #   (A1) REGISTRY ORDER  -- the user requirement (FIX A):
            #        Walk registry._by_name in its insertion (declaration)
            #        order. That dict is ordered by reg.add_compute(...) /
            #        reg.add_memory(...) call order (Python 3.7+ dict). Take
            #        COMPUTE-typed resources (rtype == ResourceType.COMPUTE)
            #        for the compute stack and MEMORY-typed ones for the
            #        memory stack, using each Resource's authoritative
            #        attrs["p_of_tot"] as the band height. We READ the
            #        registry only -- resource.py is never mutated.
            #
            #   (A2) LEGACY ORDER (fallback, no registry) -- back-compat:
            #        Scan self.cascade.kernels (the user's construction order)
            #        once; first appearance of a compute_type / memory_type
            #        fixes its position, with the per-kernel comp_perc /
            #        bw_perc as the height. This is the exact pre-rev3
            #        behavior, so omitting a registry changes nothing.
            #
            # In BOTH cases the default single-resource pool names are
            # excluded unless a kernel genuinely uses them, and a resource is
            # only stacked if some kernel in the cascade actually uses it.
            # ----------------------------------------------------------------
            # Ordered list of distinct compute sub-resource names (final
            # stacking order) and a name -> p_of_tot (band height) map.
            comp_resource_order = []
            comp_p_of_tot = {}
            # Independent ordered list / map for the memory side.
            mem_resource_order = []
            mem_p_of_tot = {}
            # name -> accumulative? Only ACCUMULATIVE (pooled) resources carry a
            # meaningful peak-share in p_of_tot; for a non-accumulative level
            # p_of_tot is just its equal 1/N slot height, which is NOT a pool
            # fraction, so its value must not be shown on the roof label.
            # Compute defaults accumulative, memory non-accumulative (matches
            # the band-drawing defaults above/below).
            comp_accum = {}
            mem_accum = {}

            # The default single-resource pool names we exclude unless used
            # (these come from kernel.py's default_comp / default_mem).
            _default_comp_pool_name = "Compute Pool"
            _default_mem_pool_name = "Memory Pool"

            # First, work out WHICH compute / memory resources are actually
            # used by at least one kernel in the cascade (so the registry
            # path never stacks a declared-but-unused resource, and so we
            # know whether a default pool is genuinely in play). We also
            # collect the per-kernel comp_perc / bw_perc here for the legacy
            # fallback height lookup.
            used_comp_names = set()   # compute_type values seen on a kernel
            used_mem_names = set()    # memory_type values seen on a kernel
            kernel_comp_perc = {}     # name -> comp_perc (legacy height)
            kernel_bw_perc = {}       # name -> bw_perc   (legacy height)
            for input_kernel in self.cascade.kernels:
                # PATH c: a maps kernel drives SEVERAL sub-resources at once,
                # named by its compute_subutils / memory_subutils keys (its
                # scalar compute_type/memory_type are just the default pools).
                # Pull the band names + their p_of_tot from the maps' resolved
                # fills so every sub-resource earns a band. Scalar kernels keep
                # the original single-compute_type / single-memory_type path.
                _cmap = getattr(input_kernel, "compute_subutils", None)
                if _cmap:
                    for _sub, (_su, _sp, _sf) in input_kernel.compute_subfills.items():
                        used_comp_names.add(_sub)
                        if _sub not in kernel_comp_perc:
                            kernel_comp_perc[_sub] = float(_sp)
                else:
                    ck_name = input_kernel.compute_type
                    used_comp_names.add(ck_name)
                    if ck_name not in kernel_comp_perc:
                        kernel_comp_perc[ck_name] = float(input_kernel.comp_perc)
                _mmap = getattr(input_kernel, "memory_subutils", None)
                if _mmap:
                    for _sub, (_su, _sp, _sf) in input_kernel.memory_subfills.items():
                        used_mem_names.add(_sub)
                        if _sub not in kernel_bw_perc:
                            kernel_bw_perc[_sub] = float(_sp)
                else:
                    mk_name = input_kernel.memory_type
                    used_mem_names.add(mk_name)
                    if mk_name not in kernel_bw_perc:
                        kernel_bw_perc[mk_name] = float(input_kernel.bw_perc)

            # The resolved registry (may be None -> legacy fallback path).
            _reg = getattr(self, "_effective_resource_registry", None)

            if _reg is not None:
                # ---- (A1) REGISTRY DECLARATION ORDER ----------------------
                # registry._by_name is an insertion-ordered dict whose key
                # order IS the reg.add_compute()/add_memory() call order.
                # We iterate it READ-ONLY.
                for res_name, res_obj in _reg._by_name.items():
                    # Only stack a registry resource if some kernel actually
                    # uses it (covers both compute and memory below).
                    is_compute = (res_obj.rtype == ResourceType.COMPUTE)
                    is_memory = (res_obj.rtype == ResourceType.MEMORY)

                    if is_compute:
                        # Skip the default compute pool unless a kernel uses it.
                        if res_name == _default_comp_pool_name and \
                           res_name not in used_comp_names:
                            continue
                        # A pooled (accumulative) resource only earns a band
                        # when some kernel actually drives it -- an unused
                        # accumulative band has no meaningful empty slot. A
                        # NON-accumulative resource is an equal-split slot that
                        # must be drawn (empty) even if no phase binds it, so
                        # its neighbours keep their fixed 1/N positions instead
                        # of collapsing upward. Compute defaults accumulative,
                        # so this is byte-identical for every existing caller.
                        _accum = bool(res_obj.attrs.get("accumulative", True))
                        if _accum and res_name not in used_comp_names:
                            continue
                        if res_name not in comp_p_of_tot:
                            # AUTHORITATIVE height = registry p_of_tot attr.
                            comp_p_of_tot[res_name] = float(
                                res_obj.attrs.get("p_of_tot", 1.0))
                            comp_accum[res_name] = _accum
                            comp_resource_order.append(res_name)

                    elif is_memory:
                        # Mirror logic for memory-typed registry resources.
                        if res_name == _default_mem_pool_name and \
                           res_name not in used_mem_names:
                            continue
                        # Memory defaults NON-accumulative (a hierarchy of
                        # levels): every declared level keeps its equal-split
                        # slot, drawn empty in phases -- or a whole cascade --
                        # that never bind it. Only an explicitly accumulative
                        # (pooled) memory resource is dropped when unused.
                        _accum = bool(res_obj.attrs.get("accumulative", False))
                        if _accum and res_name not in used_mem_names:
                            continue
                        if res_name not in mem_p_of_tot:
                            mem_p_of_tot[res_name] = float(
                                res_obj.attrs.get("p_of_tot", 1.0))
                            mem_accum[res_name] = _accum
                            mem_resource_order.append(res_name)

                # Defensive: if a kernel uses a resource that is somehow NOT
                # in the registry, still give it a band (appended AFTER the
                # declared ones) using the per-kernel perc, so it is never
                # silently dropped from the figure. PATH c: a maps kernel's
                # real sub-resources are the map keys (its compute_type /
                # memory_type are just the default pools, which must NOT earn
                # a band), so read them from the maps' resolved fills.
                for input_kernel in self.cascade.kernels:
                    _cmap = getattr(input_kernel, "compute_subutils", None)
                    if _cmap:
                        _comp_pairs = [(n, p) for n, (u, p, f)
                                       in input_kernel.compute_subfills.items()]
                    else:
                        _comp_pairs = [(input_kernel.compute_type,
                                        float(input_kernel.comp_perc))]
                    for ck_name, ck_perc in _comp_pairs:
                        if ck_name != _default_comp_pool_name and \
                           ck_name not in comp_p_of_tot:
                            comp_p_of_tot[ck_name] = float(ck_perc)
                            comp_accum[ck_name] = True   # compute defaults pooled
                            comp_resource_order.append(ck_name)
                    _mmap = getattr(input_kernel, "memory_subutils", None)
                    if _mmap:
                        _mem_pairs = [(n, p) for n, (u, p, f)
                                      in input_kernel.memory_subfills.items()]
                    else:
                        _mem_pairs = [(input_kernel.memory_type,
                                       float(input_kernel.bw_perc))]
                    for mk_name, mk_perc in _mem_pairs:
                        if mk_name != _default_mem_pool_name and \
                           mk_name not in mem_p_of_tot:
                            mem_p_of_tot[mk_name] = float(mk_perc)
                            mem_accum[mk_name] = False   # memory defaults hierarchy
                            mem_resource_order.append(mk_name)
            else:
                # ---- (A2) LEGACY KERNEL-FIRST-APPEARANCE ORDER ------------
                # Exact pre-rev3 behavior: scan the user's INPUT-ORDER kernel
                # list once; first appearance fixes the stacking position.
                for input_kernel in self.cascade.kernels:
                    # PATH c: name the bands from the maps when present (the
                    # scalar compute_type/memory_type are the default pools).
                    _cmap = getattr(input_kernel, "compute_subutils", None)
                    if _cmap:
                        _comp_pairs = [(n, p) for n, (u, p, f)
                                       in input_kernel.compute_subfills.items()]
                    else:
                        _comp_pairs = [(input_kernel.compute_type,
                                        float(input_kernel.comp_perc))]
                    # -- compute sub-resource first-appearance bookkeeping --
                    for ck_name, ck_perc in _comp_pairs:
                        if ck_name not in comp_p_of_tot:
                            # comp_perc IS the p_of_tot fraction for this kernel.
                            comp_p_of_tot[ck_name] = float(ck_perc)
                            comp_accum[ck_name] = True   # compute defaults pooled
                            # Only stack a real named sub-resource (skip default).
                            if ck_name != _default_comp_pool_name:
                                comp_resource_order.append(ck_name)

                    _mmap = getattr(input_kernel, "memory_subutils", None)
                    if _mmap:
                        _mem_pairs = [(n, p) for n, (u, p, f)
                                      in input_kernel.memory_subfills.items()]
                    else:
                        _mem_pairs = [(input_kernel.memory_type,
                                       float(input_kernel.bw_perc))]
                    # -- memory sub-resource first-appearance bookkeeping --
                    for mk_name, mk_perc in _mem_pairs:
                        if mk_name not in mem_p_of_tot:
                            mem_p_of_tot[mk_name] = float(mk_perc)
                            mem_accum[mk_name] = False   # memory defaults hierarchy
                            if mk_name != _default_mem_pool_name:
                                mem_resource_order.append(mk_name)

            # ----------------------------------------------------------------
            # STEP B: turn the ordered resource lists into FIXED GLOBAL
            # [bottom, top] bands by cumulative summation. These maps are the
            # SAME for every time interval -- the bands never re-stack.
            #
            # Example with the user's rev3 REGISTRY declaration order
            # (compute declared Special1D then Special2D; memory declared
            # FancyMem then DRAM):
            #   COMPUTE order = [Special1D, Special2D]
            #     Special1D p_of_tot = 0.25 -> band [0.00, 0.25]
            #     Special2D p_of_tot = 0.75 -> band [0.25, 1.00]
            #   MEMORY  order = [FancyMem, DRAM]   (declaration order!)
            #     FancyMem p_of_tot = 0.10 -> magnitude [0.00, 0.10]
            #                                  (drawn negative [0,   -0.10])
            #     DRAM     p_of_tot = 0.90 -> magnitude [0.10, 1.00]
            #                                  (drawn negative [-0.10,-1.00])
            # (Without a registry the legacy fallback would instead order
            #  memory by kernel first-appearance, e.g. [DRAM, FancyMem].)
            # ----------------------------------------------------------------
            # Map: compute resource name -> (fixed global band_bottom, band_top)
            comp_band_of_resource = {}
            # Running cumulative bottom as we stack the compute resources in
            # the user's input-appearance order.
            comp_running_bottom = 0.0
            for resource_name in comp_resource_order:
                this_height = comp_p_of_tot[resource_name]      # == p_of_tot
                band_bottom = comp_running_bottom
                band_top = band_bottom + this_height
                comp_band_of_resource[resource_name] = (band_bottom, band_top)
                comp_running_bottom = band_top

            # Map: memory resource name -> (fixed |band_bottom|, |band_top|).
            # We store positive magnitudes; the encoding negates them so the
            # memory bands grow downward from zero just like the legacy area.
            mem_band_of_resource = {}
            mem_running_bottom = 0.0
            for resource_name in mem_resource_order:
                this_height = mem_p_of_tot[resource_name]       # == p_of_tot
                band_bottom = mem_running_bottom
                band_top = band_bottom + this_height
                mem_band_of_resource[resource_name] = (band_bottom, band_top)
                mem_running_bottom = band_top

            ################################################################
            # CHANGE 2: assign each sub-resource a DISTINCT SOLID GREY     #
            #                                                              #
            # The old "white gap" is replaced by an OPAQUE grey band       #
            # background per resource. We assign greys from an evenly       #
            # spaced ramp (darkest #9a9a9a -> lightest #cccccc) in the     #
            # SAME registry-declaration order already used for band         #
            # stacking (comp_resource_order / mem_resource_order). The     #
            # FIRST resource gets the DARKEST grey; later resources get     #
            # progressively lighter greys. The MEMORY side uses an          #
            # INDEPENDENT ramp -- it RESTARTS the same dark->light          #
            # sequence from the beginning for the memory resources (memory  #
            # greys do NOT continue from / depend on the compute greys).    #
            # _grey_ramp() handles the N==1 case (dark end only) and        #
            # generalizes to any N.                                         #
            ################################################################
            # Compute-side ramp: one grey per compute resource, in order.
            _comp_ramp = _grey_ramp(len(comp_resource_order))
            comp_grey_of_resource = {
                rname: _comp_ramp[i]
                for i, rname in enumerate(comp_resource_order)
            }
            # Memory-side ramp: a SEPARATE, independent dark->light ramp
            # restarted from the beginning for the memory resources.
            _mem_ramp = _grey_ramp(len(mem_resource_order))
            mem_grey_of_resource = {
                rname: _mem_ramp[i]
                for i, rname in enumerate(mem_resource_order)
            }
            # Stash on the instance so the rendering block + the combined
            # "Resources" legend (CHANGE 3) can read the exact assignments.
            self._mr_comp_resource_order = list(comp_resource_order)
            self._mr_mem_resource_order = list(mem_resource_order)
            self._mr_comp_p_of_tot = dict(comp_p_of_tot)
            self._mr_mem_p_of_tot = dict(mem_p_of_tot)
            # Per-resource accumulative flag -> the roof label only shows the
            # p_of_tot number for accumulative (pooled) resources; a non-
            # accumulative slot's height is not a pool share, so it is hidden.
            self._mr_comp_accum = dict(comp_accum)
            self._mr_mem_accum = dict(mem_accum)
            self._mr_comp_grey_of_resource = dict(comp_grey_of_resource)
            self._mr_mem_grey_of_resource = dict(mem_grey_of_resource)

            # ----------------------------------------------------------------
            # STEP C: project the FIXED global bands onto every df row. Each
            # row (one Einsum running on one resource during one interval)
            # simply looks up its resource's fixed band -- no per-interval
            # restacking. The coloured fill rises from the fixed band_bottom
            # by exactly Compute_Utilization (the pool-relative df value);
            # when comp_perc == 0 the fill has zero height (guard).
            # ----------------------------------------------------------------
            # Per-row geometry columns we will attach to the df below.
            comp_band_bottom = [0.0] * len(df)   # fixed band bottom y
            comp_band_top    = [0.0] * len(df)   # mini-roof y (== fixed band top)
            comp_fill_bot    = [0.0] * len(df)   # coloured-fill BOTTOM in band
            comp_fill_top    = [0.0] * len(df)   # coloured-fill top y in the band
            mem_band_bottom  = [0.0] * len(df)   # |fixed band bottom| (magnitude)
            mem_band_top     = [0.0] * len(df)   # |mini-roof| (magnitude)
            mem_fill_bot     = [0.0] * len(df)   # |fill bottom| (magnitude)
            mem_fill_top     = [0.0] * len(df)   # |fill top| (magnitude)
            # Per-row OPAQUE grey for this row's compute / memory resource
            # band background (CHANGE 2). Same grey for every row that runs
            # on the same resource (it is a property of the resource, not the
            # kernel) so the grey rect reads as one continuous band colour.
            # CHANGE 2: placeholder default matches the new (lightened) DARK
            # ramp end (#9a9a9a); every row is overwritten below from the
            # per-resource ramp assignment, so this is only a defensive
            # initializer (kept in sync with _GREY_RAMP_DARK).
            comp_grey         = [_GREY_RAMP_DARK] * len(df)
            mem_grey          = [_GREY_RAMP_DARK] * len(df)

            ############################################################
            # WITHIN-BAND CUMULATIVE STACKING (CHANGE 2)               #
            #                                                          #
            # When several Einsums are concurrently active on the SAME #
            # compute (or memory) resource -- e.g. after pipelining -- #
            # their coloured fills must STACK cumulatively WITHIN that  #
            # resource's grey band, not all start at band_bottom (which #
            # would overdraw). We key concurrency by the kernel's       #
            # parallel-start group (Starting_Time) AND the resource     #
            # name, accumulating the fill height of earlier same-group  #
            # same-resource rows so each row's fill begins at the       #
            # cumulative bottom and rises by its own pool-relative      #
            # utilization. Rows are processed in self.kernels order     #
            # (== df row order), which is the same order the legacy     #
            # global cutil_start stacking used, so the stack ordering   #
            # is consistent. The dict resets implicitly per group       #
            # because the (start, resource) key embeds the group.       #
            ############################################################
            # (start_time, compute_resource) -> cumulative fill so far.
            _comp_stack_acc = {}
            # (start_time, memory_resource) -> cumulative fill so far.
            _mem_stack_acc = {}

            ################################################################
            # PATH c: per-(phase, sub-resource) coloured fill rows.        #
            #                                                              #
            # A maps kernel drives SEVERAL compute bands (Tensor + FMA)    #
            # and SEVERAL memory levels (L2 + DRAM) at once, so it cannot  #
            # be one rect per df row. We collect one fill row per          #
            # (kernel, sub-resource) here and draw them as extra layers;   #
            # the per-row AGGREGATE rect is suppressed (zero height) while #
            # its fill-top still carries the phase-level line height so    #
            # exactly ONE solid + ONE dashed compute/memory line is drawn  #
            # per phase (from cb_/mb_throttled_duration, unchanged).       #
            # Scalar (maps-free) rows are byte-identical to before.        #
            ################################################################
            self._mr_sub_comp_fills = []   # per (kernel, compute sub-resource)
            self._mr_sub_mem_fills = []     # per (kernel, memory sub-resource)
            # Pooled compute aggregate stacked across concurrent maps phases,
            # keyed by start time -> the phase-level solid/dashed compute line.
            _comp_agg_acc = {}

            for row_index in range(len(df)):
                _row = df.iloc[row_index]
                # The parallel-start group key (kernels sharing this start
                # value are concurrent in the legacy stacking model).
                _start_key = float(_row["Starting_Time"])

                # The source kernel: carries the optional per-sub-resource maps
                # and their resolved pool fills (compute_subfills/memory_subfills).
                _krn = self.kernels[row_index]
                _has_comp_map = bool(getattr(_krn, "compute_subutils", None))
                _has_mem_map = bool(getattr(_krn, "memory_subutils", None))

                # -- compute band for this row's compute resource --
                c_res = _row["compute_type"]
                if _has_comp_map:
                    # PATH c: one coloured fill per compute sub-resource, each
                    # WITHIN its own fixed pooled band, with concurrent
                    # same-sub-resource stacking (reuse _comp_stack_acc).
                    for _sub, (_su, _sp, _sfill) in _krn.compute_subfills.items():
                        _sb, _st = comp_band_of_resource.get(_sub, (0.0, 0.0))
                        _key = (_start_key, _sub)
                        _prev = _comp_stack_acc.get(_key, 0.0)
                        self._mr_sub_comp_fills.append({
                            phase_legend_name: _row[phase_legend_name],
                            "x": float(_row["Starting_Time"]),
                            "x2": float(_row["full_End_Time"]),
                            "y": _sb + _prev,
                            "y2": _sb + _prev + _sfill,
                        })
                        _comp_stack_acc[_key] = _prev + _sfill
                    # Pooled compute aggregate (SUM of the sub-fills ==
                    # Compute_Utilization) is where the ONE phase-level solid +
                    # dashed compute line sits; stack it across concurrent
                    # phases so pipelined maps kernels read cumulatively.
                    _agg_prev = _comp_agg_acc.get(_start_key, 0.0)
                    _agg_top = _agg_prev + float(_row["Compute_Utilization"])
                    _comp_agg_acc[_start_key] = _agg_top
                    comp_band_bottom[row_index] = _agg_prev
                    comp_band_top[row_index]    = _agg_top
                    # Zero-height AGGREGATE rect (suppressed -- the per-band
                    # fills above draw the colour); line height == _agg_top.
                    comp_fill_bot[row_index]    = _agg_top
                    comp_fill_top[row_index]    = _agg_top
                    comp_grey[row_index]        = _GREY_RAMP_DARK
                else:
                    # Look up the FIXED global band for this resource. Fall back
                    # to a degenerate zero band if (defensively) unseen.
                    c_bottom, c_top = comp_band_of_resource.get(c_res, (0.0, 0.0))
                    # comp_perc is the per-resource p_of_tot; guard zero -> no fill.
                    c_perc = float(_row["comp_perc"])
                    if c_perc == 0.0:
                        c_fill_height = 0.0
                    else:
                        # Compute_Utilization is already raw_util * p_of_tot.
                        c_fill_height = float(_row["Compute_Utilization"])
                    # Cumulative offset of earlier concurrent same-resource rows.
                    _c_key = (_start_key, c_res)
                    _c_prev = _comp_stack_acc.get(_c_key, 0.0)
                    comp_band_bottom[row_index] = c_bottom
                    comp_band_top[row_index]    = c_top                    # mini-roof
                    # Fill stacks ON TOP of any earlier concurrent same-resource
                    # fill, but always WITHIN this resource's fixed grey band.
                    comp_fill_bot[row_index]    = c_bottom + _c_prev
                    comp_fill_top[row_index]    = c_bottom + _c_prev + c_fill_height
                    comp_grey[row_index]        = comp_grey_of_resource.get(
                        c_res, _GREY_RAMP_DARK)
                    # Advance the cumulative stack for this group+resource.
                    _comp_stack_acc[_c_key] = _c_prev + c_fill_height

                # -- memory band for this row's memory resource (magnitude) --
                m_res = _row["memory_type"]
                if _has_mem_map:
                    # PATH c: memory levels are a non-accumulative hierarchy --
                    # each level fills its OWN fixed 1/N slot independently
                    # (util_r * slot_height). No cross-level pooling. The ONE
                    # phase-level solid + dashed memory line rides the DOMINANT
                    # (max-gauge, == bw_util_raw) level's slot fill-top, so the
                    # dashed throttle remnant is colinear with that band's fill.
                    _dom_gauge = -1.0
                    _dom_top = 0.0
                    for _sub, (_su, _sp, _sfill) in _krn.memory_subfills.items():
                        _sb, _st = mem_band_of_resource.get(_sub, (0.0, 0.0))
                        _key = (_start_key, _sub)
                        _prev = _mem_stack_acc.get(_key, 0.0)
                        _ftop = _sb + _prev + _sfill
                        self._mr_sub_mem_fills.append({
                            phase_legend_name: _row[phase_legend_name],
                            "x": float(_row["Starting_Time"]),
                            "x2": float(_row["full_End_Time"]),
                            "y": _sb + _prev,
                            "y2": _ftop,
                            # This row's pooled compute line level, so the
                            # COLOCATED pipe can anchor each memory slot fill
                            # just above the compute step-line (dual ignores it).
                            "comp_top": float(comp_fill_top[row_index]),
                        })
                        _mem_stack_acc[_key] = _prev + _sfill
                        if _su > _dom_gauge:
                            _dom_gauge = _su
                            _dom_top = _ftop
                    mem_band_bottom[row_index] = _dom_top
                    mem_band_top[row_index]    = _dom_top
                    # Zero-height AGGREGATE rect (suppressed); line height ==
                    # the dominant level's slot fill-top.
                    mem_fill_bot[row_index]    = _dom_top
                    mem_fill_top[row_index]    = _dom_top
                    mem_grey[row_index]        = _GREY_RAMP_DARK
                else:
                    m_bottom, m_top = mem_band_of_resource.get(m_res, (0.0, 0.0))
                    m_perc = float(_row["bw_perc"])
                    if m_perc == 0.0:
                        m_fill_height = 0.0
                    else:
                        m_fill_height = float(_row["Memory_Utilization"])
                    _m_key = (_start_key, m_res)
                    _m_prev = _mem_stack_acc.get(_m_key, 0.0)
                    mem_band_bottom[row_index] = m_bottom
                    mem_band_top[row_index]    = m_top                     # mini-roof
                    mem_fill_bot[row_index]    = m_bottom + _m_prev
                    mem_fill_top[row_index]    = m_bottom + _m_prev + m_fill_height
                    mem_grey[row_index]        = mem_grey_of_resource.get(
                        m_res, _GREY_RAMP_DARK)
                    _mem_stack_acc[_m_key] = _m_prev + m_fill_height

            # Attach the computed geometry as plain df columns so the Altair
            # encodings below can reference them directly. We DELIBERATELY use
            # new column names so nothing the legacy path reads is touched.
            df["mr_comp_band_bottom"] = comp_band_bottom
            df["mr_comp_band_top"]    = comp_band_top
            df["mr_comp_fill_bot"]    = comp_fill_bot
            df["mr_comp_fill_top"]    = comp_fill_top
            df["mr_mem_band_bottom"]  = mem_band_bottom
            df["mr_mem_band_top"]     = mem_band_top
            df["mr_mem_fill_bot"]     = mem_fill_bot
            df["mr_mem_fill_top"]     = mem_fill_top
            df["mr_comp_grey"]        = comp_grey
            df["mr_mem_grey"]         = mem_grey

            ################################################################
            # CHANGE 4: GLOBAL x-domain for FULL-WIDTH roof lines          #
            #                                                              #
            # Each resource's roof line must be a SINGLE horizontal rule   #
            # spanning the ENTIRE x domain of the cascade -- from the      #
            # global minimum start to the global maximum end -- regardless #
            # of where that resource is actually active. We compute those  #
            # two scalars once here from the df (Starting_Time is the      #
            # per-row start, full_End_Time the per-row untruncated end)    #
            # and stash them so the rendering block can build one rule per #
            # resource at its fixed band_top spanning [x_min, x_max].      #
            ################################################################
            # Defensive against an empty df (the multi_resource path always
            # has rows, but never crash if somehow not).
            if len(df) > 0:
                self._mr_x_min = float(df["Starting_Time"].min())
                self._mr_x_max = float(df["full_End_Time"].max())
            else:
                self._mr_x_min = 0.0
                self._mr_x_max = 0.0

            ################################################################
            # FIX C: re-point the DASHED throttle rules onto the SAME      #
            # band-relative y as their SOLID ideal line.                   #
            #                                                              #
            # The solid compute ideal line is drawn at y = mr_comp_fill_top #
            # (fixed band bottom + this kernel's pool-relative util) and    #
            # ends in x at cb_End_Time. The dashed compute throttle remnant #
            # already STARTS in x at cb_End_Time (x == kernel.end -         #
            # cb_throttled_duration) and ends at full_End_Time, so the x's  #
            # are already contiguous. The ONLY thing wrong was its y: it    #
            # still carried the pre-band GLOBAL cumulative util while the   #
            # solid line had been re-pointed into band coords -- leaving    #
            # the two segments disjoint. Here we overwrite each dashed      #
            # row's y with the EXACT same band-relative fill-top of its     #
            # df row (looked up via the mr_row_idx tag), so solid + dashed  #
            # read as ONE continuous polyline at the correct band height.   #
            # Memory mirrors onto the negative axis (y = -mr_mem_fill_top). #
            #                                                              #
            # comp_fill_top / mem_fill_top are positional lists indexed by  #
            # the SAME row order as self.kernels (drawing_data was appended #
            # in that order), and mr_row_idx is exactly that loop index, so #
            # comp_fill_top[mr_row_idx] is this kernel's solid-line y.      #
            ################################################################
            # Compute-side: solid ends at cb_End_Time at y =
            # comp_fill_top[idx]; dashed continues at the SAME y.
            for _row in comp_throttled_lines:
                _idx = _row["mr_row_idx"]
                _row["y"] = comp_fill_top[_idx]
            # The legacy single-resource path also emits `throttled_lines`
            # (used only when there is NO comp-side throttle). Re-point it
            # too so any multi_resource figure that hits that branch stays
            # colinear; harmless when comp_throttled_lines is non-empty.
            for _row in throttled_lines:
                _idx = _row["mr_row_idx"]
                _row["y"] = comp_fill_top[_idx]
            # Memory-side: solid ends at mb_End_Time at y =
            # -mem_fill_top[idx]; dashed continues at the SAME negated y.
            for _row in mem_throttled_lines:
                _idx = _row["mr_row_idx"]
                _row["y"] = -mem_fill_top[_idx]

            ################################################################
            # COLOCATED MULTI-RESOURCE GEOMETRY (CHANGE 3)                 #
            #                                                              #
            # Everything ABOVE in this `if self.multi_resource:` block was #
            # written for the DUAL view (compute on +Y, memory mirrored on #
            # -Y) and is left UNTOUCHED -- the dual MR path is final.      #
            #                                                              #
            # Here we additionally derive the COLOCATED-mode geometry. It  #
            # is only ever READ by the colocated rendering branch          #
            # (`else: #colocated mode`, gated again on self.multi_resource) #
            # so computing these extra columns is harmless to the dual     #
            # path, and the WHOLE block is still inside                    #
            # `if self.multi_resource:` so multi_resource=False adds NO    #
            # columns and stays byte-identical.                            #
            #                                                              #
            # THE COLOCATED MEMORY "PIPE"                                  #
            # --------------------------                                   #
            # In single-resource colocated mode the memory used-bw is a    #
            # rect [y1, y2] living in a uniform light-grey "pipe" of plot  #
            # height `bw_util_scaling * memory_well`, anchored to the      #
            # compute line per guard_position (see draw_guard_bands).      #
            #                                                              #
            # For multi-resource we make the pipe represent 100% of        #
            # memory bandwidth: a FIXED plot height == `bw_util_scaling`   #
            # (full bw 1.0 scaled by the same bw_util_scaling factor the   #
            # single-resource fill uses), anchored to the compute line     #
            # exactly as before per guard_position. The pipe is then       #
            # PERSISTENTLY PARTITIONED among ALL memory sub-resources by   #
            # p_of_tot in registry-declaration order (segment i spans      #
            # pipe fraction [Sum p_of_tot before i, Sum p_of_tot <= i]),   #
            # each segment shaded its own grey from the SAME independent   #
            # memory ramp the dual path built (mem_grey_of_resource).      #
            #                                                              #
            # The Einsum COLOURED used-bw fill for a kernel must START at  #
            # the BOTTOM of ITS memory sub-resource's segment (offset by   #
            # the cumulative p_of_tot of all memory resources declared     #
            # BEFORE that kernel's memory_type) and extend by the kernel's #
            # pool-relative Memory_Utilization, scaled into pipe space by  #
            # the SAME bw_util_scaling factor. We REUSE the dual path's    #
            # already-computed pipe-fraction lists mem_fill_bot /          #
            # mem_fill_top (which are EXACTLY [seg_bottom + cumulative,    #
            # seg_bottom + cumulative + Memory_Utilization] in p_of_tot/bw #
            # fraction space, including within-segment cumulative stacking #
            # for concurrent same-resource kernels) and merely PROJECT     #
            # them into the colocated pipe's plot Y space.                 #
            #                                                              #
            # WORKED EXAMPLE (the spec's, reproduced exactly):             #
            #   memory registry order FancyMem(0.10) then DRAM(0.90).      #
            #   pipe == 100% bw. Segments (pipe fraction):                  #
            #     FancyMem -> [0.00, 0.10]  (grey #5a5a5a, darkest first)   #
            #     DRAM     -> [0.10, 1.00]  (grey #bdbdbd, lightest last)   #
            #   EinsumA uses DRAM, pool-relative Memory_Utilization=0.135.  #
            #   mr_colo_mem_seg_bottom(EinsumA) == 0.10 (cumulative p_of_  #
            #   tot of memory resources BEFORE DRAM == FancyMem 0.10).      #
            #   Coloured fill pipe fraction == [0.10, 0.235] (NOT          #
            #   [0.00, 0.135]). FancyMem [0,0.10] stays dark grey; DRAM     #
            #   unused [0.235, 1.0] stays light grey.                       #
            ################################################################
            # The pipe height in PLOT units == bw_util_scaling (full bw
            # 1.0 * bw_util_scaling). NOTE: the legacy compute_limit_*_expr
            # strings hardcode 0.125; bw_util_scaling defaults to 0.125 and
            # the only MR caller leaves it default, so the colocated MR
            # pipe lines up with the same visual scale.
            _pipe_h = float(bw_util_scaling)

            #####################################################################
            # COMPUTE-BAND-OFFSET FIX (colocated MR pipe base)                  #
            #                                                                   #
            # In colocated multi-resource mode the compute step-line is NOT the #
            # raw pool-relative cumulative util measured from absolute 0        #
            # (Compute_Utilization_plot); it is that util OFFSET upward by the  #
            # compute sub-resource's band bottom -- exactly the value the dual  #
            # path already computed and stored in `mr_comp_fill_top`           #
            # (band_bottom + within-band cumulative pool-relative compute util, #
            # including stacking for concurrent same-resource Einsums). The     #
            # memory "pipe" must ride just above THAT offset compute line, so   #
            # the pipe BASE has to be anchored to `mr_comp_fill_top`, NOT to    #
            # the raw Compute_Utilization_plot it used to read. We ONLY move    #
            # the pipe base here; the within-pipe memory partition and the      #
            # mr_colo_mem_seg_bottom segment offsets below are computed from    #
            # mem_fill_bot/mem_fill_top exactly as before (that logic is        #
            # correct and unchanged).                                          #
            #####################################################################
            # Anchor (pipe BASE = the bottom edge of the full pipe) per
            # guard_position, mirroring draw_guard_bands' compute_limit_bot but
            # against the OFFSET compute level (mr_comp_fill_top), not the raw
            # Compute_Utilization_plot:
            #   "above"    -> base = mr_comp_fill_top
            #   "below"    -> base = mr_comp_fill_top - pipe_h
            #   "centered" -> base = mr_comp_fill_top - pipe_h/2
            # so the pipe occupies [base, base + pipe_h] sitting just above the
            # OFFSET compute step-line (which is also drawn at mr_comp_fill_top).
            mr_colo_pipe_base = [0.0] * len(df)   # pipe bottom (plot Y)
            mr_colo_pipe_top  = [0.0] * len(df)   # pipe top    (plot Y)
            # Cumulative p_of_tot (pipe FRACTION, 0..1) of all memory
            # resources declared BEFORE this row's memory_type. This is the
            # per-row offset column the spec asks for (== 0.10 for EinsumA
            # on DRAM in the worked example).
            mr_colo_mem_seg_bottom = [0.0] * len(df)
            # The Einsum coloured used-bw rect, projected into plot Y:
            #   y  = base + mem_fill_bot * pipe_h   (segment-offset bottom)
            #   y2 = base + mem_fill_top * pipe_h   (+ Memory_Utilization)
            mr_colo_mem_fill_y  = [0.0] * len(df)
            mr_colo_mem_fill_y2 = [0.0] * len(df)

            for _ri in range(len(df)):
                _r = df.iloc[_ri]
                # OFFSET compute level for this row (band bottom + within-band
                # cumulative pool-relative compute util). This is the same y
                # the colocated compute step-line is now drawn at, so the pipe
                # stays colinear with the OFFSET compute line.
                _cplot = float(_r["mr_comp_fill_top"])
                if self.guard_position == "below":
                    _base = _cplot - _pipe_h
                elif self.guard_position == "centered":
                    _base = _cplot - _pipe_h / 2.0
                else:  # "above" (default) -- pipe sits ON the compute line
                    _base = _cplot
                mr_colo_pipe_base[_ri] = _base
                mr_colo_pipe_top[_ri]  = _base + _pipe_h

                # This kernel's memory sub-resource segment bottom (pipe
                # fraction) == cumulative p_of_tot of memory resources
                # declared BEFORE it. mem_band_of_resource[name] == the
                # FIXED (band_bottom, band_top) magnitudes the dual path
                # already built in registry-declaration order -- band_bottom
                # IS exactly that "cumulative p_of_tot before" offset.
                _m_res = _r["memory_type"]
                _seg_b, _seg_t = mem_band_of_resource.get(_m_res, (0.0, 0.0))
                mr_colo_mem_seg_bottom[_ri] = _seg_b

                # REUSE the dual path's pipe-fraction fill lists. mem_fill_bot
                # == seg_bottom + (cumulative concurrent same-resource fill);
                # mem_fill_top == that + this kernel's Memory_Utilization.
                # Project both into plot Y via the pipe base + pipe height.
                mr_colo_mem_fill_y[_ri]  = _base + mem_fill_bot[_ri] * _pipe_h
                mr_colo_mem_fill_y2[_ri] = _base + mem_fill_top[_ri] * _pipe_h

            # Attach as NEW df columns (distinct names; the legacy/dual
            # paths never read them so nothing they touch changes).
            df["mr_colo_pipe_base"]      = mr_colo_pipe_base
            df["mr_colo_pipe_top"]       = mr_colo_pipe_top
            df["mr_colo_mem_seg_bottom"] = mr_colo_mem_seg_bottom
            df["mr_colo_mem_fill_y"]     = mr_colo_mem_fill_y
            df["mr_colo_mem_fill_y2"]    = mr_colo_mem_fill_y2

            #####################################################################
            # COMPUTE-BAND-OFFSET FIX (colocated MR compute throttle dashes)    #
            #                                                                   #
            # In the colocated single-resource path the memory-side dashed      #
            # throttle remainder is NOT drawn (it is gated on self.dual_mode).  #
            # The compute throttle dashed rules ARE drawn (comp_throttled_lines #
            # / throttled_lines) and continue the compute step-line past        #
            # cb_End_Time. In colocated MR the compute step-line is now drawn   #
            # at the OFFSET compute level mr_comp_fill_top (band bottom + the    #
            # kernel's within-band cumulative pool-relative util), NOT the raw  #
            # Compute_Utilization_plot. So the dashed remainder must continue   #
            # at that SAME offset level to stay colinear with the (now offset)  #
            # compute step-line. Re-point each colocated throttle row's y to    #
            # mr_comp_fill_top (was Compute_Utilization_plot). The dual path's  #
            # own band-relative re-point earlier is left intact; only the       #
            # colocated (not dual_mode) y is changed here, and the whole block  #
            # is still inside `if self.multi_resource:` so single-resource      #
            # output is byte-identical (never reaches this code).               #
            #####################################################################
            if not self.dual_mode:
                for _row in comp_throttled_lines:
                    _idx = _row["mr_row_idx"]
                    # OFFSET compute level == colocated compute step-line y.
                    _row["y"] = float(
                        df.iloc[_idx]["mr_comp_fill_top"])
                for _row in throttled_lines:
                    _idx = _row["mr_row_idx"]
                    _row["y"] = float(
                        df.iloc[_idx]["mr_comp_fill_top"])


        # Connecting vertical lines between adjacent tiles of the same Einsum
        connector_lines, mem_connector_lines = self.draw_einsum_connections(df)
 
      
        # === Guard band positioning logic
        base = self.draw_guard_bands(df)

        ########################################################################
        # base_dashes: the chart used to drive the colocated DASHED guard-band #
        # extensions only. When show_ideal_dashes is True (default), it just  #
        # aliases `base` -- spec stays byte-identical. When False, we filter  #
        # df to rows whose source kernel had .throttle()/.dilate() applied   #
        # (kernel.was_transformed == True) and rebuild a new base over the   #
        # filtered df, so vanilla kernels' dashed extensions disappear while #
        # transformed kernels' dashes keep drawing.                          #
        ########################################################################
        if self.show_ideal_dashes:
            base_dashes = base
        else:
            # df rows are 1:1 with self.kernels in order, so the boolean mask
            # built from self.kernels indexes df correctly.
            _was_transformed_mask = [bool(getattr(k, "was_transformed", False))
                                     for k in self.kernels]
            _df_dashes = df[_was_transformed_mask].reset_index(drop=True)
            base_dashes = self.draw_guard_bands(_df_dashes)

        # === Tile dividers
        tile_boundaries = self.draw_tile_boundaries(df)
      
        if self.dual_mode:
          mem_tile_boundaries = self.draw_mem_tile_boundaries(df)

        y_scale = alt.Scale(type='log') if log_scale else alt.Scale()
        y_scale = alt.Scale(domain=[y_min, y_max])
        tick_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        y_axis  = alt.Axis(
            title="Compute Utilization",
            values=tick_values,                           # <- no 1.2 tick
            labelExpr="format(datum.value, '.1f')"        # optional; format nicely
        )
      
        if self.dual_mode:
            # === Dual mode mirrored areas

            # Get a dual axes without adding negative numbers
            shared_y_axis = alt.Y(
                'cutil_start:Q',
                title="Memory Utilization | Compute Utilization",
                scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(
                    labelExpr="abs(datum) == 0 ? '0' : format(abs(datum), '.2f')"
                )
            )

            

            compute_area = base.mark_rect(opacity=1).encode(
                x='Starting_Time:Q',
                x2= 'full_End_Time:Q', #'End_Time:Q',
                y = 'cutil_start:Q', 
              # alt.Y(
              #     'cutil_start:Q',
              #     title="Memory Utilization | Compute Utilization",
              #     scale=alt.Scale(domain=[-1, 1]),
              #   ),
                y2='Compute_Utilization_plot:Q',
                color=alt.Color(phase_legend_name+':N', 
                                scale=self.tinted_color_scale,#tint_scale(self.color_scale, tint=alpha), 
                                sort=self.einsum_order,
                                legend=None
                               ),
                tooltip=tooltip_fields
            )

            memory_area = base.mark_rect(opacity=1).encode(
                x='Starting_Time:Q',
                x2='full_End_Time:Q', #'End_Time:Q',
                #y='mutil_start:Q',
                y=alt.Y(
                  'mutil_start:Q',
                  title="Memory Utilization | Compute Utilization",
                  scale=alt.Scale(domain=[-1, 1]),
                   axis=alt.Axis(
                       #labelExpr="datum.value"
                       labelExpr="abs(datum.value) == 0 ? '0' : format(abs(datum.value), '.2f')"
                     )
                ), 
                y2=alt.Y2('neg_mem_util:Q'),
                color=alt.Color(phase_legend_name+':N', 
                                scale=self.tinted_color_scale, 
                                sort=self.einsum_order,
                                legend=None
                               ),                
               tooltip=tooltip_fields
            ).transform_calculate(
                mutil_start='-datum.mutil_start',
                neg_mem_util='-datum.Memory_Utilization_plot'
            )

            compute_lines = base.mark_line(size=line_thickness, interpolate='step-after').encode(
                x='Starting_Time:Q',
                x2='cb_End_Time:Q',
                y='Compute_Utilization_plot:Q',
                color=alt.Color(phase_legend_name+':N', scale=self.color_scale, sort=self.einsum_order, legend=None),
                tooltip=tooltip_fields
            )

            memory_lines = base.mark_line(size=line_thickness, interpolate='step-after').encode(
                x='Starting_Time:Q',
                x2='mb_End_Time:Q',
                y='neg_mem_util:Q',
                color=alt.Color(phase_legend_name+':N', scale=self.color_scale, sort=self.einsum_order, legend=None),
                tooltip=tooltip_fields
            ).transform_calculate(
                neg_mem_util='-datum.Memory_Utilization_plot'
            )

            zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_line(size=2,
                                                                      interpolate='step-after',
                                                                      color="black").encode(y='y:Q')

            ####################################################################
            # Phase 2 (grey-band rev): SOLID GREY BANDS + full-width roofs     #
            #                                                                  #
            # ONLY entered when the caller opted in via multi_resource=True.    #
            # When multi_resource is False this whole block is skipped and the  #
            # compute_area / memory_area / compute_lines / memory_lines objects #
            # defined above are layered exactly as the legacy code did -- which #
            # is what keeps the single-resource regression snapshots            #
            # byte-identical (the hard regression gate).                        #
            #                                                                  #
            # When multi_resource IS set we:                                    #
            #   (0) draw an OPAQUE per-resource GREY rectangle spanning that    #
            #       resource's FIXED band [band_bottom, band_top] -- this       #
            #       REPLACES the old "white gap" (CHANGE 2). Layered FIRST so   #
            #       it sits BEHIND everything in that band.                     #
            #   (1) REPLACE the four dual-mode marks with band-relative         #
            #       versions; the Einsum colour fill is drawn OPAQUE ON TOP of  #
            #       the grey, stacking WITHIN the band when concurrent on the   #
            #       same resource (mr_*_fill_bot -> mr_*_fill_top).             #
            #   (4) draw each resource's roof as a SINGLE full-x-width rule at  #
            #       its fixed band_top (CHANGE 4), plus a BLACK label           #
            #       (CHANGE 1). Layered LAST so it sits ON TOP of the 100%      #
            #       reference lines and stays visible.                          #
            #                                                                  #
            # Net per-band z-order: grey rect (opaque) -> Einsum colour fill    #
            # (opaque) -> roof line + black label.                              #
            #                                                                  #
            # COLOR stays the Einsum identity scale (unchanged). A SEPARATE     #
            # combined "Resources" grey-swatch legend is composed later         #
            # (CHANGE 3); the Einsum colour legend is unchanged.                #
            ####################################################################
            # Layer lists folded into the dual-mode chart. Both empty on the
            # legacy path so layering is a no-op there.
            multi_resource_grey_layers = []   # OPAQUE grey band backgrounds
            multi_resource_extra_layers = []
            if self.multi_resource:
                ############################################################
                # MULTI-RESOURCE Y-DOMAIN PADDING (label headroom)         #
                #                                                          #
                # WHY: on the multi-resource path the cumulative-extreme    #
                # resource roofs land EXACTLY at +1.0 (compute) and -1.0    #
                # (memory). Their BLACK roof labels (e.g. "Special2D       #
                # (0.75)" on the +1.0 compute roof, "DRAM (0.90)" on the    #
                # -1.0 memory roof) are intentionally drawn just ABOVE /    #
                # just BELOW the roof line, which pushes them OUTSIDE a     #
                # [-1,1] frame and the plot frame clips them. We pad ONLY   #
                # the multi-resource y SCALE to [-1.1, 1.1] so that extra   #
                # 0.1 of headroom on each side brings those extreme-band    #
                # labels back INSIDE the visible plot. This is gated        #
                # strictly inside `if self.multi_resource:` so the generic  #
                # dual-mode path (simple_dual / tiled_dual / pipelined_dual #
                # regression cases) keeps its byte-identical [-1,1] domain. #
                #                                                          #
                # The DISPLAYED ticks are NOT padded: we pin an explicit    #
                # tick `values=` set that still tops out at 1.0 / -1.0 (in  #
                # 0.25 steps) so the padded scale gives headroom WITHOUT    #
                # ever showing ugly 1.1 / -1.1 tick labels.                 #
                ############################################################
                _mr_y_scale = alt.Scale(domain=[-1.1, 1.1])
                _mr_y_tick_values = [
                    -1.0, -0.75, -0.5, -0.25,
                    0.0,
                    0.25, 0.5, 0.75, 1.0,
                ]
                _mr_y_axis = alt.Axis(
                    values=_mr_y_tick_values,
                    labelExpr="abs(datum.value) == 0 ? '0' : format(abs(datum.value), '.2f')"
                )
                ############################################################
                # CHANGE 1: PERSISTENT FULL-WIDTH GREY BANDS               #
                #                                                          #
                # Previously the grey band for a resource was drawn ONLY    #
                # over the time intervals where that resource was actively  #
                # used (one rect per active df row, x = Starting_Time ..    #
                # full_End_Time). The requirement is now that EVERY         #
                # resource's grey band is a CONTINUOUS background spanning  #
                # the ENTIRE x-domain of the cascade                        #
                # [global_min_start, global_max_end], at its FIXED          #
                # [band_bottom, band_top], regardless of whether anything   #
                # is using it at a given time (e.g. the Special1D region    #
                # [0,0.25] is shaded grey for the whole timeline even when  #
                # idle). The per-kernel Einsum COLOUR fills (compute_area / #
                # memory_area below) still draw ONLY over active intervals, #
                # ON TOP of this persistent grey.                           #
                #                                                          #
                # IMPLEMENTATION: a SINGLE small data-driven layer per side #
                # (compute / memory). We build ONE DataFrame with exactly   #
                # one row PER RESOURCE (NOT per kernel / per interval, so   #
                # the layer count stays tiny -- it grows only with the      #
                # small number of distinct resources, never with the kernel #
                # count). Each row carries:                                 #
                #     x       = self._mr_x_min   (global min Starting_Time)  #
                #     x2      = self._mr_x_max   (global max full_End_Time)   #
                #     y       = band_bottom      (fixed)                      #
                #     y2      = band_top         (fixed)                      #
                #     resource= <resource name>                              #
                #     grey    = <assigned hex>                               #
                # and is drawn as ONE OPAQUE mark_rect.                      #
                #                                                          #
                # CHANGE 3: the resource->grey mapping is encoded as a REAL #
                # Altair colour scale (domain = resource names, range =     #
                # their assigned grey hexes) WITH its own                   #
                # alt.Legend(title="Resources"). Combined with the existing #
                # .resolve_scale(color='independent') on the layered chart, #
                # this renders the "Resources" legend on the RIGHT margin   #
                # (outside the plot) STACKED with the native Einsum colour  #
                # legend, instead of as a pixel-overlay box over the plot.  #
                #                                                          #
                # Z-order: this persistent grey layer is appended FIRST to  #
                # multi_resource_grey_layers, so it sits BEHIND the Einsum  #
                # colour fills and the roof rules/labels.                   #
                #                                                          #
                # OPACITY: grey STAYS solid/opaque (opacity=1.0). Only the  #
                # Einsum fills become translucent (CHANGE 2, below).        #
                ############################################################
                # One row per COMPUTE resource: fixed band, full x-domain.
                _comp_grey_rows = []
                for _rn in self._mr_comp_resource_order:
                    _b, _t = comp_band_of_resource.get(_rn, (0.0, 0.0))
                    _comp_grey_rows.append({
                        "x":  self._mr_x_min,                 # global x start
                        "x2": self._mr_x_max,                 # global x end
                        "y":  _b,                             # fixed bottom
                        "y2": _t,                             # fixed top
                        "resource": _rn,
                        "grey": self._mr_comp_grey_of_resource.get(
                            _rn, _GREY_RAMP_DARK),
                    })
                # One row per MEMORY resource: fixed band MIRRORED negative.
                _mem_grey_rows = []
                for _rn in self._mr_mem_resource_order:
                    _b, _t = mem_band_of_resource.get(_rn, (0.0, 0.0))
                    _mem_grey_rows.append({
                        "x":  self._mr_x_min,
                        "x2": self._mr_x_max,
                        "y":  -_b,                            # mirror negative
                        "y2": -_t,
                        "resource": _rn,
                        "grey": self._mr_mem_grey_of_resource.get(
                            _rn, _GREY_RAMP_DARK),
                    })
                _comp_grey_df = pd.DataFrame(
                    _comp_grey_rows,
                    columns=["x", "x2", "y", "y2", "resource", "grey"])
                _mem_grey_df = pd.DataFrame(
                    _mem_grey_rows,
                    columns=["x", "x2", "y", "y2", "resource", "grey"])

                # CHANGE 3: build the REAL resource->grey colour scale +
                # legend. The legend lists COMPUTE resources first then
                # MEMORY resources, each in registry-declaration order, with
                # the exact grey assigned to its band. We feed a combined
                # domain/range so BOTH sides share one "Resources" legend
                # (the greys are unique enough across the lightened ramp;
                # compute + memory each restart the ramp independently).
                _res_domain = list(self._mr_comp_resource_order) + \
                    list(self._mr_mem_resource_order)
                _res_range = (
                    [self._mr_comp_grey_of_resource.get(r, _GREY_RAMP_DARK)
                     for r in self._mr_comp_resource_order]
                    + [self._mr_mem_grey_of_resource.get(r, _GREY_RAMP_DARK)
                       for r in self._mr_mem_resource_order]
                )
                _resource_grey_scale = alt.Scale(
                    domain=_res_domain, range=_res_range)
                # A real, legended fill encoding. resolve_scale(
                # color='independent') (already applied to the layered
                # chart) keeps this separate from the Einsum colour scale
                # so both legends render stacked on the RIGHT margin.
                _resource_fill = alt.Fill(
                    'resource:N',
                    scale=_resource_grey_scale,
                    legend=alt.Legend(title="Resources"),
                )

                ##### TWEAK 1: MR-only translucent grey bands #####
                # This whole block is already strictly gated by the
                # enclosing `if self.multi_resource:` (started ~line 1085),
                # so dropping the persistent grey band opacity from a hard
                # 1.0 to 0.65 ONLY affects the multi_resource path. The
                # bands stay clearly grey but a little translucent; the
                # non-multi_resource path never builds these rects and is
                # byte-identical. (Einsum colour fills keep their own ~0.5.)
                # ONE mark_rect for the persistent compute bands
                # (full x-domain, fixed y band). opacity=0.65 -> grey is
                # slightly translucent (MR only) but still clearly grey.
                compute_grey_band = alt.Chart(_comp_grey_df).mark_rect(
                    opacity=0.65
                ).encode(
                    x='x:Q',
                    x2='x2:Q',
                    y='y:Q',
                    y2='y2:Q',
                    fill=_resource_fill,
                )
                ##### TWEAK 1: MR-only translucent grey bands #####
                # Same gating rationale as the compute band above: still
                # inside `if self.multi_resource:`, so opacity=0.65 (a
                # little translucent, still clearly grey) is MR-only.
                # ONE mark_rect for the persistent memory bands,
                # mirrored onto the NEGATIVE axis (y / y2 already negated in
                # the DataFrame). It carries the padded multi-resource y
                # scale + pinned-tick axis so it composes with the rest.
                memory_grey_band = alt.Chart(_mem_grey_df).mark_rect(
                    opacity=0.65
                ).encode(
                    x='x:Q',
                    x2='x2:Q',
                    y=alt.Y(
                        'y:Q',
                        title="Memory Utilization | Compute Utilization",
                        # Multi-resource: padded [-1.1,1.1] scale (label
                        # headroom) but ticks pinned to top out at 1.0/-1.0.
                        scale=_mr_y_scale,
                        axis=_mr_y_axis,
                    ),
                    y2=alt.Y2('y2:Q'),
                    fill=_resource_fill,
                )
                # Layered FIRST (behind the colour fill + roof). This is the
                # persistent full-width grey: ONE rect per resource.
                multi_resource_grey_layers = [
                    compute_grey_band, memory_grey_band,
                ]

                ############################################################
                # PATH c: per-(phase, sub-resource) coloured fills.        #
                #                                                          #
                # A maps kernel's aggregate compute_area / memory_area     #
                # rects are suppressed (zero height); the actual colour is #
                # drawn here, one rect per (phase, sub-resource), each in  #
                # its own fixed band/slot. Layered ON the grey but UNDER   #
                # the solid ideal + dashed lines (added to the grey layer  #
                # list, which composes first). Uses the SAME lightened     #
                # Einsum scale + opacity as the dual aggregate fills, so a  #
                # single-sub-resource maps kernel matches the scalar look. #
                # Only added when some kernel actually carries a map, so    #
                # scalar multi_resource figures are visually unchanged.     #
                ############################################################
                if getattr(self, "_mr_sub_comp_fills", None):
                    _mr_sub_comp_df = pd.DataFrame(
                        self._mr_sub_comp_fills,
                        columns=[phase_legend_name, "x", "x2", "y", "y2"])
                    mr_sub_compute_area = alt.Chart(
                        _mr_sub_comp_df
                    ).mark_rect(opacity=1.0).encode(
                        x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q',
                        color=alt.Color(phase_legend_name+':N',
                                        scale=self.tinted_color_scale,
                                        sort=self.einsum_order, legend=None),
                    )
                    multi_resource_grey_layers.append(mr_sub_compute_area)
                if getattr(self, "_mr_sub_mem_fills", None):
                    _mr_sub_mem_df = pd.DataFrame(
                        self._mr_sub_mem_fills,
                        columns=[phase_legend_name, "x", "x2", "y", "y2"])
                    mr_sub_memory_area = alt.Chart(
                        _mr_sub_mem_df
                    ).mark_rect(opacity=1.0).encode(
                        x='x:Q', x2='x2:Q',
                        y=alt.Y('neg_y:Q', scale=_mr_y_scale, axis=_mr_y_axis),
                        y2=alt.Y2('neg_y2:Q'),
                        color=alt.Color(phase_legend_name+':N',
                                        scale=self.tinted_color_scale,
                                        sort=self.einsum_order, legend=None),
                    ).transform_calculate(
                        neg_y='-datum.y', neg_y2='-datum.y2')
                    multi_resource_grey_layers.append(mr_sub_memory_area)

                ##### TWEAK 1: MR Einsum compute fill SOLID again #####
                #####################################################################
                # TASK A: MR Einsum COMPUTE fill -> "SOLID but LIGHT" (old look)    #
                #                                                                   #
                # Strictly inside `if self.multi_resource:` so ONLY the MR path     #
                # changes (the non-MR / single-resource path never builds this      #
                # rect, so multi_resource=False output stays byte-identical).       #
                #                                                                   #
                # We are restoring the ORIGINAL non-multi_resource dual-mode look   #
                # for the Einsum AREA fills: that path used the LIGHTENED scale at  #
                # full opacity, i.e. `mark_rect(opacity=1)` + the tinted scale.     #
                # So here we swap the colour scale from the full-saturation         #
                # `self.color_scale` (which rendered too dark) to the lightened    #
                # `self.tinted_color_scale`. We KEEP opacity=1.0 -> the fill is     #
                # SOLID (NOT translucent), just a lighter shade of the same Einsum  #
                # identity colour. `self.tinted_color_scale` is already built by    #
                # `self.update_color_scale(...)` in draw() (the original dual path  #
                # consumes it too). Band geometry (mr_comp_fill_bot ->              #
                # mr_comp_fill_top, within-band cumulative stacking) is unchanged.  #
                #####################################################################
                compute_area = base.mark_rect(opacity=1.0).encode(
                    x='Starting_Time:Q',
                    x2='full_End_Time:Q',
                    y='mr_comp_fill_bot:Q',
                    y2='mr_comp_fill_top:Q',
                    color=alt.Color(phase_legend_name+':N',
                                    scale=self.tinted_color_scale,
                                    sort=self.einsum_order,
                                    legend=None),
                    tooltip=tooltip_fields
                )

                #####################################################################
                # TASK A: MR Einsum MEMORY fill -> "SOLID but LIGHT" (old look)     #
                #                                                                   #
                # Same MR-only gating as compute_area above (inside                 #
                # `if self.multi_resource:`). Mirror onto the NEGATIVE axis (grows  #
                # downward from 0). To match the original dual-mode `memory_area`   #
                # we use the LIGHTENED `self.tinted_color_scale` at opacity=1.0:    #
                # SOLID fill, lighter shade -- NOT translucent. Stacks within the   #
                # band via mr_mem_fill_bot/top exactly as before.                   #
                #####################################################################
                memory_area = base.mark_rect(opacity=1.0).encode(
                    x='Starting_Time:Q',
                    x2='full_End_Time:Q',
                    y=alt.Y(
                        'neg_mem_fill_bot:Q',
                        title="Memory Utilization | Compute Utilization",
                        # Multi-resource: padded [-1.1,1.1] scale (label
                        # headroom) but ticks pinned to top out at 1.0/-1.0.
                        scale=_mr_y_scale,
                        axis=_mr_y_axis
                    ),
                    y2=alt.Y2('neg_mem_fill_top:Q'),
                    color=alt.Color(phase_legend_name+':N',
                                    scale=self.tinted_color_scale,
                                    sort=self.einsum_order,
                                    legend=None),
                    tooltip=tooltip_fields
                ).transform_calculate(
                    neg_mem_fill_bot='-datum.mr_mem_fill_bot',
                    neg_mem_fill_top='-datum.mr_mem_fill_top'
                )

                # -- (3) The SOLID ideal line (kept, re-pointed to bands) ----
                # This is the existing solid "ideal" line. Its x extent is the
                # unchanged compute ideal end time cb_End_Time (and mb_End_Time
                # for memory) -- those columns are untouched, so the solid
                # ideal line still renders exactly as before, just at the
                # band-relative compute level (mr_comp_fill_top: the
                # within-band stacked fill-top of this kernel) instead of the
                # legacy global Compute_Utilization_plot y.
                compute_lines = base.mark_line(
                    size=line_thickness, interpolate='step-after'
                ).encode(
                    x='Starting_Time:Q',
                    x2='cb_End_Time:Q',
                    y='mr_comp_fill_top:Q',
                    color=alt.Color(phase_legend_name+':N',
                                    scale=self.color_scale,
                                    sort=self.einsum_order, legend=None),
                    tooltip=tooltip_fields
                )

                # Memory side solid ideal line, mirrored negative, ending at
                # the memory ideal time mb_End_Time.
                memory_lines = base.mark_line(
                    size=line_thickness, interpolate='step-after'
                ).encode(
                    x='Starting_Time:Q',
                    x2='mb_End_Time:Q',
                    y='neg_mem_fill_top:Q',
                    color=alt.Color(phase_legend_name+':N',
                                    scale=self.color_scale,
                                    sort=self.einsum_order, legend=None),
                    tooltip=tooltip_fields
                ).transform_calculate(
                    neg_mem_fill_top='-datum.mr_mem_fill_top'
                )

                # -- (4) FULL-WIDTH roof lines + BLACK labels (CHANGE 4/1) ---
                # Each used resource gets ONE single horizontal rule at its
                # FIXED cumulative band_top spanning the ENTIRE x domain of
                # the cascade [self._mr_x_min, self._mr_x_max], regardless of
                # where the resource is actually active. We build a tiny
                # one-row-per-resource DataFrame (NOT a Python loop of charts:
                # a single data-driven mark_rule layer keyed by columns, so
                # the layer count does NOT grow with the number of kernels --
                # it grows only with the small number of distinct resources).
                #
                # Distinguishability (vs the dashed darkgray +/-1.0 / 0.0
                # "100% of ALL resources" reference hlines): the resource
                # roofs are SOLID THIN BLACK rules. So where a resource roof
                # coincides with the dashed darkgray 100% line, both remain
                # visible (solid thin black on top of dashed darkgray).
                #
                # FIX 1: the label must be PLAINLY BLACK and LEGIBLE. The
                # previous version drew a heavy white stroke/halo
                # (stroke='white', strokeWidth=2) AND anchored every label at
                # the global x-MIN (over the y-axis ticks), so the labels
                # rendered as unreadable white smudges stacked at x=0. We now
                # (a) DROP the white halo entirely (solid color="black" only,
                # bold weight is fine) and (b) anchor each label at the RIGHT
                # END of its full-width roof (the global x-MAX) sitting in the
                # open space at the right edge, NOT at x=0 over the axis.
                ##### TWEAK 3: roof RULE lines = dark grey, not black #####
                # This is inside `if self.multi_resource:` so it is MR-only.
                # The full-width per-resource roof RULES were solid black,
                # which was being confused with the solid BLACK y=0 divider
                # line. Recolour ONLY the roof rule lines to a dark grey
                # (#505050) so the black y=0 line and the dark-grey resource
                # roofs are distinguishable. Size stays 1 (thin, solid,
                # full-width). The roof LABELS stay black (handled by the
                # separate _LBL_KW color="black" below) and the y=0 divider
                # stays solid BLACK (handled later, unchanged).
                _ROOF_COLOR = "#505050"  # solid thin DARK GREY resource roof
                _ROOF_SIZE = 1          # thin (the 100% lines are size 2)
                # Build the per-resource roof rows in TWO DataFrames -- one for
                # compute (positive band_top, label sits ABOVE its roof so the
                # text baseline is 'bottom') and one for memory (negated
                # band_top, label sits BELOW its roof so the baseline is
                # 'top'). We split by side because mark_text.baseline is a
                # MARK property, not a data-driven encoding channel (Vega-Lite
                # rejects a field there); two static-baseline layers keep the
                # layer count tiny (it grows with the small number of
                # resources, NOT with the kernel count).
                _comp_roof_rows = []
                for _rn in self._mr_comp_resource_order:
                    _b, _t = comp_band_of_resource.get(_rn, (0.0, 0.0))
                    _comp_roof_rows.append({
                        "roof_y": _t,                       # +band_top
                        "x": self._mr_x_min,                # global x start
                        "x2": self._mr_x_max,               # global x end
                        "roof_label": _mr_roof_label(
                            _rn,
                            self._mr_comp_p_of_tot.get(_rn, 0.0),
                            self._mr_comp_accum.get(_rn, True)),
                    })
                _mem_roof_rows = []
                for _rn in self._mr_mem_resource_order:
                    _b, _t = mem_band_of_resource.get(_rn, (0.0, 0.0))
                    _mem_roof_rows.append({
                        "roof_y": -_t,                      # mirror negative
                        "x": self._mr_x_min,
                        "x2": self._mr_x_max,
                        "roof_label": _mr_roof_label(
                            _rn,
                            self._mr_mem_p_of_tot.get(_rn, 0.0),
                            self._mr_mem_accum.get(_rn, False)),
                    })
                # Guard against an empty side (e.g. no memory resources): an
                # empty DataFrame with the right columns renders nothing.
                _comp_roof_df = pd.DataFrame(
                    _comp_roof_rows,
                    columns=["roof_y", "x", "x2", "roof_label"])
                _mem_roof_df = pd.DataFrame(
                    _mem_roof_rows,
                    columns=["roof_y", "x", "x2", "roof_label"])

                # The roof rules: ONE solid thin black horizontal rule per
                # resource spanning the full global x domain. Compute + memory
                # roofs are emitted as two single data-driven mark_rule
                # layers (keyed by the per-resource rows).
                compute_roof_rules = alt.Chart(_comp_roof_df).mark_rule(
                    size=_ROOF_SIZE, color=_ROOF_COLOR
                ).encode(x='x:Q', x2='x2:Q', y='roof_y:Q')
                memory_roof_rules = alt.Chart(_mem_roof_df).mark_rule(
                    size=_ROOF_SIZE, color=_ROOF_COLOR
                ).encode(x='x:Q', x2='x2:Q', y='roof_y:Q')

                # The roof labels (FIX 1): SOLID BLACK glyph fill, NO halo /
                # stroke at all, anchored at the RIGHT END of the roof line
                # (the global x-MAX, encoded via the 'x2' column) so each
                # label sits in the open space at the right edge instead of
                # being smudged over the y-axis at x=0. align='right' with a
                # small negative dx tucks the text just INSIDE the right edge
                # so it stays within the plot. Compute labels sit just ABOVE
                # their roof (baseline 'bottom', small negative dy); memory
                # labels sit just BELOW (baseline 'top', small positive dy).
                _LBL_KW = dict(
                    align='right', fontSize=8,
                    fontWeight='bold',        # bold is allowed; aids legibility
                    color="black",            # SOLID BLACK glyph fill, no halo
                )
                # NOTE: x is encoded from the 'x2' column (== self._mr_x_max,
                # the right end of the full-width roof), NOT 'x' (the x-min).
                compute_roof_labels = alt.Chart(_comp_roof_df).mark_text(
                    baseline='bottom', dx=-3, dy=-3, **_LBL_KW
                ).encode(x='x2:Q', y='roof_y:Q', text='roof_label:N')
                memory_roof_labels = alt.Chart(_mem_roof_df).mark_text(
                    baseline='top', dx=-3, dy=3, **_LBL_KW
                ).encode(x='x2:Q', y='roof_y:Q', text='roof_label:N')

                # Combine into the layer list (stashed for LAST layering).
                roof_rules = alt.layer(compute_roof_rules, memory_roof_rules)
                roof_labels = alt.layer(
                    compute_roof_labels, memory_roof_labels)

                # Stash so the roofs/labels can be layered LAST (after the
                # 100% hlines) -- guaranteeing every roof, including the
                # cumulative-top one at +/-1.0, is drawn ON TOP and visible.
                self._mr_roof_layers = [roof_rules, roof_labels]
                # The early extra-layer list stays empty: roofs go in last.
                multi_resource_extra_layers = []

            #chart = compute_area + memory_area + compute_lines + memory_lines + zero_line
            chart = alt.layer(
                *multi_resource_grey_layers,
                compute_area,
                memory_area,
                compute_lines,
                memory_lines,
                zero_line,
                *multi_resource_extra_layers
            ).resolve_scale(color='independent')
          
            if tile_boundaries is not None and include_tile_boundaries:
                chart += tile_boundaries
            if mem_tile_boundaries is not None:
                chart += mem_tile_boundaries
            chart = chart.encode(
                x=alt.X('Time_Stamp:Q', title=x_title),
            )                
        elif self.multi_resource:
            ####################################################################
            # COLOCATED MULTI-RESOURCE RENDER PATH (CHANGE 2 + CHANGE 3)       #
            #                                                                  #
            # Reached ONLY when dual_mode=False AND multi_resource=True (the    #
            # auto-enable to dual was removed in CHANGE 1). This is the NEW     #
            # second mode; it lives in its own `elif` so the original single-  #
            # resource colocated code below (the final `else:`) is left        #
            # BYTE-IDENTICAL and the regression gate for *_colocated holds.    #
            #                                                                  #
            # What is the SAME as single-resource colocated:                   #
            #   * The Einsum compute step-line `compute_ticks` is UNCHANGED    #
            #     (same encoding, colour, step interpolation -- CHANGE 2 keeps #
            #     it as-is); it sits on the global compute cumulative.         #
            #   * Tile boundaries, tooltips, x encoding, hline(1.0), the       #
            #     y=0 / legend handling downstream -- all unchanged.           #
            #                                                                  #
            # What is NEW:                                                     #
            #   * (CHANGE 2) Full-width solid dark-grey (#505050, size 1)      #
            #     roof rules spanning the ENTIRE x-domain at each COMPUTE      #
            #     sub-resource's CUMULATIVE p_of_tot boundary, with black      #
            #     right-edge "Name (p_of_tot)" labels -- built by REUSING the  #
            #     exact same registry-order / cumulative-band / roof-row       #
            #     helpers the dual path already populated                      #
            #     (self._mr_comp_resource_order, comp_band_of_resource,        #
            #     self._mr_comp_p_of_tot, self._mr_x_min/_max,                  #
            #     _GREY_RAMP_*). Layered LAST (see the FIX B block) so they    #
            #     sit on top, identical treatment to dual roofs.               #
            #   * (CHANGE 3) The memory "pipe" is PERSISTENTLY PARTITIONED     #
            #     among all memory sub-resources by p_of_tot in registry       #
            #     order, each segment shaded its own grey from the SAME        #
            #     independent memory ramp (mem_grey_of_resource), spanning     #
            #     the full x-extent. The Einsum coloured used-bw rect is       #
            #     re-pointed to start at the BOTTOM of its memory sub-         #
            #     resource's segment (mr_colo_mem_fill_y -> _y2, computed      #
            #     above) instead of the single uniform guard_fill.            #
            ####################################################################
            legend = alt.Legend(title=phase_legend_name)

            #####################################################################
            # COMPUTE-BAND-OFFSET FIX (colocated MR compute step-line)          #
            #                                                                   #
            # The compute step-line must NOT use the raw pool-relative          #
            # cumulative util measured from absolute 0 (Compute_Utilization_    #
            # plot). In colocated multi-resource mode every compute             #
            # sub-resource owns a band [cumulative p_of_tot before it,          #
            # +p_of_tot]; the step-line for a kernel running on that resource   #
            # must be OFFSET up by the band bottom so it sits BETWEEN the       #
            # full-width grey compute boundary lines (which are already at the  #
            # correct cumulative p_of_tot). The dual path already computed      #
            # exactly that offset level per row as `mr_comp_fill_top`           #
            # (band_bottom + within-band cumulative pool-relative compute util, #
            # incl. concurrent same-resource stacking). REUSE it here -- only   #
            # the y channel changes; x/x2 (step-after to cb_End_Time), colour   #
            # and tooltip stay exactly as the single-resource colocated path.   #
            # This whole block is the colocated-MR-only `elif self.multi_       #
            # resource:` branch, so single-resource colocated output (which     #
            # uses the separate `else:` branch below, still reading             #
            # Compute_Utilization_plot) stays byte-identical.                   #
            #####################################################################
            compute_ticks = base.mark_line(size=line_thickness, interpolate='step-after').encode(
                x='Time_Stamp:Q',
                x2='cb_End_Time:Q',
                y=alt.Y('mr_comp_fill_top:Q', title="Compute Utilization", scale=y_scale, axis=y_axis),
                color=alt.Color(phase_legend_name+':N',
                                sort=self.einsum_order,
                                scale=self.color_scale,
                                legend = None
                               ),
                tooltip=tooltip_fields
            )

            ################################################################
            # CHANGE 3(a): PERSISTENT PARTITIONED memory pipe segments     #
            #                                                              #
            # Replace the SINGLE uniform-grey guard_fill with N stacked    #
            # grey rects -- one per memory sub-resource, in registry-      #
            # declaration order. Segment i spans pipe fraction             #
            # [Sum p_of_tot before i, Sum p_of_tot <= i] of the pipe;      #
            # mem_band_of_resource[name] holds exactly those FIXED         #
            # (band_bottom, band_top) magnitudes (built by the dual code   #
            # we reuse). Each segment is projected into plot Y using the   #
            # per-row pipe base/height computed earlier                    #
            # (mr_colo_pipe_base / _pipe_h). Because the pipe base depends  #
            # on Compute_Utilization_plot (it is anchored to the compute   #
            # line per guard_position), the segment rects are DATA-DRIVEN  #
            # per df row (NOT a tiny one-row-per-resource frame like the   #
            # dual full-width bands): each kernel-interval gets its memory #
            # segments stacked on its own compute level, exactly mirroring #
            # how the single-resource guard_fill tracked the compute line  #
            # per interval. We emit ONE mark_rect per memory resource via  #
            # a transform_calculate that re-derives that resource's        #
            # segment [bottom, top] in plot Y from the row's pipe base.    #
            #                                                              #
            # Greys come from the SAME independent memory ramp the dual    #
            # path built (mem_grey_of_resource); the resource->grey map is #
            # encoded as a REAL Altair Fill scale with an                  #
            # alt.Legend(title="Resources") so -- combined with the        #
            # resolve_scale(color='independent') below -- the combined     #
            # native "Resources" legend renders on the RIGHT next to the   #
            # Einsum legend, exactly as in dual (CHANGE 4).                 #
            ################################################################
            _pipe_scaling = float(bw_util_scaling)
            # Build the resource->grey scale (memory side only for the
            # colocated pipe; the compute roofs are unlabelled rules like
            # dual, the Resources legend lists the memory segments).
            _colo_res_domain = list(self._mr_mem_resource_order)
            _colo_res_range = [
                self._mr_mem_grey_of_resource.get(r, _GREY_RAMP_DARK)
                for r in self._mr_mem_resource_order
            ]
            _colo_resource_fill = alt.Fill(
                'resource:N',
                scale=alt.Scale(domain=_colo_res_domain,
                                range=_colo_res_range),
                legend=alt.Legend(title="Resources"),
            )

            # One persistent grey segment rect PER memory sub-resource.
            # Each is the FULL df re-used as the base (so the segment
            # tracks every kernel interval's compute-anchored pipe) with a
            # transform_calculate deriving THIS resource's fixed segment
            # [seg_bottom, seg_top] (pipe fraction) projected into plot Y:
            #   y  = mr_colo_pipe_base + seg_bottom * bw_util_scaling
            #   y2 = mr_colo_pipe_base + seg_top    * bw_util_scaling
            # spanning the kernel-interval x just like the old guard_fill.
            mr_colo_mem_segment_layers = []
            for _seg_name in self._mr_mem_resource_order:
                _seg_b, _seg_t = mem_band_of_resource.get(
                    _seg_name, (0.0, 0.0))
                _seg_grey = self._mr_mem_grey_of_resource.get(
                    _seg_name, _GREY_RAMP_DARK)
                # transform_calculate constants: this resource's fixed
                # pipe-fraction bottom/top and its grey + name (so the
                # scale legend lists every memory resource).
                _seg_layer = alt.Chart(df).transform_calculate(
                    seg_y=f"datum.mr_colo_pipe_base + "
                          f"{_seg_b} * {_pipe_scaling}",
                    seg_y2=f"datum.mr_colo_pipe_base + "
                           f"{_seg_t} * {_pipe_scaling}",
                    resource=f"'{_seg_name}'",
                ).mark_rect(opacity=0.65).encode(
                    x='Time_Stamp:Q',
                    x2='full_End_Time:Q',
                    y='seg_y:Q',
                    y2='seg_y2:Q',
                    fill=_colo_resource_fill,
                )
                mr_colo_mem_segment_layers.append(_seg_layer)

            ################################################################
            # CHANGE 3(b): the Einsum coloured used-bw rect, OFFSET        #
            #                                                              #
            # Identical role to the single-resource colocated memory_rects #
            # (same colour scale, same opacity 0.5, same x extent) BUT its #
            # y/y2 are re-pointed from the legacy y1/y2 to the segment-     #
            # offset mr_colo_mem_fill_y -> mr_colo_mem_fill_y2 computed     #
            # earlier: it now STARTS at the bottom of its memory sub-      #
            # resource's segment (offset by cumulative p_of_tot of memory  #
            # resources before this kernel's memory_type) and extends by   #
            # Memory_Utilization scaled by bw_util_scaling.                #
            ################################################################
            memory_rects = base.mark_rect(opacity=0.5).encode(
                x='Time_Stamp:Q',
                x2= 'full_End_Time:Q',
                y='mr_colo_mem_fill_y:Q',
                y2='mr_colo_mem_fill_y2:Q',
                color=alt.Color(phase_legend_name+':N',
                                scale=self.color_scale,
                                sort=self.einsum_order,
                                legend = None
                               ),
                tooltip=tooltip_fields
            )

            ################################################################
            # Guard top / bottom EDGE rules -- offset onto the FULL pipe.  #
            #                                                              #
            # In single-resource colocated these mark the pipe top/bottom  #
            # at compute_limit_top / compute_limit_bot. For MR the pipe is #
            # the full 100%-bw pipe [mr_colo_pipe_base, mr_colo_pipe_top], #
            # so we draw the edges there instead (consistently offset).    #
            ################################################################
            guard_band_layers = list(mr_colo_mem_segment_layers)
            if self.guard_position in ("above", "centered"):
              guard_top = base.mark_rule(
                  color="gray", strokeDash=[0,0], size=3
              ).encode(
                  x='Time_Stamp:Q',
                  x2='mb_End_Time:Q',
                  y='mr_colo_pipe_top:Q',
                  tooltip=tooltip_fields,
              )
              guard_band_layers.append(guard_top)
            if self.guard_position in ["below", "centered"]:
              guard_bot = base.mark_rule(
                  color="gray", strokeDash=[0,0], size=3
              ).encode(
                  x='Time_Stamp:Q',
                  x2='mb_End_Time:Q',
                  y='mr_colo_pipe_base:Q',
                  tooltip=tooltip_fields,
              )
              guard_band_layers.append(guard_bot)

            guard_band = alt.layer(*guard_band_layers)

            ################################################################
            # CHANGE 2: full-width COMPUTE roof rules + black labels       #
            #                                                              #
            # REUSE the dual path's registry-order / cumulative-band       #
            # machinery verbatim: self._mr_comp_resource_order is the      #
            # registry-declaration-ordered compute resource list,          #
            # comp_band_of_resource[name] == (cumulative bottom, cumulative #
            # top) so band_top IS the cumulative p_of_tot boundary, and    #
            # self._mr_x_min/_max are the global x-domain. We build the    #
            # SAME one-row-per-resource roof DataFrame + roof-label frame   #
            # (same _ROOF_COLOR #505050, size 1, solid; same black bold    #
            # right-edge "Name (p_of_tot)" label styling) as the dual      #
            # compute roofs. Stashed in self._mr_roof_layers so the        #
            # existing FIX B block layers them LAST (on top), identical to #
            # how the dual path does it.                                   #
            ################################################################
            _ROOF_COLOR = "#505050"   # solid thin DARK GREY (same as dual)
            _ROOF_SIZE = 1            # thin solid (matches dual roofs)
            _colo_comp_roof_rows = []
            for _rn in self._mr_comp_resource_order:
                _b, _t = comp_band_of_resource.get(_rn, (0.0, 0.0))
                _colo_comp_roof_rows.append({
                    "roof_y": _t,                       # cumulative p_of_tot
                    "x": self._mr_x_min,                # global x start
                    "x2": self._mr_x_max,               # global x end
                    "roof_label": _mr_roof_label(
                        _rn,
                        self._mr_comp_p_of_tot.get(_rn, 0.0),
                        self._mr_comp_accum.get(_rn, True)),
                })
            _colo_comp_roof_df = pd.DataFrame(
                _colo_comp_roof_rows,
                columns=["roof_y", "x", "x2", "roof_label"])
            # ONE solid thin dark-grey rule per compute resource, spanning
            # the full global x domain at its cumulative band_top.
            _colo_comp_roof_rules = alt.Chart(_colo_comp_roof_df).mark_rule(
                size=_ROOF_SIZE, color=_ROOF_COLOR
            ).encode(x='x:Q', x2='x2:Q', y='roof_y:Q')
            # Black, bold, right-edge labels (same styling as dual roofs).
            _LBL_KW = dict(
                align='right', fontSize=8, fontWeight='bold', color="black",
            )
            _colo_comp_roof_labels = alt.Chart(_colo_comp_roof_df).mark_text(
                baseline='bottom', dx=-3, dy=-3, **_LBL_KW
            ).encode(x='x2:Q', y='roof_y:Q', text='roof_label:N')
            # Stash for LAST layering by the existing FIX B block (it is
            # gated on multi_resource and iterates self._mr_roof_layers).
            self._mr_roof_layers = [
                alt.layer(_colo_comp_roof_rules),
                alt.layer(_colo_comp_roof_labels),
            ]

            ################################################################
            # PATH c: per-(phase, memory level) coloured used-bw fills.    #
            #                                                              #
            # A maps kernel's single aggregate memory_rects rect is        #
            # suppressed (zero height); its actual per-level colour is     #
            # drawn here. Each memory level's slot fill (slot-fraction      #
            # 0..1) is projected into the SAME colocated pipe space every   #
            # segment/edge uses: base (anchored to the pooled compute step- #
            # line per guard_position) + slot_fraction * bw_util_scaling.   #
            # Only built when some kernel carries a memory map, so scalar   #
            # colocated multi_resource output is unchanged.                #
            ################################################################
            mr_sub_mem_colo_layer = None
            if getattr(self, "_mr_sub_mem_fills", None):
                _colo_sub_rows = []
                for _f in self._mr_sub_mem_fills:
                    _ct = float(_f["comp_top"])
                    if self.guard_position == "below":
                        _b = _ct - _pipe_scaling
                    elif self.guard_position == "centered":
                        _b = _ct - _pipe_scaling / 2.0
                    else:  # "above"
                        _b = _ct
                    _colo_sub_rows.append({
                        phase_legend_name: _f[phase_legend_name],
                        "x": _f["x"], "x2": _f["x2"],
                        "y":  _b + float(_f["y"]) * _pipe_scaling,
                        "y2": _b + float(_f["y2"]) * _pipe_scaling,
                    })
                _colo_sub_df = pd.DataFrame(
                    _colo_sub_rows,
                    columns=[phase_legend_name, "x", "x2", "y", "y2"])
                mr_sub_mem_colo_layer = alt.Chart(
                    _colo_sub_df
                ).mark_rect(opacity=0.5).encode(
                    x='x:Q', x2='x2:Q', y='y:Q', y2='y2:Q',
                    color=alt.Color(phase_legend_name+':N',
                                    scale=self.color_scale,
                                    sort=self.einsum_order, legend=None),
                )

            # Compose: persistent grey memory segments + edges (guard_band)
            # BEHIND the Einsum coloured used-bw rects BEHIND the compute
            # step-line. resolve_scale(color='independent') keeps the
            # "Resources" grey Fill scale separate from the Einsum colour
            # scale so both legends render stacked on the RIGHT (CHANGE 4).
            _colo_layers = [guard_band, memory_rects]
            if mr_sub_mem_colo_layer is not None:
                _colo_layers.append(mr_sub_mem_colo_layer)
            _colo_layers.append(compute_ticks)
            chart = alt.layer(*_colo_layers).resolve_scale(color='independent')

            if tile_boundaries is not None and include_tile_boundaries:
                chart += tile_boundaries

            chart = chart.encode(
              x=alt.X('Time_Stamp:Q', title=x_title,
                      axis=(alt.Axis(grid=True, labels=False, ticks=False, title=None)
                            if not show_x_axis else alt.Undefined))
            )

        else: #colocated mode
            legend = alt.Legend(title=phase_legend_name)

            compute_ticks = base.mark_line(size=line_thickness, interpolate='step-after').encode(
                x='Time_Stamp:Q',
                x2='cb_End_Time:Q',
                y=alt.Y('Compute_Utilization_plot:Q', title="Compute Utilization", scale=y_scale, axis=y_axis),
                color=alt.Color(phase_legend_name+':N',
                                sort=self.einsum_order,
                                scale=self.color_scale,
                                legend = None
                               ),
                tooltip=tooltip_fields
            )

            memory_rects = base.mark_rect(opacity=0.5).encode(
                x='Time_Stamp:Q',
                x2= 'full_End_Time:Q', #'End_Time:Q',
                y='y1:Q',
                y2='y2:Q',
                color=alt.Color(phase_legend_name+':N',
                                scale=self.color_scale,
                                sort=self.einsum_order,
                                legend = None
                               ),
                tooltip=tooltip_fields
            )

            # guard_band = base.mark_rect(opacity=alpha, color="#69645b").encode( # color="darkgray"
            #     x='Time_Stamp:Q',
            #     x2= 'full_End_Time:Q', #'End_Time:Q',
            #     y='compute_limit_bot:Q',
            #     y2='compute_limit_top:Q',
            #     tooltip=tooltip_fields
            # )

            # shaded band fill
            guard_fill = base.mark_rect(
                color=KernelColor.lightenColor("#cac9c6", alpha) #opacity=alpha,
            ).encode(
                x='Time_Stamp:Q',
                x2='full_End_Time:Q',   # or 'End_Time:Q' if you prefer
                y='compute_limit_bot:Q',
                y2='compute_limit_top:Q',
                tooltip=tooltip_fields,
            )

            guard_band_layers = [guard_fill]

            # solid top & bottom edges
            if self.guard_position in ("above", "centered"):
              guard_top = base.mark_rule(
                  color="gray", strokeDash=[0,0], size=3
              ).encode(
                  x='Time_Stamp:Q',
                  x2='mb_End_Time:Q',#'full_End_Time:Q',
                  y='compute_limit_top:Q',
                  tooltip=tooltip_fields,
                  # color=alt.Color("Einsum:N", legend=None)
              )
              guard_band_layers.append(guard_top)

            if self.guard_position in ["below", "centered"]:
              guard_bot = base.mark_rule(
                  color="gray", strokeDash=[0,0], size=3
              ).encode(
                  x='Time_Stamp:Q',
                  x2='mb_End_Time:Q',#'full_End_Time:Q',
                  y='compute_limit_bot:Q',
                  tooltip=tooltip_fields,
                  # color=alt.Color("Einsum:N", legend=None)
              )
              guard_band_layers.append(guard_bot)

            guard_band = alt.layer(*guard_band_layers)



            chart = guard_band + memory_rects + compute_ticks

            chart = alt.layer(
                guard_band,
                memory_rects,
                compute_ticks,
            ).resolve_scale(color='independent')


            if tile_boundaries is not None and include_tile_boundaries:
                chart += tile_boundaries
                #print("Included tile boundaries")

            chart = chart.encode(
              x=alt.X('Time_Stamp:Q', title=x_title,
                      axis=(alt.Axis(grid=True, labels=False, ticks=False, title=None)
                            if not show_x_axis else alt.Undefined))
            )

        #### END OF IF/ELSE BLOCK ####

        ### Apply to both types of charts:
        if include_boundaries:
            df["fusion_group"] = df["fusion_group"].astype(str)
            chart += gen_fusion_group_boundaries(df)

        # Connect tiles...

        if connector_lines is not None:
            chart += connector_lines
          
        if mem_connector_lines is not None and self.dual_mode:
            chart += mem_connector_lines

      
        ########################################################################
        # show_ideal_dashes filter for the per-kernel throttled-line dicts.   #
        # Each row in `*_throttled_lines` carries an internal `_was_transformed` #
        # tag that we set above when appending. Here we:                      #
        #   * drop rows whose source kernel is vanilla (was_transformed=False) #
        #     when show_ideal_dashes is False; and                           #
        #   * ALWAYS strip the internal `_was_transformed` key so it never    #
        #     reaches pd.DataFrame() and therefore never appears in the spec. #
        # At default (show_ideal_dashes=True) the filter is a no-op pass-     #
        # through and the strip leaves the row dicts schema-identical to     #
        # before this edit -- regression specs stay byte-identical.          #
        ########################################################################
        def _gate_throttle_rows(rows):
            if not self.show_ideal_dashes:
                rows = [r for r in rows if r.get("_was_transformed")]
            return [{k: v for k, v in r.items() if k != "_was_transformed"}
                    for r in rows]

        throttled_lines      = _gate_throttle_rows(throttled_lines)
        mem_throttled_lines  = _gate_throttle_rows(mem_throttled_lines)
        comp_throttled_lines = _gate_throttle_rows(comp_throttled_lines)

        # Let's take care of throttling
        mem_throttled_df = pd.DataFrame(mem_throttled_lines)
      
        if not mem_throttled_df.empty and self.dual_mode:
            mem_throttled_rules = alt.Chart(mem_throttled_df).mark_rule(
                strokeDash=[4, 4], size=line_thickness
            ).encode(
                x='x:Q',
                x2='x2:Q',
                y='y:Q',
                color=alt.Color(phase_legend_name+':N', 
                                sort=self.einsum_order, 
                                scale=self.color_scale, 
                                legend=None),
                tooltip=[phase_legend_name+':N', 'x:Q', 'x2:Q']
            )
            chart += mem_throttled_rules


        # Now update the guard band lines with dashes too
        if not self.dual_mode:
            #####################################################################
            # COMPUTE-BAND-OFFSET FIX (colocated MR guard-edge throttle dashes) #
            #                                                                   #
            # These dashed rules continue the solid guard top/bottom EDGE rules #
            # past mb_End_Time. In single-resource colocated the solid edges    #
            # sit at compute_limit_top / compute_limit_bot (anchored to the raw #
            # Compute_Utilization_plot via draw_guard_bands), so the dashes use #
            # those same columns -- left BYTE-IDENTICAL here.                   #
            #                                                                   #
            # In colocated MR the solid edges were re-anchored to the OFFSET    #
            # pipe (mr_colo_pipe_top / mr_colo_pipe_base, which are now built   #
            # on the offset compute level mr_comp_fill_top, not the raw util),  #
            # so the dashed continuation must ride those SAME offset pipe       #
            # columns to stay colinear with the solid edges. Gate strictly on   #
            # self.multi_resource: True -> offset pipe columns; False -> the    #
            # original compute_limit_* columns unchanged (single-resource       #
            # colocated stays byte-identical).                                  #
            #####################################################################
            _mr_colo = getattr(self, "multi_resource", False)
            _gt_y = 'mr_colo_pipe_top:Q'  if _mr_colo else 'compute_limit_top:Q'
            _gb_y = 'mr_colo_pipe_base:Q' if _mr_colo else 'compute_limit_bot:Q'
            # Solid top & bottom edges -- now driven by `base_dashes`, which is
            # either `base` (default, show_ideal_dashes=True) or a filtered
            # base built over only transformed kernels' rows (when
            # show_ideal_dashes=False). Spec stays byte-identical at default.
            if self.guard_position in ("above", "centered"):
              guard_top_throttle = base_dashes.mark_rule(
                  color="gray", strokeDash=[4,4], size=3
              ).encode(
                  x='mb_End_Time:Q',
                  x2='full_End_Time:Q',#'full_End_Time:Q',
                  y=_gt_y,
                  tooltip=tooltip_fields,
                  # color=alt.Color("Einsum:N", legend=None)
              )
              chart += guard_top_throttle

            if self.guard_position in ["below", "centered"]:
              guard_bot_throttle = base_dashes.mark_rule(
                  color="gray", strokeDash=[4,4], size=3
              ).encode(
                  x='mb_End_Time:Q',
                  x2='full_End_Time:Q',#'full_End_Time:Q',
                  y=_gb_y,
                  tooltip=tooltip_fields,
                  # color=alt.Color("Einsum:N", legend=None)
              )
              chart += guard_bot_throttle
          

        comp_throttled_df = pd.DataFrame(comp_throttled_lines)
        if not comp_throttled_df.empty:
            comp_throttled_rules = alt.Chart(comp_throttled_df).mark_rule(
                strokeDash=[4, 4], size=line_thickness
            ).encode(
                x='x:Q',
                x2='x2:Q',
                y='y:Q',
                color=alt.Color(phase_legend_name+':N', 
                                sort=self.einsum_order, 
                                scale=self.color_scale,
                                legend=None),
                tooltip=[phase_legend_name+':N', 'x:Q', 'x2:Q']
            )
            chart += comp_throttled_rules

        throttled_df = pd.DataFrame(throttled_lines)
        if not throttled_df.empty and comp_throttled_df.empty:
            throttled_rules = alt.Chart(throttled_df).mark_rule(
                strokeDash=[4, 4], size=line_thickness
            ).encode(
                x='x:Q',
                x2='x2:Q',
                y='y:Q',
                color=alt.Color(phase_legend_name+':N', 
                                sort=self.einsum_order, 
                                scale=self.color_scale,
                                legend=None),
                tooltip=[phase_legend_name+':N', 'x:Q', 'x2:Q']
            )
            chart += throttled_rules
          
          
        if title is None:
          title=f"Campaign Diagram: {self.cascade.name}"

        if not self.dual_mode:
          chart = alt.layer(chart, hline(1.0, color="darkgray", scale=y_scale))
        elif self.multi_resource:
          ##################################################################
          # Phase 2 (rev): "100% of all resources" reference line(s).      #
          #                                                                #
          # The legacy dual path deliberately omits hline(1.0); we add it  #
          # back ONLY on the opt-in multi_resource dual view (this elif is #
          # unreachable when multi_resource is False, so the regression    #
          # snapshots are unaffected). We reuse the very same hline()      #
          # helper the non-dual path uses. Top edge at +1.0 marks "100% of #
          # ALL compute"; we also mirror it at -1.0 for "100% of ALL       #
          # memory" since this is the mirrored dual view.                  #
          ##################################################################
          chart = alt.layer(chart, hline(1.0, color="darkgray"))
          chart = alt.layer(chart, hline(-1.0, color="darkgray"))

        ##### TWEAK 2: MR-only BLACK y=0 divider #####
        # The y=0 line is the compute/memory divider. In the
        # multi_resource view we want it solid BLACK so it reads as a
        # firm separator over the (now translucent) grey bands. Strictly
        # gate on self.multi_resource: when True draw a solid black y=0
        # rule (same default size=2, dash=(0,0) == solid). When False we
        # fall through to the EXACT original
        # `hline(0.0, color="darkgray", dash=(0,0))` call, byte-for-byte
        # unchanged, so the non-MR dual cases (simple_dual / tiled_dual /
        # pipelined_dual) and the colocated cases stay byte-identical.
        if getattr(self, "multi_resource", False):
          chart = alt.layer(chart, hline(0.0, color="black", dash=(0,0)))
        elif self.dual_mode:
          chart = alt.layer(chart, hline(0.0,  color="darkgray", dash=(0,0)))
        else:  # colocated: respect the explicit y domain so y_min>0 can crop
          chart = alt.layer(chart, hline(0.0,  color="darkgray", dash=(0,0), scale=y_scale))

        ####################################################################
        # FIX B: layer the MINI-ROOF rules + labels LAST.                  #
        #                                                                  #
        # The roofs were intentionally NOT layered with the early extra    #
        # layers. By layering them here -- AFTER hline(1.0)/hline(-1.0)/   #
        # hline(0.0) -- every resource roof (FIX 2: SOLID THIN BLACK,      #
        # mark_rule size=1, full x-domain width), INCLUDING the            #
        # cumulative-top roof whose band_top is exactly the grand total    #
        # (+1.0 compute / -1.0 memory), is drawn ON TOP of the grey band   #
        # rects, the colour fills, AND the DASHED darkgray size-2 "100% of #
        # all resources" reference line, so the solid thin black roofs     #
        # stay visually distinct from those dashed reference lines and     #
        # fully visible and labelled. Gated on multi_resource; getattr     #
        # keeps the #
        # legacy path (which never sets _mr_roof_layers) untouched.        #
        ####################################################################
        if getattr(self, "multi_resource", False):
            for _roof_layer in getattr(self, "_mr_roof_layers", []):
                chart = alt.layer(chart, _roof_layer)

        # horizontal_line = (
        #     alt.Chart()
        #       .mark_rule(color='black', size=2)            # or strokeDash=[4,3]
        #       .encode(y=alt.datum(1.0))
        # )
        # chart = chart + horizontal_line

        #print(f"About to finalize the chart!")
        chart = chart.properties(
            width=width,
            height=height,
            title = title
        )
        # .interactive() binds the scales to an interval selection, which makes
        # Vega fall back to the DATA extent (with zero=True) and ignore the
        # explicit y domain -- so a non-zero y_min would never crop. For static
        # PDF export interaction is moot; only add it when not cropping so the
        # default (y_min=0) path is byte-identical. (2026-05-21)
        if y_min == 0.0:
            chart = chart.interactive()

        legend_only = alt.Chart(df.head(1)).mark_line(opacity=0).encode(
            color=alt.Color(phase_legend_name+':N', 
                            sort=self.einsum_order, 
                            scale=self.color_scale,
                            legend=alt.Legend(title=phase_legend_name)
                           )
        )
        #legend = mini_mem_legend(df, x_px=592, y_px=16, title="Memory utilization")
        if show_legend:
            chart = chart + legend_only

        ########################################################################
        # CHANGE 3: "Resources" legend now renders on the RIGHT (native).      #
        #                                                                      #
        # The "Resources" legend USED to be a pixel-overlay box floating OVER  #
        # the plot (resource_grey_legend(...), composed here via `chart +      #
        # legend`). Per the requirement it must instead sit in the RIGHT       #
        # margin alongside the native Einsum colour legend, NOT over the       #
        # chart. We achieved this the PREFERRED way: the persistent           #
        # full-width grey-band rects (CHANGE 1) encode their colour via a      #
        # REAL Altair scale -- alt.Fill('resource:N',                          #
        # scale=alt.Scale(domain=[resource names], range=[grey hexes]),        #
        # legend=alt.Legend(title="Resources")). Because the layered           #
        # multi_resource chart already calls                                   #
        # .resolve_scale(color='independent'), Vega-Lite renders that          #
        # Resources legend on the RIGHT, STACKED with the Einsum colour        #
        # legend (which is still emitted by `legend_only` above). So there is  #
        # NOTHING to add here anymore -- the pixel-overlay legend has been     #
        # removed entirely. This block is intentionally left as a no-op note   #
        # documenting where the Resources legend now comes from.               #
        ########################################################################

        return (chart, df) if return_df else chart


    def draw_einsum_connections(self, df):
        """
        Connect the same Einsum block to its subsequent block
        """
      
        connectors = []
        mem_connectors = []
        for einsum_name, group in df[df["runtime"] > 0].groupby(phase_legend_name):
            # within an Einsum, sort all the tiles by Starting Time
            #sorted_group = group.reset_index(drop=True) 
            sorted_group = group.sort_values("Starting_Time").reset_index(drop=True)
          
          #group.sort_values("Starting_Time").reset_index(drop=True)
            for i in range(1, len(sorted_group)):
                prev = sorted_group.loc[i - 1]
                curr = sorted_group.loc[i]

                # Help with smoothing things out, o/w we have blips
                # if abs(curr["Starting_Time"] - prev["full_End_Time"]) < 1e-6 and prev["runtime"] > 1e-15 and curr["runtime"] > 1e-15:
                #print(f"{einsum_name}, {curr["Starting_Time"]}, {prev["Compute_Utilization_plot"]},\
                #        {curr["Compute_Utilization_plot"]}, {curr["Compute_Utilization"]}, Duration: {prev["runtime"]}")
                connectors.append({
                    "x": curr["Starting_Time"],
                    "y": prev["Compute_Utilization_plot"],
                    "y2": curr["Compute_Utilization_plot"],
                    phase_legend_name: einsum_name,
                    "duration": curr["throttled_duration"]
                })
                mem_connectors.append({
                    "x": curr["Starting_Time"],
                    "y": -prev["Memory_Utilization_plot"],
                    "y2": -curr["Memory_Utilization_plot"],
                    phase_legend_name: einsum_name,
                    "duration": curr["throttled_duration"]
                })
        
        connector_df = pd.DataFrame(connectors)
        if not connector_df.empty:
            connector_lines = alt.Chart(connector_df).mark_rule(size=2).encode(
                x='x:Q',
                y='y:Q',
                y2='y2:Q',
                color=alt.Color(phase_legend_name+':N', 
                                scale=self.color_scale, 
                                sort=self.einsum_order, 
                                legend=None
                               ),
                tooltip=[phase_legend_name+':N', 'x:Q', 'y:Q', 'y2:Q', 'duration:Q']
            )
        else:
          connector_lines = None
          
        mem_connector_df = pd.DataFrame(mem_connectors)
      
        if not mem_connector_df.empty:
          mem_connector_lines = alt.Chart(mem_connector_df).mark_rule(size=2).encode(
              x='x:Q',
              y='y:Q',
              y2='y2:Q',
              color=alt.Color(phase_legend_name+':N', 
                              scale=self.color_scale, 
                              sort=self.einsum_order, 
                              legend=None
                             ),
              tooltip=[phase_legend_name+':N', 'x:Q']
          )
        else:
          mem_connector_lines = None
      
        return connector_lines, mem_connector_lines 

    def draw_guard_bands(self, df):
        ####################################################################
        # Build the Vega-Lite expression strings that lay out the          #
        # colocated-mode memory-bandwidth "pipe" in single-resource mode.  #
        #                                                                  #
        # Each `transform_calculate` expression below describes WHERE on   #
        # the y-axis the pipe sits relative to the compute step line:     #
        #                                                                  #
        #   * `compute_limit_top` / `compute_limit_bot`  ->  the *empty*   #
        #     grey "well" the pipe lives inside. Its full height is        #
        #     `bw_util_scaling * datum.memory_well`, i.e. the maximum      #
        #     possible memory-bandwidth pool scaled into plot units.       #
        #                                                                  #
        #   * `y1` / `y2`  ->  the COLORED bandwidth fill, which is the    #
        #     actual `Memory_Utilization` of the kernel scaled by the      #
        #     same `bw_util_scaling`. (At 100% util it fills the well; at  #
        #     150% util it overflows above 1.0 -- visible only when the    #
        #     scaling factor is large enough to make the overflow read.)   #
        #                                                                  #
        # Historically the scaling was a hardcoded 0.125 (so 100% bw       #
        # rendered at 12.5% of the y-axis and the `bw_util_scaling` kwarg  #
        # on `draw()` was silently ignored on this path). Now we read it   #
        # from `self.bw_util_scaling` (stored in `draw()`), so callers can #
        # actually grow or shrink the pipe by passing the kwarg.           #
        ####################################################################

        # Pull the scaling factor stashed in `draw()`. Keep a local alias so #
        # the f-strings below stay short and easy to scan.                   #
        bw_scale = self.bw_util_scaling

        if self.guard_position == "above":
            # Well sits ABOVE the compute line; bw fill grows upward from it.
            self.compute_limit_top_expr = f"datum.Compute_Utilization_plot + {bw_scale} * datum.memory_well"
            self.compute_limit_bot_expr = "datum.Compute_Utilization_plot"
            y1_expr = "datum.Compute_Utilization_plot"
            y2_expr = f"datum.Compute_Utilization_plot + datum.Memory_Utilization * {bw_scale}"
        elif self.guard_position == "below":
            # Well sits BELOW the compute line; bw fill grows downward from it.
            self.compute_limit_top_expr = "datum.Compute_Utilization_plot"
            self.compute_limit_bot_expr = f"datum.Compute_Utilization_plot - {bw_scale} * datum.memory_well"
            y1_expr = f"datum.Compute_Utilization_plot - datum.Memory_Utilization * {bw_scale}"
            y2_expr = "datum.Compute_Utilization_plot"
        else:  # centered
            # Well straddles the compute line; bw fill is symmetric.
            self.compute_limit_top_expr = f"datum.Compute_Utilization_plot + {bw_scale} * datum.memory_well"
            self.compute_limit_bot_expr = f"datum.Compute_Utilization_plot - {bw_scale} * datum.memory_well"
            y1_expr = f"datum.Compute_Utilization_plot + datum.Memory_Utilization * {bw_scale}"
            y2_expr = f"datum.Compute_Utilization_plot - datum.Memory_Utilization * {bw_scale}"

        base = alt.Chart(df).transform_calculate(
            compute_limit_top=self.compute_limit_top_expr,
            compute_limit_bot=self.compute_limit_bot_expr,
            y1=y1_expr,
            y2=y2_expr
        )
        return base

    def draw_tile_boundaries(self, df):
        tile_divider_df = []
        for einsum in df[phase_legend_name].unique():
            starts = df[df[phase_legend_name] == einsum]["Starting_Time"].sort_values().tolist()
            for s in starts[1:]:
                row = df[(df[phase_legend_name] == einsum) & (df["Starting_Time"] == s)]
                if not row.empty:
                    tile_divider_df.append({
                        "x": s,
                        phase_legend_name: einsum,
                        "Compute_Utilization_plot": float(row["Compute_Utilization_plot"].iloc[0]),
                        "Compute_Utilization": float(row["Compute_Utilization"].iloc[0]),
                        "memory_well": float(row["memory_well"].iloc[0]) # we actually need this b/c of the compute_limit_top_expr
                    })
        tile_divider_df = pd.DataFrame(tile_divider_df)

        tile_boundaries = None
      
        if not tile_divider_df.empty:
            if self.dual_mode:
                # Use your formula exactly:
                y_bot_expr = "datum.Compute_Utilization_plot - datum.Compute_Utilization"
                y_bottom   = "y_bot:Q"           # will be created below
            else:
                y_bot_expr = self.compute_limit_bot_expr   # already defined string
                y_bottom   = "compute_limit_bot:Q"
              
            tile_boundaries = alt.Chart(tile_divider_df).transform_calculate(
                                compute_limit_top=self.compute_limit_top_expr,
                                compute_limit_bot=self.compute_limit_bot_expr,
                                y_bot=y_bot_expr                     
                              ).mark_rule(
                                size=1, 
                                color="#D3D3D3",
                                opacity=self.alpha + .1
                              ).encode(
                                x="x:Q",
                                y=y_bottom,
                                y2="compute_limit_top:Q",
                                tooltip=["x:Q"],
                              )
        #print(f"Is tile boundaries none? {tile_boundaries==None}")
        return tile_boundaries

    def draw_mem_tile_boundaries(self, df):
      '''
      TODO:
      - go from previous Einsum to current Einsum
      '''
      pass 
      


def auto_pipeline_chart_by_group_size(
    df,
    spread=True,
    chart_kwargs=None,
    loader=None,
    diagram_class=None
):
    """
    Automatically generates a pipeline and Altair chart, with each fusion group
    using the number of Einsums it contains as the stage count.

    Parameters:
        df (pd.DataFrame): DataFrame with 'fusion_group' and 'Einsum' columns
        spread (bool): Passed to untiled_pipeline_group
        chart_kwargs (dict): Kwargs to CampaignDiagramAltair2.draw()
        loader (callable): Function to create Cascade from df (default: load_df_as_cascade)
        diagram_class (type): Diagram class (default: CampaignDiagramAltair2)

    Returns:
        chart, pipeline_df
    """
    import pandas as pd

    chart_kwargs = chart_kwargs or {}
    loader = loader or load_df_as_cascade

    # Ensure the dataframe is sorted as intended (by order of appearance)
    df = df.reset_index(drop=True)
    # Drop rows with empty fusion_group just in case
    mask = df["fusion_group"].notnull() & (df["fusion_group"].astype(str).str.strip() != "")
    fg_df = df[mask]

    # Get unique fusion_group order as they appear
    ordered_groups = []
    seen = set()
    for fg in fg_df["fusion_group"]:
        if fg not in seen:
            ordered_groups.append(fg)
            seen.add(fg)
    
    # Count Einsums per group
    group_counts = fg_df.groupby("fusion_group")[phase_legend_name].count().to_dict()

    # Build initial cascade
    cascade = loader(df)

    # Sequentially apply untiled_pipeline_group with correct stage count
    p = cascade
    for fg in ordered_groups:
        n_stages = group_counts[fg]
        p = p.untiled_pipeline_group(fg, stages=n_stages, spread=spread)
    
    # Generate the chart and dataframe
    chart, pipeline_df = CampaignDiagramAltair2(p).draw(**chart_kwargs)
    return chart, pipeline_df


def hline(y=1.0, *, color="black", size=2, dash=(4,2), scale=alt.Undefined):
    # scale: pass the chart's shared y Scale so this reference line respects the
    # explicit [y_min, y_max] domain instead of contributing its datum (and the
    # zero=True default) to the layered-scale union. With the default y_min=0 the
    # datum stays inside the domain, so rendering is unchanged. (2026-05-21)
    return (
        alt.Chart(pd.DataFrame({"y": [y]}))
        .mark_rule(color=color, size=size, strokeDash=list(dash))
        .encode(y=alt.Y("y:Q", scale=scale))   # <- horizontal line across the whole chart
    )
  
def mini_mem_legend(df: pd.DataFrame, *, x_px=592, y_px=16, title="Memory utilization") -> alt.Chart:
    # compute min/max from the per-kernel memory util (not the stacked plot value)
    if "Memory_Utilization" in df.columns and not df["Memory_Utilization"].empty:
        mem_min = float(df["Memory_Utilization"].min())
        mem_max = float(df["Memory_Utilization"].max())
    else:
        mem_min = mem_max = 0.0

    # 3 lines of text, positioned in pixel space (top-right by default)
    title_layer = alt.Chart(pd.DataFrame({"txt": [title]})).mark_text(
        fontWeight="bold"
    ).encode(
        x=alt.value(x_px),
        y=alt.value(y_px),
        text="txt:N",
        color=alt.value("#333")
    )

    min_layer = alt.Chart(pd.DataFrame({"txt": [f"min = {mem_min:.2f}"]})).mark_text().encode(
        x=alt.value(x_px),
        y=alt.value(y_px + 18),
        text="txt:N",
        color=alt.value("#333")
    )

    max_layer = alt.Chart(pd.DataFrame({"txt": [f"max = {mem_max:.2f}"]})).mark_text().encode(
        x=alt.value(x_px),
        y=alt.value(y_px + 34),
        text="txt:N",
        color=alt.value("#333")
    )

    # optional soft-white background behind the legend for readability
    bg = alt.Chart(pd.DataFrame({"x": [x_px - 150], "y": [y_px - 6]})).mark_rect(
        fill="white", fillOpacity=0.82, stroke="#ddd"
    ).encode(
        x=alt.value(x_px - 150),
        y=alt.value(y_px - 6)
    ).properties(width=150, height=52)

    return bg + title_layer + min_layer + max_layer


################################################################################
# Phase 2 (grey-band rev): per-resource SOLID GREY ramp helpers                #
#                                                                              #
# The "white gap" concept was replaced by an OPAQUE per-resource grey band     #
# background. Every compute sub-resource (and, on an INDEPENDENT ramp, every   #
# memory sub-resource) is assigned a distinct grey from an evenly-spaced ramp  #
# between a DARK end (#7a7a7a) and a LIGHT end (#b0b0b0). The FIRST resource   #
# in ResourceRegistry declaration order gets the DARKEST grey; later resources #
# get progressively lighter greys. These helpers are module-level and only    #
# ever exercised on the multi_resource path, so the single-resource regression #
# gate is unaffected (the legacy path never calls them).                       #
################################################################################

################################################################################
##### TWEAK 2: darken the persistent full-width grey-band ramp #####            #
# These two endpoints drive the ENTIRE multi_resource grey behaviour: the      #
# _grey_ramp() helper interpolates evenly between them for N resources, and    #
# the compute + memory sides each restart this SAME sequence independently.    #
# Because the persistent grey-band rects AND the native "Resources" legend     #
# swatches both read the resulting per-resource hex map, darkening the two     #
# endpoints here keeps the legend matching the bands automatically. These      #
# helpers are only ever exercised on the multi_resource path, so the           #
# single-resource / non-MR regression gate is unaffected. Per the spec the     #
# ramp is WIDENED from #7a7a7a->#b0b0b0 to #5a5a5a (darkest, FIRST resource    #
# in registry order) -> #bdbdbd (lightest, LAST resource) for more contrast    #
# between shades. The grey-band rect opacity stays 0.65 (handled at the        #
# mark_rect, unchanged).                                                       #
################################################################################
# Ramp endpoints, as required: darkest -> lightest.
_GREY_RAMP_DARK = "#5a5a5a"   # FIRST resource in registry order (darkest)
_GREY_RAMP_LIGHT = "#bdbdbd"  # LAST resource in registry order (lightest)


def _mr_roof_label(name, p_of_tot, accumulative):
    """Roof label for a multi-resource band.

    The p_of_tot number is only a share worth showing for an ACCUMULATIVE
    (pooled) resource, where it is that resource's peak-share of the pool
    (e.g. Tensor 0.88 / FMA 0.12). A non-accumulative level's p_of_tot is
    just its equal 1/N slot height (e.g. 0.50), which is NOT a pool fraction,
    so showing it wrongly implies a share -- for those we label the name only.
    """
    if accumulative:
        return f"{name} ({p_of_tot:.2f})"
    return f"{name}"


def _hex_to_rgb(hex_color):
    """Convert a '#rrggbb' string to an (r, g, b) int tuple (0-255)."""
    # Strip a leading '#' if present, then slice the three byte pairs.
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(rgb):
    """Convert an (r, g, b) tuple back to a lowercase '#rrggbb' string."""
    # Clamp each channel into [0, 255] defensively, then format as 2 hex digits.
    r, g, b = (max(0, min(255, int(round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _grey_ramp(n):
    """Return a list of n hex greys evenly spaced from DARK to LIGHT.

    For n == 1 we return ONLY the dark end (the requirement: "if N==1 use the
    dark end"). For n >= 2 we linearly interpolate each RGB channel between
    _GREY_RAMP_DARK (index 0, darkest) and _GREY_RAMP_LIGHT (index n-1,
    lightest), giving evenly-spaced steps. The list index corresponds to the
    resource's position in ResourceRegistry declaration order.
    """
    # Degenerate / single-resource case: just the dark end.
    if n <= 1:
        return [_GREY_RAMP_DARK]
    # Decode both ramp endpoints once.
    dark = _hex_to_rgb(_GREY_RAMP_DARK)
    light = _hex_to_rgb(_GREY_RAMP_LIGHT)
    ramp = []
    # Walk i = 0 .. n-1; fraction t goes 0.0 (dark) -> 1.0 (light) evenly.
    for i in range(n):
        t = i / (n - 1)
        # Per-channel linear interpolation dark + t*(light - dark).
        interp = tuple(
            dark[c] + t * (light[c] - dark[c]) for c in range(3)
        )
        ramp.append(_rgb_to_hex(interp))
    return ramp


def resource_grey_legend(
    comp_resource_order, comp_p_of_tot, comp_grey_of_resource,
    mem_resource_order, mem_p_of_tot, mem_grey_of_resource,
    *, x_px=120, y_px=16, title="Resources",
):
    """Build a pixel-positioned combined "Resources" legend overlay.

    Mirrors the existing pixel-overlay legend pattern (see mini_mem_legend /
    legend_only) so it composes into the chart with a simple `chart + legend`.
    For each resource we draw its assigned grey SWATCH (a small mark_rect) plus
    BLACK text "ResourceName (p_of_tot)". COMPUTE resources are listed first
    (in registry order), then MEMORY resources (in registry order), each with
    the exact grey it was assigned for its band background. Returns None if
    there are no resources at all (defensive; the multi_resource path always
    has at least one).
    """
    # Flatten the two ordered groups into a single rendering list, compute
    # first then memory, preserving each group's registry order.
    rows = []
    for rname in comp_resource_order:
        rows.append((rname, comp_p_of_tot.get(rname, 0.0),
                     comp_grey_of_resource.get(rname, _GREY_RAMP_DARK)))
    for rname in mem_resource_order:
        rows.append((rname, mem_p_of_tot.get(rname, 0.0),
                     mem_grey_of_resource.get(rname, _GREY_RAMP_DARK)))

    # Nothing to show -> no overlay (keeps composition a clean no-op).
    if not rows:
        return None

    # Layout constants (pixel space). Each entry is one row 18px tall; the
    # title sits on its own line above the first swatch.
    row_h = 18                       # vertical pitch between legend rows
    swatch_w = 14                    # grey swatch width  (px)
    swatch_h = 12                    # grey swatch height (px)
    text_dx = swatch_w + 6           # text starts just right of the swatch
    first_row_y = y_px + row_h       # y of the first resource row

    layers = []

    # Soft white background card behind the whole legend for readability,
    # sized to fit the title line + one line per resource row. We size it
    # via x/x2/y/y2 PIXEL-VALUE encoding (NOT .properties(width/height)):
    # setting width/height on a legend layer conflicts with the main
    # chart's own width/height when composed with `chart + legend`
    # (Altair raises "inconsistent values for height").
    card_w = 200
    card_h = row_h * (len(rows) + 1) + 10
    layers.append(
        alt.Chart(pd.DataFrame({"_": [0]})).mark_rect(
            fill="white", fillOpacity=0.85, stroke="#ddd"
        ).encode(
            x=alt.value(x_px - 8),
            x2=alt.value(x_px - 8 + card_w),
            y=alt.value(y_px - 8),
            y2=alt.value(y_px - 8 + card_h),
        )
    )

    # Bold BLACK title line.
    layers.append(
        alt.Chart(pd.DataFrame({"txt": [title]})).mark_text(
            fontWeight="bold", align="left", baseline="middle"
        ).encode(
            x=alt.value(x_px),
            y=alt.value(y_px),
            text="txt:N",
            color=alt.value("black"),
        )
    )

    # One swatch + one black label per resource row. The swatch is a single
    # filled SQUARE point mark (NOT a mark_rect + .properties): only the bg
    # card may set width/height, otherwise layering charts with conflicting
    # width/height raises "inconsistent values for height". A square point
    # sized via the `size` mark property avoids any per-layer .properties.
    _swatch_size = swatch_w * swatch_h    # point area in px^2
    for idx, (rname, p_tot, grey_hex) in enumerate(rows):
        row_y = first_row_y + idx * row_h
        # The grey swatch -- OPAQUE, the exact band-background grey.
        layers.append(
            alt.Chart(pd.DataFrame({"_": [0]})).mark_point(
                shape="square", filled=True, fill=grey_hex,
                fillOpacity=1.0, stroke="#888", strokeWidth=0.5,
                size=_swatch_size,
            ).encode(
                x=alt.value(x_px + swatch_w / 2.0),
                y=alt.value(row_y),
            )
        )
        # The BLACK label: "ResourceName (p_of_tot)".
        layers.append(
            alt.Chart(
                pd.DataFrame({"txt": [f"{rname} ({p_tot:.2f})"]})
            ).mark_text(align="left", baseline="middle").encode(
                x=alt.value(x_px + text_dx),
                y=alt.value(row_y),
                text="txt:N",
                color=alt.value("black"),
            )
        )

    # Compose all overlay layers into a single chart.
    legend = layers[0]
    for extra in layers[1:]:
        legend = legend + extra
    return legend


################################################################################
# Phase 2 (grey-band rev): resource identification = GREY + roof + legend.     #
#                                                                              #
# Earlier rejected approaches (a per sub-resource strokeDash border + dash     #
# legend; then a "white gap" capped by a per-active-interval mini-roof) have   #
# all been superseded. The CURRENT visual identifies a resource by an OPAQUE   #
# per-resource GREY band background (ramp #7a7a7a -> #b0b0b0 in registry       #
# order, independent ramps for compute vs memory), a SINGLE full-x-width thin  #
# black roof rule at its fixed band_top with a BLACK label, and a separate     #
# combined "Resources" grey-swatch legend. The module-level helpers for this   #
# are _grey_ramp() (ramp computation) and resource_grey_legend() (the          #
# pixel-overlay legend); the band geometry + grey assignment are computed      #
# inline in CampaignDiagramAltair2.draw() (see the "SOLID GREY BAND geometry"  #
# block). All of it is gated behind multi_resource.                            #
################################################################################


# Lighten an Altair Scale
def tint_scale(color_scale: alt.Scale, *, tint: float = 0.6) -> alt.Scale:
    color_scale_dict = color_scale.to_dict()
    color_range = color_scale_dict.get("range", None) 
                  #getattr(scale, "range", alt.Undefined)

    if color_range is alt.Undefined or not color_range:
        raise ValueError(
            "Input scale has no explicit range to tint. "
            "Provide a scale with `range=[...colors...]` (not just a `scheme`)."
        )
      
    color_scale_dict["range"] = [KernelColor.lightenColor(c, tint) for c in color_scale.range]  
    return alt.Scale(**color_scale_dict)

# Alias 
CampaignDiagram = CampaignDiagramAltair2