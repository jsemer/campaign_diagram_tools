# A resource can be a memory type or compute type
# For a given resource type we can add resources
# A resource, when added must indicate:
#   - its name
#   - how much (as a fraction) it contributes to the resource pool
# Wheh adding kernels:
# - indicate the fraction of its specific resource.
#    - underlying code will calculate the fraction of the total usage.
# resource.py


# This particular file was created with the aid of ChatGPT-5
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import pandas as pd
except Exception:  # pandas optional; timeseries() will return lists if missing
    pd = None


# Resource model
class ResourceType(Enum):
    MEMORY = "memory_type"
    COMPUTE = "compute_type"

    @staticmethod
    def parse(value: Union["ResourceType", str]) -> "ResourceType":
        if isinstance(value, ResourceType):
            return value
        s = str(value).strip().lower()
        if s in {"memory", "mem", "memory_type"}:
            return ResourceType.MEMORY
        if s in {"compute", "comp", "compute_type"}:
            return ResourceType.COMPUTE
        raise ValueError(f"Unknown resource type: {value!r}")


@dataclass(frozen=True)
class Resource:
    name: str
    rtype: ResourceType
    # Attributes for a resource
    attrs: Dict[str, Any] = field(default_factory=dict)  # e.g., util, capacity, etc.


# Keep track of our resources
class ResourceRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, Resource] = {}

        # The single-resource default pools. Tracked so derive_p_of_tot() can
        # leave them out of an axis's equal-split count -- a genuine
        # multi-resource axis must not have its 1/N slots diluted by the
        # never-drawn default band.
        self._default_names: set = set()

        # our defaults
        self.add_compute("Compute Pool", compute_util=0.0)
        self._default_names.add("Compute Pool")
        self.add_memory("Memory Pool", bw_util=0.0)
        self._default_names.add("Memory Pool")

    def add_compute(self, name: str, accumulative: bool = True,
                    **attrs: Any) -> Resource:
        # Compute sub-resources POOL onto one axis (Tensor + FMA draw from the
        # same FLOP/s budget), so they are accumulative by default: their band
        # heights are peak-shares that sum across the axis.
        attrs["accumulative"] = accumulative
        return self._add(name, ResourceType.COMPUTE, **attrs)

    def add_memory(self, name: str, accumulative: bool = False,
                   **attrs: Any) -> Resource:
        # Memory levels form a HIERARCHY (L2, DRAM, ...) rather than a shared
        # pool, so they default to non-accumulative: derive_p_of_tot() hands
        # each an equal 1/N slot and never sums their heights.
        attrs["accumulative"] = accumulative
        return self._add(name, ResourceType.MEMORY, **attrs)

    def derive_p_of_tot(self) -> None:
        """Set each resource's ``p_of_tot`` band height from its accumulative flag.

        Opt-in: a caller that already hand-set ``p_of_tot`` and never calls this
        stays byte-identical. Two regimes, chosen per resource by the flag pinned
        at add_compute / add_memory time:

          * ACCUMULATIVE resources POOL. When they carry a ``peak`` attr their
            height is the peak share ``peak[r] / sum(peaks of the accumulative
            resources on the same axis)`` -- the fraction of the combined pool.
            Accumulative resources with an explicit ``p_of_tot`` and NO ``peak``
            are left exactly as declared (the historical hand-set-shares path),
            so existing compute callers are unchanged.

          * NON-ACCUMULATIVE resources do NOT pool: the axis region splits into
            EQUAL slices, ``p_of_tot = 1/N`` with N the number of
            non-accumulative resources on that axis (peak ratios ignored). Each
            slot is then an independent 0..1 gauge. Empty slots keep their slice.

        The constructor's default pools are excluded from the N count. Idempotent.
        """
        for rtype in (ResourceType.COMPUTE, ResourceType.MEMORY):
            # Per-axis default: compute pools accumulate, memory levels do not.
            axis_default_accum = (rtype is ResourceType.COMPUTE)
            on_axis = [
                res for name, res in self._by_name.items()
                if res.rtype is rtype and name not in self._default_names
            ]
            accum = [
                res for res in on_axis
                if bool(res.attrs.get("accumulative", axis_default_accum))
            ]
            non_accum = [res for res in on_axis if res not in accum]

            # Non-accumulative: equal 1/N slices, independent of peak ratios.
            n = len(non_accum)
            if n:
                for res in non_accum:
                    res.attrs["p_of_tot"] = 1.0 / n

            # Accumulative: peak-share, but ONLY when every one carries a peak;
            # otherwise leave the caller's explicit p_of_tot untouched so the
            # legacy hand-set-shares path is byte-identical.
            if accum and all("peak" in res.attrs for res in accum):
                total_peak = sum(float(res.attrs["peak"]) for res in accum)
                if total_peak > 0.0:
                    for res in accum:
                        res.attrs["p_of_tot"] = float(res.attrs["peak"]) / total_peak
  
    def _add(self, name: str, rtype: ResourceType, **attrs: Any) -> Resource:
        if name in self._by_name:
            raise ValueError(f"Resource {name!r} already exists.")
        res = Resource(name=name, rtype=rtype, attrs=dict(attrs))
        self._by_name[name] = res
        return res

    def get(self, name: str) -> Resource:
        return self._by_name[name]


# This is for each kernel:

@dataclass
class ComputeUse:
    resource: Resource
    util: Optional[float] = None  # normalized 0-1 if known
    notes: str = ""
 
@dataclass
# utilization of this Memory resource for this kernel.
class MemoryUse:
    resource: Resource
    util: Optional[float] = None  # normalized 0-1 if known
    # gbps: Optional[float] = None  # fallback: absolute bandwidth
    # bytes: Optional[int] = None   # optional for accounting
    notes: str = ""


# Bind a Resource to a Kernel...

class ResourceBinder:
    """
    Attaches resource usage to Kernel objects without modifying Kernel's class.
    Expects Kernel(name, start, duration). If Kernel has .end, it's used.
    """
    def __init__(self, registry: ResourceRegistry):
        self.reg = registry
        self._comp: Dict[int, Dict[str, ComputeUse]] = {}
        self._mem:  Dict[int, Dict[str, MemoryUse]]  = {}

    @staticmethod
    def _kernelid(kernel) -> int: 
      return id(kernel)
      
    @staticmethod
    def _kend(k) -> float: 
      # here, it gets k.end but if that doesn't work, calculate it
      return getattr(k, "end", k.start + k.duration)

    def use_compute(self, kernel, 
                    resource_name: str, 
                    util: Optional[float] = None, **kw) -> None:
        """If util is None, default to kernel.compute_util."""
        res = self.reg.get(resource_name)
        if res.rtype is not ResourceType.COMPUTE:
            raise ValueError(f"{resource_name} is not a compute resource.")
        if util is None:
            util = float(getattr(kernel, "compute_util", 0.0))
        self._comp.setdefault(self._kernelid(kernel), {})[resource_name] = ComputeUse(res, float(util), **kw)

    def use_memory(self, kernel, resource_name: str,
                   util: Optional[float] = None, **kw) -> None:
                   #gbps: Optional[float] = None,
                   #bytes: Optional[int] = None, **kw) -> None:
        """If util is None, default to kernel.bw_util"""
        res = self.reg.get(resource_name)
        if res.rtype is not ResourceType.MEMORY:
            raise ValueError(f"{resource_name} is not a memory resource.")
        if util is None:
            util = float(getattr(kernel, "bw_util", 0.0))
        self._mem.setdefault(self._kernelid(kernel), {})[resource_name] = MemoryUse(res, util=util, **kw)

    ###########################################################################
    # Resource accessors for a single kernel                                  #
    #                                                                         #
    # Each kernel currently maps to AT MOST one compute resource and one      #
    # memory resource. The four methods below expose, for a given kernel,     #
    # either the bound Resource OBJECT or just its NAME string. They are the  #
    # canonical, non-overlapping accessors -- previously there were two       #
    # methods both named `get_mem_resource`, so the object-returning one was  #
    # silently shadowed by the name-returning one. That ambiguity is removed  #
    # here: object getters end in `_resource`, name getters in `_resource_    #
    # name`.                                                                  #
    #                                                                         #
    # TODO: support a kernel with multiple compute/memory types. We would     #
    #       need to aggregate the per-resource usages into a single pool.     #
    ###########################################################################

    def get_comp_resource(self, kernel) -> Resource:
        # Look up every compute usage recorded for this specific kernel
        # instance (keyed by the kernel's object id).
        resource_dict = self._comp.get(self._kernelid(kernel))

        # We do not yet support a kernel spanning >1 compute resource, so
        # bail out loudly rather than silently picking one of them.
        if len(resource_dict.keys()) > 1:
            raise NotImplementedError(
                f"{kernel.name} has more than one compute resource")

        # There is exactly one entry -- grab its name (the dict key) and
        # return the Resource OBJECT that the name maps to.
        resource_name = next(iter(resource_dict))
        return resource_dict[resource_name].resource

    def get_mem_resource(self, kernel) -> Resource:
        # Look up every memory usage recorded for this specific kernel
        # instance (keyed by the kernel's object id).
        resource_dict = self._mem.get(self._kernelid(kernel))

        # As with compute, a kernel using more than one memory resource is
        # not yet supported; fail explicitly instead of guessing.
        if len(resource_dict.keys()) > 1:
            raise NotImplementedError(
                f"{kernel.name} has more than one memory resource")

        # Exactly one entry: return the Resource OBJECT it maps to.
        resource_name = next(iter(resource_dict))
        return resource_dict[resource_name].resource

    def get_comp_resource_name(self, kernel) -> str:
        # Same single-compute-resource lookup as get_comp_resource, but here
        # we return just the resource's NAME (the dict key) as a string.
        resource_dict = self._comp.get(self._kernelid(kernel))

        if len(resource_dict.keys()) > 1:
            raise NotImplementedError(
                f"{kernel.name} has more than one compute resource")

        resource_name = next(iter(resource_dict))
        return resource_name

    def get_mem_resource_name(self, kernel) -> str:
        # Same single-memory-resource lookup as get_mem_resource, but here we
        # return just the resource's NAME (the dict key) as a string.
        resource_dict = self._mem.get(self._kernelid(kernel))

        if len(resource_dict.keys()) > 1:
            raise NotImplementedError(
                f"{kernel.name} has more than one memory resource")

        resource_name = next(iter(resource_dict))
        return resource_name


    def usages_for(self, kernel) -> Tuple[Dict[str, ComputeUse], Dict[str, MemoryUse]]:
        return self._comp.get(self._kernelid(kernel), {}), self._mem.get(self._kernelid(kernel), {})

    # ---------- cumulative + individual utilization over time ----------

    def _events(self, kernels: Iterable[Any]) -> List[float]:
        ev = {k.start for k in kernels} | {self._kend(k) for k in kernels}
        return sorted(ev)

    def _active_in(self, kernels: Iterable[Any], t0: float, t1: float) -> List[Any]:
        out = []
        for k in kernels:
            ke = self._kend(k)
            if not (ke <= t0 or k.start >= t1):  # overlaps [t0, t1)
                out.append(k)
        return out

    ###########################################################################
    # Per-interval utilization timeseries (methods of ResourceBinder)         #
    #                                                                         #
    # The four methods below (_norm_mem, _pool_weight, timeseries, summarize) #
    # were accidentally dedented to MODULE scope even though they all take    #
    # `self`. They are re-indented here so they are bound METHODS of          #
    # ResourceBinder again. Their internal logic is intentionally UNCHANGED;  #
    # only their indentation / class membership is corrected.                 #
    ###########################################################################

    def _norm_mem(self, mu: MemoryUse) -> Optional[float]:
        """Return normalized memory utilization (0..1) if present; else None."""
        return None if mu.util is None else float(mu.util)

    def _pool_weight(self, res: Resource) -> float:
        """Fraction of the global pool this resource represents (default 1.0)."""
        try:
            w = float(res.attrs.get("p_of_tot", 1.0))
        except Exception:
            w = 1.0
        return max(0.0, w)

    def timeseries(
        self,
        kernels: List[Any],
        *,
        include_individual: bool = True,
        clamp_01: bool = False,
        relative_to_pool: bool = True,
        emit_total: bool = False,
    ):
        """
        Build per-interval utilization.

        Returns (cum_df, ind_df) if pandas is available, else (cum_rows, ind_rows).

        cum: rows per (interval, resource, kind) with util_cumulative (sum across
             active kernels), already scaled by resource pool share if
             relative_to_pool=True.
        ind: rows per (interval, resource, kernel, kind) with util_individual
             (per-resource fraction) and util_individual_pool (scaled-to-pool).
        """
        events = self._events(kernels)
        if len(events) < 2:
            return (None, None) if pd is not None else ([], [])

        cum_rows: List[Dict[str, Any]] = []
        ind_rows: List[Dict[str, Any]] = []

        for i in range(len(events) - 1):
            t0, t1 = events[i], events[i + 1]
            dur = t1 - t0
            if dur <= 0:
                continue

            active = self._active_in(kernels, t0, t1)

            comp_sum: Dict[str, float] = {}
            mem_sum: Dict[str, float] = {}

            total_comp = 0.0
            total_mem = 0.0

            for k in active:
                kc, km = self.usages_for(k)

                # --- compute resources ---
                for rn, cu in kc.items():
                    u = float(cu.util)
                    w = self._pool_weight(cu.resource) if relative_to_pool else 1.0
                    eff = u * w
                    comp_sum[rn] = comp_sum.get(rn, 0.0) + eff
                    total_comp += eff

                    if include_individual:
                        ind_rows.append(dict(
                            kind="compute", resource=rn, kernel=k.name,
                            t0=t0, t1=t1, duration=dur,
                            util_individual=u,             # per-resource fraction
                            util_individual_pool=eff       # scaled to pool
                        ))

                # --- memory resources ---
                for rn, mu in km.items():
                    u0 = self._norm_mem(mu)
                    if u0 is None:
                        continue
                    w = self._pool_weight(mu.resource) if relative_to_pool else 1.0
                    eff = u0 * w
                    mem_sum[rn] = mem_sum.get(rn, 0.0) + eff
                    total_mem += eff

                    if include_individual:
                        ind_rows.append(dict(
                            kind="memory", resource=rn, kernel=k.name,
                            t0=t0, t1=t1, duration=dur,
                            util_individual=u0,
                            util_individual_pool=eff
                        ))

            # write cumulative rows per resource
            for rn, s in comp_sum.items():
                cum_rows.append(dict(
                    kind="compute", resource=rn, t0=t0, t1=t1, duration=dur,
                    util_cumulative=min(1.0, s) if clamp_01 else s
                ))
            for rn, s in mem_sum.items():
                cum_rows.append(dict(
                    kind="memory", resource=rn, t0=t0, t1=t1, duration=dur,
                    util_cumulative=min(1.0, s) if clamp_01 else s
                ))

            # optional roll-up across all resources of a kind
            if emit_total:
                cum_rows.append(dict(
                    kind="compute", resource="__TOTAL__", t0=t0, t1=t1, duration=dur,
                    util_cumulative=min(1.0, total_comp) if clamp_01 else total_comp
                ))
                cum_rows.append(dict(
                    kind="memory", resource="__TOTAL__", t0=t0, t1=t1, duration=dur,
                    util_cumulative=min(1.0, total_mem) if clamp_01 else total_mem
                ))

        if pd is None:
            return cum_rows, ind_rows

        cum_df = pd.DataFrame(cum_rows)
        # TODO: `_pd` is undefined here; should be `pd`. Logic left UNCHANGED
        #       per the data-model scope -- only re-indented into the class.
        ind_df = pd.DataFrame(ind_rows) if include_individual else _pd.DataFrame([])
        return cum_df, ind_df

    def summarize(self, cum_df):
        """Peak and time-weighted average per (kind, resource)."""
        if pd is None:
            raise RuntimeError("pandas required for summarize().")
        if cum_df is None or cum_df.empty:
            return cum_df

        def _agg(g):
            dur = g["duration"]
            u = g["util_cumulative"].fillna(0.0)
            T = float(dur.sum())
            return pd.Series(dict(
                peak=float(u.max()),
                time_avg=float((u * dur).sum() / T) if T > 0 else 0.0,
            ))

        return (cum_df.groupby(["kind", "resource"], as_index=False)
                .apply(_agg)
                .reset_index(drop=True))
