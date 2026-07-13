import copy

import logging

#n Create a custom logger for this file/module
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

from numpy import isclose

class Intervals:
    def __init__(self, kernels):

        logger.debug("Initialize intervals")

        # Sort by (start time, end time)
        self.kernels = sorted(kernels, key=lambda k: (k.start, -k.duration))
        # print(f"Intervals.py -- DEBUG:")
        # for kernel in self.kernels:
        #   print(f"{kernel.name},{kernel.start}, {kernel.end}\n")

        self.intervals = []
        self._group_kernels_into_intervals(kernels)
        # for interval in self.intervals:
        #   print(f"Processing interval...")
        #self.pretty_print()

    def _group_kernels_into_intervals(self, kernels):
        """Group kernels into intervals based on overlapping durations and same start time."""

        # Copy the sorted list to events (why?)
        events = list(kernels)


        while events:
            # Get the next kernel and its start time
            kernel = events.pop(0)
          
            # Initialize the minimum end time with the first kernel's end time
            current_start_time = kernel.start
            min_end_time = kernel.end

            # Start a new interval with the current kernel
            # Collect all kernels that start at the same time
            active_kernels = [kernel]
            while events and isclose(events[0].start, current_start_time): #events[0].start == current_start_time:
                next_kernel = events.pop(0)
                active_kernels.append(next_kernel)
                min_end_time = min(min_end_time, next_kernel.end)

            # Check if there are any subsequent kernels starting before the min_end_time
            # If so, chop off the interval at the time that kernel starts
            for event in events:
                # TODO: Optimize so we don't traverse the whole list
                min_end_time = min(min_end_time, event.start)

            logger.debug(f"{min_end_time = }")

            # Now go through kernels in the interval to split them appropriately
            updated_kernels = []

            for idx in reversed(range(len(active_kernels))):
                active_kernel = active_kernels[idx]

                if isclose(active_kernel.end, min_end_time):
                    logger.debug(f"Adding: {active_kernel}")
                    updated_kernels.append(active_kernel.copy())
                else:
                    logger.debug(f"Splitting: {active_kernel}")
                    # Split the kernel
                    first_part, remainder = active_kernel.split(min_end_time)

                    # Replace the original full kernel in the interval with the first part
                    if first_part is not None:
                        logger.debug(f"First part: {first_part}")
                        updated_kernels.append(first_part)

                    # Insert the remainder back at the front of the events list
                    if remainder is not None:
                        logger.debug("Remainder {remainder}")
                        #print(f"Remainder {active_kernel.name}{remainder}")
                        events.insert(0, remainder)

            # After processing, add the adjusted active kernels to the interval
            interval = Interval()
            interval.kernels = sorted(updated_kernels, key=lambda k: (k.start, -k.duration)) #updated_kernels

            interval.check()

            # Append new interval to intervals
            self.intervals.append(interval)

    def __len__(self):

        return len(self.intervals)

    def __iter__(self):
        """Return an iterator over the interval instances."""

        return iter(self.intervals)

    def __getitem__(self, index):
        """ Return an indexed interval """

        return self.intervals[index]

    def copy(self):
        return copy.deepcopy(self)

    def duration(self):
        """ Find duration of cascade """

        total_duration = 0

        for interval in self.intervals:
            total_duration += interval.duration

        return total_duration

    def avg_compute_util(self):
        """ Average compute utilization """

        total_compute = 0
        total_duration = 0

        for interval in self.intervals:
            interval_duration = interval.duration
            total_compute += interval_duration * interval.compute_util()
            total_duration += interval_duration

        return total_compute/total_duration


    def avg_bw_util(self):
        """ Average bw utilization """

        total_bw = 0
        total_duration = 0

        for interval in self.intervals:
            interval_duration = interval.duration
            total_bw += interval_duration * interval.bw_util()
            total_duration += interval_duration

        return total_bw/total_duration


    def throttle(self):


      
        prev_end_time = 0

        for n, interval in enumerate(self.intervals):

            # Update the start time of all kernels in the interval and get the new end time
            new_end_time = interval.update_start_times(prev_end_time)

            # Scale the tasks in the interval for overutilization
            scaled_end_time = interval.scale_durations()

            # Update prev_end_time to the new_end_time of the interval
            prev_end_time = max(new_end_time, scaled_end_time)
          
        # print(f"In throttle:")
        # self.pretty_print()
        return self


    def flatten(self):
        """Returns a flat list of kernels

        Returns kernels for all intervals

        """

        flattened_kernels = []
        prev_end_time = 0

        for n, interval in enumerate(self.intervals):
                flattened_kernels.extend(interval.kernels)

        return flattened_kernels


    def pretty_print(self):
        """Pretty print the intervals"""

        for i, interval in enumerate(self):
            print(f"\nInterval: {i}, {interval.start}, {interval.end}")
            for k, kernel in enumerate(interval):
                print(f"Kernel ({k}) - {kernel}")
                print(f"     {kernel.start}--{kernel.end}\n")
            

    def __repr__(self):
        return f"Intervals({len(self.intervals)} intervals)"

class Interval:
    def __init__(self):
        self.kernels = []

    @property
    def start(self):
        """The start of the interval is the start of the first kernel."""

        return self.kernels[0].start if self.kernels else None

    @property
    def duration(self):
        """The duration of the interval is the duration of the first kernel """

        return self.kernels[0].duration if self.kernels else None

    @property
    def end(self):
        """The end of the interval is the end of the first kernel (since all kernels in an interval share the same end time)."""

        return self.kernels[0].end if self.kernels else None

    def __len__(self):

        return len(self.kernels)

    def __iter__(self):
        """Return an iterator over the kernel instances."""

        return iter(self.kernels)

    def __getitem__(self, index):
        """ Return an indexed kernel """

        return self.kernels[index]

    def compute_util(self):
        """ Calculate average compute utilization """

        total_compute_util = 0

        for kernel in self.kernels:
            total_compute_util += kernel.compute_util

        return total_compute_util

    def bw_util(self):
        """ Calculate average bw utilization """

        total_bw_util = 0

        for kernel in self.kernels:
            total_bw_util += kernel.bw_util

        return total_bw_util


    def check(self):
        """
        Check that all kernels have the same start and end time
        within an interval
        """
      
        min_start = min([kernel.start for kernel in self])
        max_start = max([kernel.start for kernel in self])

        if min_start != max_start:
            print(f"Broken interval (starts)")
            #self.pretty_print()

        min_end = min([kernel.end for kernel in self])
        max_end = max([kernel.end for kernel in self])

        if min_end != max_end:
            print(f"Broken interval (ends)")
            #self.pretty_print()

    def update_start_times(self, new_start_time):
        """Update the start time of all kernels

        Update the starting time within the interval to match the
        interval's new start time.  Also returns the updated end time
        (max of all kernel end times).

        """

        for kernel in self.kernels:
            kernel.start = new_start_time

        # Recalculate and return the new end time for the interval
        new_end_time = max(kernel.end for kernel in self.kernels)

#        print(f"{new_end_time = }")

        return new_end_time


    def scale_durations(self):
        """Scales the durations of the kernels

        Scale kernel durations in the interval so that the maximum sum
        of compute_util or bw_util becomes 1.0.

        """

        total_compute_util = self.total_compute_util()
        total_bw_util = self.total_bw_util()

        #####################################################################
        # PATH c: accumulative-aware aggregation across CONCURRENT kernels. #
        #                                                                   #
        # The decision is now driven by each sub-resource's `accumulative`  #
        # flag (resolved once at update_global_util_view time and stashed   #
        # on the kernel), NOT by which axis it lives on:                    #
        #   - accumulative sub-resources POOL onto one budget -> their fills #
        #     SUM across concurrent kernels;                                #
        #   - non-accumulative sub-resources are independent 0..1 gauges ->  #
        #     PER LEVEL we take the max across concurrent kernels, then the  #
        #     single busiest level. Summing them (e.g. L2 0.6 + DRAM 0.6 ->  #
        #     1.2) would spuriously throttle.                               #
        # A mixed axis takes max(pool sum, busiest gauge). Only kick in when #
        # some kernel carries the corresponding map; pure-scalar intervals  #
        # keep the exact legacy sum, so single-resource output is           #
        # byte-identical.                                                   #
        #####################################################################
        if any(getattr(k, "compute_subutils", None) for k in self.kernels):
            total_compute_util = self._axis_demand("compute")
        if any(getattr(k, "memory_subutils", None) for k in self.kernels):
            total_bw_util = self._axis_demand("memory")

        # Find the maximum of the two sums
        max_util = max(total_compute_util, total_bw_util)

#        print(f"{max_util = }")

        if max_util <= 1.0:
            return self.kernels[0].end

        # Kernel scale factor to make the max_util equal to 1.0
        scale_factor = 1.0 / max_util

        # Scale the attributes of each kernel proportionally
        for kernel in self.kernels:
            kernel.dilate(max_util)
            # orig_duration = kernel.duration
            # kernel.duration *= max_util

            # #kernel.throttled_duration *= max_util
            # kernel.throttled_duration = kernel.duration - kernel.orig_duration

            # kernel.compute_util *= scale_factor
            # kernel.bw_util *= scale_factor

            # print(f"WE HAVE: {kernel.name} -- {orig_duration}, {max_util}, {kernel.duration}, {kernel.throttled_duration}")
        return self.kernels[0].end

    def total_compute_util(self):
        """Return the total compute utilization for this interval."""
        return sum(kernel.compute_util for kernel in self.kernels)

    def total_bw_util(self):
        """Return the total bandwidth utilization for this interval."""
        return sum(kernel.bw_util for kernel in self.kernels)

    def _axis_demand(self, axis):
        """Flag-driven demand for one axis across this interval's kernels (PATH c).

        ``axis`` is "compute" or "memory". Each map-carrying kernel exposes its
        per-sub-resource fills and the resolved ``accumulative`` flag (pinned by
        update_global_util_view). We partition by that flag:

          - ACCUMULATIVE sub-resources POOL, so their fills (util * p_of_tot)
            SUM across every concurrent kernel -- they share one budget;
          - NON-ACCUMULATIVE sub-resources are independent 0..1 gauges, so PER
            LEVEL we take the max across concurrent kernels, then the single
            busiest level.

        A scalar kernel (no map on this axis) contributes its pooled util to the
        accumulative pool, matching the legacy all-sum behavior. When pool and
        gauge contributions coexist the demand is max(pool sum, busiest gauge).
        Only ever called when at least one kernel carries the map, so the
        pure-scalar sum path is untouched.
        """
        if axis == "compute":
            submap_attr, subfills_attr, subaccum_attr, scalar_attr = (
                "compute_subutils", "compute_subfills",
                "compute_subaccum", "compute_util")
        else:
            submap_attr, subfills_attr, subaccum_attr, scalar_attr = (
                "memory_subutils", "memory_subfills",
                "memory_subaccum", "bw_util")

        pool_sum = 0.0
        have_pool = False
        level_max = {}
        for kernel in self.kernels:
            fills = getattr(kernel, subfills_attr, None)
            if getattr(kernel, submap_attr, None) and fills:
                flags = getattr(kernel, subaccum_attr, None) or {}
                for level, (u, _p, fill) in fills.items():
                    # Default accumulative for an unflagged level -> pool.
                    if flags.get(level, True):
                        pool_sum += fill
                        have_pool = True
                    else:
                        level_max[level] = max(level_max.get(level, 0.0), u)
            else:
                pool_sum += float(getattr(kernel, scalar_attr))
                have_pool = True

        gauge = max(level_max.values()) if level_max else 0.0
        if level_max and have_pool:
            return max(pool_sum, gauge)
        if level_max:
            return gauge
        return pool_sum

    def pretty_print(self):

        for kernel in self:
            print(f"{kernel}\n")

    def __repr__(self):
        return f"Interval(start={self.start:.2f}, end={self.end:.2f}, kernels={self.kernels})"



# Example usage

if __name__ == "__main__":

    import sys
    import os

    sys.path.append(os.path.abspath("./"))
    from .kernel import *

    kernels = [
        Kernel("K01", 0, 5, .50, .20),  # Kernel from time 0 to 5
        Kernel("K02", 3, 7, .50, .20),  # Kernel from time 3 to 10
        Kernel("K03", 2, 6, .60, .30),  # Kernel from time 2 to 8
        Kernel("K04", 1, 4, .70, .15),  # Kernel from time 1 to 5
        Kernel("K05", 4, 2, .40, .25),  # Kernel from time 4 to 6
        Kernel("K06", 5, 3, .80, .40),  # Kernel from time 5 to 8
        Kernel("K07", 10, 5, .60, .30), # Kernel from time 10 to 15
        Kernel("K08", 8, 4, .50, .20),  # Kernel from time 8 to 12
        Kernel("K09", 7, 3, .55, .22),  # Kernel from time 7 to 10
        Kernel("K10", 6, 2, .65, .35),  # Kernel from time 6 to 8
        Kernel("K11", 9, 3, .30, .10),  # Kernel from time 9 to 12
        Kernel("K12", 3, 2, .60, .25)    # Another kernel starting at 3, overlapping with the previous ones
]

    # Instantiate Intervals class, which automatically groups kernels into intervals
    intervals = Intervals(kernels)

    # Display results
    if False:
        print("Intervals")
        print(intervals)
        for interval in intervals.intervals:
            print(interval)

    if True:
        print("Unthrottled kernels")

        adjusted_kernels = intervals.adjust_kernels(throttle=False)

        # Display the adjusted kernels
        for kernel in adjusted_kernels:
            print(kernel)

    if True:
        print("Throttled kernels")

        adjusted_kernels = intervals.adjust_kernels()

    # Display the adjusted kernels
    for kernel in adjusted_kernels:
        print(kernel)
