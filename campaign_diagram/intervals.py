import copy

import logging

#n Create a custom logger for this file/module
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)



class Intervals:
    def __init__(self, kernels):

        logger.debug("Initialize intervals")

        # Sort by (start time, end time)
        self.kernels = sorted(kernels, key=lambda k: (k.start, -k.duration))

        self.intervals = []
        self._group_kernels_into_intervals(kernels)

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
            while events and events[0].start == current_start_time:
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

                if active_kernel.end == min_end_time:
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
                        events.insert(0, remainder)

            # After processing, add the adjusted active kernels to the interval
            interval = Interval()
            interval.kernels = updated_kernels
            interval.check()

            # Append new interval to intervals
            self.intervals.append(interval)

    def __len__(self):

        return len(self.intervals)

    def __iter__(self):
        """Return an iterator over the interval instances."""

        return iter(self.intervals)


    def copy(self):

        return copy.deepcopy(self)

    def throttle(self):

        prev_end_time = 0

        for n, interval in enumerate(self.intervals):

            # Update the start time of all kernels in the interval and get the new end time
            new_end_time = interval.update_start_times(prev_end_time)

            # Scale the tasks in the interval for overutilization
            scaled_end_time = interval.scale_durations()

            # Update prev_end_time to the new_end_time of the interval
            prev_end_time = max(new_end_time, scaled_end_time)

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
            print(f"Interval: {i}")
            for k, kernel in enumerate(interval):
                print(f"Kernel ({k}) - {kernel}")

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
    def end(self):
        """The end of the interval is the end of the first kernel (since all kernels in an interval share the same end time)."""

        return self.kernels[0].end if self.kernels else None

    def __len__(self):

        return len(self.kernels)

    def __iter__(self):
        """Return an iterator over the kernel instances."""

        return iter(self.kernels)

    def check(self):

        min_start = min([kernel.start for kernel in self])
        max_start = max([kernel.start for kernel in self])

        if min_start != max_start:
            print(f"Broken interval (starts)")
            self.pretty_print()

        min_end = min([kernel.end for kernel in self])
        max_end = max([kernel.end for kernel in self])

        if min_end != max_end:
            print(f"Broken interval (ends)")
            self.pretty_print()

    def update_start_times(self, new_start_time):
        """Update the start time of all kernels

        Update the startg time within the interval to match the
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

        # Find the maximum of the two sums
        max_util = max(total_compute_util, total_bw_util)

#        print(f"{max_util = }")

        if max_util <= 1.0:
            return self.kernels[0].end

        # Kernel scale factor to make the max_util equal to 1.0
        scale_factor = 1.0 / max_util

        # Scale the attributes of each kernel proportionally
        for kernel in self.kernels:

            orig_duration = kernel.duration
            kernel.duration *= max_util

            kernel.throttled_duration *= max_util
            kernel.throttled_duration += kernel.duration - orig_duration

            kernel.compute_util *= scale_factor
            kernel.bw_util *= scale_factor

        return self.kernels[0].end

    def total_compute_util(self):
        """Return the total compute utilization for this interval."""
        return sum(kernel.compute_util for kernel in self.kernels)

    def total_bw_util(self):
        """Return the total bandwidth utilization for this interval."""
        return sum(kernel.bw_util for kernel in self.kernels)

    def pretty_print(self):

        for kernel in self:
            print(f"{kernel}")

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
