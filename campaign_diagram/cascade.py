import copy

from typing import Tuple
from typing import List

import logging

from campaign_diagram.kernel import *
from campaign_diagram.intervals import *

kernel_color_map = KernelColor()

class Cascade:
    """A class to manage a collection of Kernel instances."""

    # TODO: add support for  "+"
    # TODO: add supprot for len()

    def __init__(self, kernels: List[Kernel], sequential=False):

        self.logger = logging.getLogger('campaign_diagram.cascade')
        self.logger.setLevel(logging.DEBUG)

        if sequential:
            self.assign_starts(kernels)

        self.intervals = Intervals(kernels)


    @classmethod
    def fromIntervals(cls, intervals):

        c = cls([])
        c.intervals = intervals

        return c

    @property
    def kernels(self):
        """ Flatten intevals into a flat list of kernels """

        return self.intervals.flatten()

    def __len__(self):

        # TODO: Optimize?

        return len(self.kernels)


    def __iter__(self):
        """Return an iterator over the Kernel instances."""

        return iter(self.kernels)

    def is_sequential(self):

        last_end = 0

        for kernel in self.kernels:
            if kernel.start != last_end:
                return False

            last_end = kernel.end

        return True

#    def sort(self):
#        """ Sort the kernels by start time for display """
#
#        self.kernels = sorted(self.kernels,
#                              key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name))

    def assign_starts(self, kernels, offset=0):
        """ Assume the tasks are sequential and assign start times """

        last_end = offset
        for kernel in kernels:
            kernel.set_start(last_end)
            last_end = kernel.end

    def assign_colors(self):

        for kernel in self.kernels:
            name = kernel.name
            kernel.set_color(kernel_color_map.getColor(name))

    def split(self, parts):
        """Split each task of a cascade into "parts" parts

        Note: This only works for a cascade that is a simple
        sequential series of kernels.

        Todo: Check invariant!

        """

        parts_fraction = 1.0/parts

        split_kernels = []

        for kernel in self.kernels:
            split_kernels.append(copy.copy(kernel).scale_duration(parts_fraction))

        split_cascade = Cascade([copy.copy(kernel) for kernel in parts*split_kernels],
                        sequential=True)

        return split_cascade

    def pipeline(self, stages=2, spread=False):
        """Pipeline a set of tasks

        Note: Input must be a sequential set of tiles
        Note: This is not meaningful after a cascade is throttled

        """

        assert self.is_sequential()

        tasks = []

        orig_kernels = copy.deepcopy(self.kernels)

        for stage in range(stages):
            name = orig_kernels[stage].name
            task = self._create_spacers(stage, name)
            task.extend(orig_kernels[stage::stages])
            task.extend(self._create_spacers(stages-stage-1, name))

            tasks.append(task)

        # Start with a default previous_end value of zero
        new_kernels = []
        previous_end = 0

        # Iterate over both lists in tandem
        for kernels in zip(*tasks):

            max_duration = max([kernel.duration for kernel in kernels])

            for kernel in kernels:

                # Just skip kernels with duration == 0
                if kernel.duration == 0:
                    continue

                kernel.set_start(previous_end)

                if spread:
                    if kernel.duration !=0 and kernel.duration < max_duration:
                        kernel.dilate(max_duration/kernel.duration)

                new_kernels.append(kernel)

            # TODO: Handle resource overutilization

            previous_end += max_duration

#            TODO: Add debug tracing
#            print(f"{kernel1.start = }")
#            print(f"{kernel2.start = }")
#            print(f"{previous_end =}")

        # TODO: allow Cascade creation without overwriting start/end times

#        for k in new_kernels:
#            print(f"{k}")

        t = Cascade(new_kernels)

        return t

    def _create_spacers(self, count, name=None):

        spacer = Kernel(name=name,
                        duration=0,
                        compute_util=0,
                        bw_util=0)

        spacers =  [copy.copy(spacer) for _ in range(count)]

        return spacers

    def throttle(self):
        """Throttle a cascde to keep within resource constraints."""

        self.logger.debug("Starting throttle")

        new_intervals = copy.deepcopy(self.intervals)

        # Throttle the kernels in place
        new_intervals.throttle()

        return Cascade.fromIntervals(new_intervals)


    def pretty_print(self, intervals=False):

        if not intervals:
            for kernel in self.kernels:
                print(f"{kernel}")
        else:
            self.intervals.pretty_print()

    def __str__(self):
        ""

        "Returns a human-readable string representation of the CampaignDiagram's state."""
        kernel_states = "\n".join([str(kernel) for kernel in self.kernels])
        return f"CampaignDiagram with kernels:\n{kernel_states}"
