import copy

from typing import Tuple
from typing import List

from campaign_diagram.kernel import *

class Cascade:
    """A class to manage a collection of Kernel instances."""

    # TODO: add support for  "+"
    # TODO: add supprot for len()

    def __init__(self, kernels: List[Kernel], offset=0):
        kernel_color = KernelColor()

        last_end = offset
        for kernel in kernels:
            kernel.set_start(last_end)
            last_end = kernel.end

        for kernel in kernels:
            name = kernel.name
            kernel.set_color(kernel_color.getColor(name))

        self.kernels = kernels

    def sort(self):
        """ Sort the kernels for display """

        self.kernels.sort(key=lambda k: (k.start, -k.bw_util))

    def split(self, parts):
        """ Split each task of a cascade into "parts" parts """

        parts_fraction = 1.0/parts

        split_kernels = []

        for kernel in self.kernels:
            split_kernels.append(copy.copy(kernel).scale_duration(parts_fraction))

        split_cascade = Cascade([copy.copy(kernel) for kernel in parts*split_kernels])

        return split_cascade

    def pipeline(self, stages=2, spread=False):

        tasks = []

        for stage in range(stages):
            task = self._create_spacers(stage)
            task.extend(copy.deepcopy(self).kernels[stage::stages])
            task.extend(self._create_spacers(stages-stage-1))

            tasks.append(task)

        # Start with a default previous_end value of zero
        previous_end = 0

        # Iterate over both lists in tandem
        for kernels in zip(*tasks):

            max_duration = max([kernel.duration for kernel in kernels])

            for kernel in kernels:
                kernel.set_start(previous_end)

                if spread:
                    if kernel.duration !=0 and kernel.duration < max_duration:
                        kernel.dilate(max_duration/kernel.duration)

            # TODO: Handle resource overutilization

            previous_end += max_duration


#            TODO: Add debug tracing
#            print(f"{kernel1.start = }")
#            print(f"{kernel2.start = }")
#            print(f"{previous_end =}")

        # TODO: allow Cascade creation without overwriting start/end times

        t = Cascade([])
        t.kernels = [kernel for task in tasks for kernel in task]
        return t

    def _create_spacers(self, count):

        spacer = Kernel(name=None,
                        duration=0,
                        compute_util=0,
                        bw_util=0)

        spacers =  [copy.copy(spacer) for _ in range(count)]

        return spacers


    def __iter__(self):
        """Return an iterator over the Kernel instances."""
        return iter(self.kernels)

    def __str__(self):
        """Returns a human-readable string representation of the CampaignDiagram's state."""
        kernel_states = "\n".join([str(kernel) for kernel in self.kernels])
        return f"CampaignDiagram with kernels:\n{kernel_states}"
