import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from campaign_diagram.cascade import *


# Class to draw the plot using a list of Kernel objects
class CampaignDiagram:
    def __init__(self, cascade):

        cascade.sort()
        self.cascade = cascade

    # TODO: Move all bw scaling to here...
    def draw(self, bw_util_scaling=0.25):
        fig, ax = plt.subplots()

        labels = {}
        current_parallel_start = None

        for kernel in self.cascade:

#            print(f"Plotting: {kernel.name = } {kernel.start = } {kernel.duration} {kernel.compute_util = }")

            # Only label each kernel name once
            if kernel.name is None or kernel.name in labels:
                label = None
            else:
                label = kernel.name
                labels[kernel.name] = True


            if kernel.start != current_parallel_start:
                current_parallel_start = kernel.start
                current_compute_util = kernel.compute_util
                bw_util_available = 1.0 * bw_util_scaling
            else:
                current_compute_util += kernel.compute_util

            # Draw the compute line with a label (name)
            ax.plot([current_parallel_start, kernel.end],
                    [current_compute_util, current_compute_util],
                    color=kernel.compute_color,
                    lw=2,
                    label=label)

            # Draw the memory rectangle centered at compute_util (no label for memory)

            rect_height = bw_util_scaling * kernel.bw_util
            rect_bottom = current_compute_util - rect_height / 2
            rect = patches.Rectangle((current_parallel_start, rect_bottom),
                                     kernel.duration,
                                     rect_height,
                                     color=kernel.bw_color,
                                     alpha=0.5)
            ax.add_patch(rect)

            # Draw the light gray box instead of the two dotted lines

            bw_top = current_compute_util + bw_util_available / 2  # Top of the memory limit
            bw_bottom = current_compute_util - bw_util_available / 2  # Bottom of the memory limit
            bw_rect = patches.Rectangle((current_parallel_start, bw_bottom),
                                        kernel.duration,
                                        bw_util_available,
                                        color='lightgray',
                                        alpha=0.3)
            ax.add_patch(bw_rect)

            bw_util_available -= bw_util_scaling * kernel.bw_util

        # Format the plot

        # TODO: Max util is not calculated correctly

        start_min = min([kernel.start for kernel in self.cascade]) - 0.1
        end_max = max([kernel.end for kernel in self.cascade]) + 0.1
        max_util = max([kernel.compute_util + 1.0*bw_util_scaling for kernel in self.cascade])

        ax.set_xlim(start_min, end_max)
        ax.set_ylim(0, 1.4*max_util)
        ax.set_xlabel('Time')
        ax.set_ylabel('Compute Utilization')

        # Move the legend outside the right side of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        plt.show()

        return self

    def __str__(self):
        """Returns a human-readable string representation of the CampaignDiagram's state."""
        kernel_states = "\n".join([str(kernel) for kernel in self.kernel_list])
        return f"CampaignDiagram with kernels:\n{kernel_states}"





if __name__ == "__main__":

    # Example usage with multiple Kernel instances
    kernel1a = Kernel(name='EinsumA',
                     duration=3,
                     compute_util=0.7,
                     bw_util=0.25)

    kernel1b = Kernel(name='EinsumB',
                     duration=10,
                     compute_util=0.2,
                     bw_util=0.9)

    kernel1c = Kernel(name='EinsumC',
                     duration=2,
                     compute_util=0.6,
                     bw_util=0.4)



    # Create the plot with a list of kernel instances
    cascade1 = Cascade([kernel1a, kernel1b, kernel1c])
    CampaignDiagram(cascade1).draw()
