import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from campaign_diagram.cascade import *


# Class to draw the plot using a list of Kernel objects
class CampaignDiagram:
    def __init__(self, cascade):

        cascade.assign_colors()

        kernels = cascade.kernels

        self.kernels =  sorted(kernels,
                               key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name))

    def draw(self, title="Campaign Diagram", bw_util_scaling=0.25):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12.8, 9.6))

        # Get drawing data
        drawing_data, max_compute_util = self.get_drawing_data(bw_util_scaling)

        # Render the kernels
        self.render_drawing_data(ax, drawing_data)

        # Final formatting and display of the plot
        self.format_plot(ax, max_compute_util, title, bw_util_scaling)

        return self

    def get_drawing_data(self, bw_util_scaling):
        labels = {}
        current_parallel_start = None
        max_compute_util = 1.0
        drawing_data = []
        cumulative_compute_util = 0
        cumulative_bw_util = 0

        for kernel in self.kernels:
            # Only label each kernel name once
            if kernel.name is None or kernel.name in labels:
                label = None
            else:
                label = kernel.name
                labels[kernel.name] = True

            if kernel.start != current_parallel_start:
                current_parallel_start = kernel.start
                cumulative_compute_util = kernel.compute_util
                cumulative_bw_util = 0
            else:
                cumulative_compute_util += kernel.compute_util

            if cumulative_compute_util > 1.0:
                print(f"{kernel.start:.2f}: Compute Overflow ({cumulative_compute_util:.2f})")

            max_compute_util = max(max_compute_util, cumulative_compute_util)

            # Create LineDrawingInfo for the compute line
            compute_line = LineDrawingInfo(
                start=current_parallel_start,
                end=kernel.end,
                util=cumulative_compute_util,
                color=kernel.compute_color,
                label=label
            )

            # Create RectangleDrawingInfo for memory utilization rectangle
            current_bw_util = kernel.bw_util
            current_bw_util_scaled = bw_util_scaling * current_bw_util
            rect_height = current_bw_util_scaled
            rect_bottom = cumulative_compute_util - rect_height / 2
            memory_rect = RectangleDrawingInfo(
                start=current_parallel_start,
                bottom=rect_bottom,
                width=kernel.duration,
                height=rect_height,
                color=kernel.bw_color,
                alpha=0.5
            )

            # Create RectangleDrawingInfo for available bandwidth rectangle
            bw_util_available = 1.0 - cumulative_bw_util
            bw_util_available_scaled = bw_util_scaling * bw_util_available
            bw_bottom = cumulative_compute_util - bw_util_available_scaled / 2
            bw_rect = RectangleDrawingInfo(
                start=current_parallel_start,
                bottom=bw_bottom,
                width=kernel.duration,
                height=bw_util_available_scaled,
                color='lightgray',
                alpha=0.3
            )

            cumulative_bw_util += current_bw_util
            if cumulative_bw_util > 1.0:
                print(f"{kernel.start:.2f}: Bandwidth overflow ({cumulative_bw_util:.2f})")
                cumulative_bw_util = 1.0

            # Collect all drawing info in KernelDrawingInfo
            drawing_info = KernelDrawingInfo(compute_line, memory_rect, bw_rect)
            drawing_data.append(drawing_info)

        return drawing_data, max_compute_util

    def render_drawing_data(self, ax, drawing_data):
        for info in drawing_data:
            # Draw the compute line
            ax.plot(
                [info.compute_line.start, info.compute_line.end],
                [info.compute_line.util, info.compute_line.util],
                color=info.compute_line.color,
                lw=2,
                label=info.compute_line.label
            )

            # Draw the memory rectangle
            memory_rect = patches.Rectangle(
                (info.memory_rect.start, info.memory_rect.bottom),
                info.memory_rect.width,
                info.memory_rect.height,
                color=info.memory_rect.color,
                alpha=info.memory_rect.alpha
            )
            ax.add_patch(memory_rect)

            # Draw the bandwidth rectangle
            bw_rect = patches.Rectangle(
                (info.bw_rect.start, info.bw_rect.bottom),
                info.bw_rect.width,
                info.bw_rect.height,
                color=info.bw_rect.color,
                alpha=info.bw_rect.alpha
            )
            ax.add_patch(bw_rect)

    def format_plot(self, ax, max_compute_util, title, bw_util_scaling):
        # Determine plot boundaries
        start_min = min([kernel.start for kernel in self.kernels]) - 0.1
        end_max = max([kernel.end for kernel in self.kernels]) + 0.1

        # Set title, limits, and labels
        ax.set_title(title)
        ax.set_xlim(start_min, end_max)
        ax.set_ylim(0, max_compute_util + bw_util_scaling)
        ax.set_xlabel('Time')
        ax.set_ylabel('Compute Utilization')

        # Move the legend outside the right side of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Show the plot
        plt.show()

    def __str__(self):
        """Returns a human-readable string representation of the CampaignDiagram's state."""
        kernel_states = "\n".join([str(kernel) for kernel in self.kernel_list])
        return f"CampaignDiagram with kernels:\n{kernel_states}"


class KernelDrawingInfo:
    def __init__(self, compute_line, memory_rect, bw_rect):
        self.compute_line = compute_line  # Instance of LineDrawingInfo
        self.memory_rect = memory_rect    # Instance of RectangleDrawingInfo
        self.bw_rect = bw_rect            # Instance of RectangleDrawingInfo


class LineDrawingInfo:
    def __init__(self, start, end, util, color, label=None):
        self.start = start
        self.end = end
        self.util = util  # Represents the cumulative compute utilization
        self.color = color
        self.label = label  # Optional label for the line (e.g., kernel name)


class RectangleDrawingInfo:
    def __init__(self, start, bottom, width, height, color, alpha=0.5):
        self.start = start
        self.bottom = bottom
        self.width = width
        self.height = height
        self.color = color
        self.alpha = alpha  # Transparency of the rectangle



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
