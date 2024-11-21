import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from campaign_diagram.cascade import *


# Class to draw the plot using a list of Kernel objects
class CampaignDiagram:
    def __init__(self, cascade):

        cascade.assign_colors()
        self.cascade = cascade

        kernels = cascade.kernels
        self.kernels =  sorted(kernels,
                               key=lambda k: (k.start, -k.bw_util, k.compute_util, k.name))

    def draw(self, title=None, bw_util_scaling=0.25, ax=None):
        """Draw the campaign diagram """

        # Add some space before diagram
        print("")

        # Create figure and axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(12.8, 9.6))
        else:
            fig = ax.get_figure()

        # Get drawing data
        drawing_data, min_compute_util, max_compute_util = self.get_drawing_data(bw_util_scaling)

        # Render the kernels
        self.render_drawing_data(ax, drawing_data)

        if title is None:
            title = f"Campaign Diagram: {self.cascade.name}"

        # Final formatting and display of the plot
        self.format_plot(ax, min_compute_util, max_compute_util, title, bw_util_scaling)

        return self

    def get_drawing_data(self, bw_util_scaling):
        labels = {}
        current_parallel_start = None
        min_compute_util = bw_util_scaling  # Hack to set y-min at 0
        max_compute_util = 1.0
        drawing_data = []
        cumulative_compute_util = 0
        cumulative_bw_util = 0

        for kernel in self.kernels:

            # Only label each kernel name once
            cropped_name = kernel.name.split('.')[0]

            if cropped_name is None or cropped_name in labels:
                label = None
            else:
                label = cropped_name
                labels[label] = True

            if kernel.start != current_parallel_start:
                current_parallel_start = kernel.start
                cumulative_compute_util = kernel.compute_util
                cumulative_bw_util = 0
            else:
                cumulative_compute_util += kernel.compute_util

            if cumulative_compute_util > 1.0:
                print(f"{kernel.start:.2f}: Compute Overflow ({cumulative_compute_util:.2f})")

            min_compute_util = min(min_compute_util, cumulative_compute_util)
            max_compute_util = max(max_compute_util, cumulative_compute_util)

            # Create LineDrawingInfo for the compute line
            compute_line = LineDrawingInfo(
                start=current_parallel_start,
                end=kernel.end,
                util=cumulative_compute_util,
                color=kernel.compute_color,
                label=label,
                throttled_duration=kernel.throttled_duration
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
            drawing_info = KernelDrawingInfo(kernel.origin,
                                             compute_line,
                                             memory_rect,
                                             bw_rect)

            drawing_data.append(drawing_info)

        return drawing_data, min_compute_util, max_compute_util

    def render_drawing_data(self, ax, drawing_data):
        for n, info in enumerate(drawing_data):

            # Deal with splits of an original kernel
            found_it = False

            # Search for the next piece of a split kernel
            for m, candidate_info in enumerate(drawing_data[n+1:]):
                if info.origin == candidate_info.origin:
                    # Found a continuation of the current kernel
                    if info.compute_line.util == candidate_info.compute_line.util:
                        # Same heights - extend width and draw (or extend again) later
                        info.extend(candidate_info)
                        drawing_data[n+1+m] = info
                        found_it = True
                        break
                    else:
                        # Different heights - draw line connecting segments
                        info.compute_line.draw_v(ax, candidate_info)
                        break
                else:
                    # Stop looking on seeing new instance of same kernel
                    if info.origin.name == candidate_info.origin.name:
                        break

            if found_it:
                continue

            # Draw the compute line
            info.compute_line.draw(ax)

            # Draw the memory rectangle
            info.memory_rect.draw(ax)

            # Draw the bandwidth rectangle
            info.bw_rect.draw(ax)


    def format_plot(self, ax, min_compute_util, max_compute_util, title, bw_util_scaling):
        # Determine plot boundaries
        start_min = min([kernel.start for kernel in self.kernels]) - 0.1
        end_max = max([kernel.end for kernel in self.kernels]) + 0.1

        # Set title, limits, and labels
        ax.set_title(title)

        ax.set_xlim(start_min, end_max)
        ax.set_ylim(min_compute_util - bw_util_scaling,
                    max_compute_util + bw_util_scaling)

        ax.set_xlabel('Time')
        ax.set_ylabel('Compute Utilization')

        # Move the legend outside the right side of the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Show the plot
        plt.show()

        print(f"Cascade duration: {self.cascade.duration():.2f}")
        print(f"Cascade average compute utilization: {self.cascade.avg_compute_util():.2f}")
        print(f"Cascade average bw utilization: {self.cascade.avg_bw_util():.2f}")


    def __str__(self):
        """Returns a human-readable string representation of the CampaignDiagram's state."""
        kernel_states = "\n".join([str(kernel) for kernel in self.kernel_list])
        return f"CampaignDiagram with kernels:\n{kernel_states}"


class KernelDrawingInfo:
    def __init__(self, origin,compute_line, memory_rect, bw_rect):
        self.origin = origin
        self.compute_line = compute_line  # Instance of LineDrawingInfo
        self.memory_rect = memory_rect    # Instance of RectangleDrawingInfo
        self.bw_rect = bw_rect            # Instance of RectangleDrawingInfo

    def extend(self, extension):
        """ Extend the length of self with width of extension """

        extra_width = extension.memory_rect.width
        extra_throttled_duration = extension.compute_line.throttled_duration

        self.compute_line.end += extra_width
        self.compute_line.throttled_duration += extra_throttled_duration

        self.memory_rect.width += extra_width

        self.bw_rect.width += extra_width

        return self


class LineDrawingInfo:
    def __init__(self, start, end, util, color, label=None, throttled_duration=0):
        self.start = start
        self.end = end
        self.util = util  # Represents the cumulative compute utilization
        self.throttled_duration = throttled_duration
        self.color = color
        self.label = label  # Optional label for the line (e.g., kernel name)

    def draw(self, ax):

        throttle_point = self.end - self.throttled_duration

        ax.plot(
            [self.start, throttle_point],
            [self.util, self.util],
            color=self.color,
            lw=2,
            label=self.label
        )

        ax.plot(
            [throttle_point, self.end],
            [self.util, self.util],
            linestyle=':',
            color=self.color,
            lw=2,
            label=None
        )

    def draw_v(self, ax, next):

        ax.plot(
            [self.end, self.end],
            [self.util, next.compute_line.util],
            color=self.color,
            lw=2,
            label=self.label
        )


class RectangleDrawingInfo:
    def __init__(self, start, bottom, width, height, color, alpha=0.5):
        self.start = start
        self.bottom = bottom
        self.width = width
        self.height = height
        self.color = color
        self.alpha = alpha  # Transparency of the rectangle

    def draw(self, ax):
        rect = patches.Rectangle(
            (self.start, self.bottom),
            self.width,
            self.height,
            color=self.color,
            alpha=self.alpha
            )

        ax.add_patch(rect)


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
