

# Class to hold the parameters for Kernel
class Kernel:
    def __init__(self,
                 name,
                 duration,
                 compute_util,
                 bw_util,
                 bw_util_limit=1.0):

        self.name = name
        self.start = 0
        self.duration = duration
        self.compute_util = compute_util
        self.bw_util = bw_util
        self.bw_util_limit = bw_util_limit
        self.compute_color = None
        self.bw_color = None

    @property
    def end(self):
        """ Return end based on start and duration properties """

        return self.start + self.duration

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

    def scale_duration(self, scale):
        self.duration *= scale

        return self

    def dilate(self, dilation):
        self.duration *= dilation

        inverse_dilation = 1.0 / dilation

        self.compute_util *= inverse_dilation
        self.bw_util *= inverse_dilation
        self.bw_util_limit *= inverse_dilation

        return self

    def __str__(self):
        """Returns a human-readable string representation of the Kernel's state."""
        return (f"Kernel(name={self.name}, start={self.start}, duration={self.duration}, "
                f"compute_util={self.compute_util}, bw_util={self.bw_util}, "
                f"compute_color={self.compute_color}, bw_util_limit={self.bw_util_limit})")

class KernelColor:
    # Define a list of 24 common colors in hexadecimal format
    colors = [
        "#0000FF",  # Blue
        "#00FF00",  # Green
        "#FF0000",  # Red
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#C0C0C0",  # Silver
        "#808080",  # Gray
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008000",  # Dark Green
        "#800080",  # Purple
        "#000080",  # Navy
        "#FFA500",  # Orange
        "#FFC0CB",  # Pink
        "#FFD700",  # Gold
        "#F08080",  # Light Coral
        "#6495ED",  # Cornflower Blue
        "#228B22",  # Forest Green
        "#DB7093",  # Pale Violet Red
        "#FF6347",  # Tomato
        "#87CEEB",  # Sky Blue
        "#FFE4C4",  # Bisque
        "#98FB98",  # Pale Green
        "#9370DB"   # Medium Purple
    ]

    def __init__(self):
        self.current_index = 0  # Start with the first color
        self.name2color = {}

    def getColor(self, name):

        if name in self.name2color:
            return self.name2color[name]
        else:
            color = self.nextColor()
            self.name2color[name] = color
            return color

    def nextColor(self) -> str:
        """Return the next color in the list and wrap around if necessary."""
        color = self.colors[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.colors)
        return color

    @staticmethod
    def lightenColor(hex_color: str, amount: float) -> str:
        """Lighten the given hex color by the specified amount.

        Args:
            hex_color: A string representing the color in hex format (e.g., '#rrggbb').
            amount: A float value between 0-1 indicating the percentage to lighten the color.

        Returns:
            A new hex color string that is lightened.
        """
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)

        # Lighten the color
        r = min(int(r + (255 - r) * amount), 255)
        g = min(int(g + (255 - g) * amount), 255)
        b = min(int(b + (255 - b) * amount), 255)

        # Convert back to hex
        return f'#{r:02X}{g:02X}{b:02X}'


