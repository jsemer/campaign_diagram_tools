from palettable.colorbrewer.qualitative import Set3_12
import hashlib

class KernelColor:
    # # Let's use palatteable instead 
    # def __init__(self):
    #     self.colors = Set3_12.hex_colors  # 12 visually distinct colors
    #     self.current_index = 0
    #     self.name2color = {}
      
    # # Define a list of 24 common colors in hexadecimal format
    EMER_COLORS = [
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
  
    DARK_COLORBLIND_FRIENDLY = [
        "#332288",  # dark blue
        "#117733",  # dark green
        "#882255",  # dark red
        "#44AA99",  # teal
        "#999933",  # mustard
        "#AA4499",  # purple
        "#DDCC77",  # sand
        "#88CCEE",  # sky blue
        "#CC6677",  # rose
        "#661100",  # rust
        "#6699CC",  # steel blue
        "#AA4466",  # mauve
        "#4477AA",  # bright blue
        "#228833",  # strong green
        "#EE6677",  # salmon
    ]
    def __init__(self, default="DARK_CB"):
        if default=="DARK_CB":
          self.colors = self.DARK_COLORBLIND_FRIENDLY
        else:
          self.colors = self.EMER_COLORS
        self.current_index = 0  # Start with the first color
        self.name2color = {}

    def getColor(self, name):
        """
        Assign a color to a kernel name if a color has not yet been assigned.
        Otherwise retrieve the assigned color.
        """
        name = name.split('.')[0]

        if name in self.name2color:
            return self.name2color[name]
        else:
            if self.current_index < len(self.colors):
                color = self.nextColor()
            else:
                color = self.hashColor(name)
            self.name2color[name] = color
            return color
          
    def hashColor(self, name: str) -> str:
        """Generate a unique fallback color using hashing."""
        h = hashlib.md5(name.encode()).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f'#{r:02X}{g:02X}{b:02X}'
      
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
