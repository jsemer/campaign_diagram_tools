## I need a debug printer
import pprint

class DebugPrinter:
    def __init__(self):
        self.debug = True

    def print(self, to_print):
        if self.debug:
            pprint.pprint(to_print)

    def toggle(self):
        self.debug = not self.debug

    def set(self):
        self.debug = True

    def unset(self):
        self.debug = False


