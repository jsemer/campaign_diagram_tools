from enum import Enum, auto

# Defines fusion between current Einsum and the downstream Einsum
class FusionType(Enum):
  STRONG = auto()
  WEAK = auto()
  NONE = auto()



