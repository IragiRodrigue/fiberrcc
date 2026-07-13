"""FiberRCNN modeling package."""

from . import heads, losses  # noqa: F401

try:
    from . import roi_heads  # noqa: F401
except ModuleNotFoundError:
    pass  # detectron2 not available

__all__ = ["heads", "losses"]
