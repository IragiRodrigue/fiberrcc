"""Backbone registration for FiberRCNN (ConvNeXt, Swin)."""
from . import fiber_backbones  # noqa: F401 — triggers BACKBONE_REGISTRY registrations

__all__ = ["fiber_backbones"]
