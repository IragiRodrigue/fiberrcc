"""
FiberRCNN v2 — Advanced Scientific Nanofiber Analysis Framework
================================================================
Importing this package registers all custom Detectron2 components:

  * FiberROIHeads  (ROI_HEADS_REGISTRY)
  * ConvNeXt / Swin backbones  (BACKBONE_REGISTRY)

Example
-------
>>> import fiberrcnn
>>> from fiberrcnn.engine.inference import FiberPredictor
>>> predictor = FiberPredictor.from_config("configs/fiber_rcnn_r50_fpn.yaml",
...                                        "output/model_final.pth")
>>> result = predictor.predict("image.png")
"""

from __future__ import annotations

# --- trigger Detectron2 registrations (optional: requires detectron2) ---
try:
    from fiberrcnn.modeling.roi_heads import FiberROIHeads          # noqa: F401
    from fiberrcnn.modeling.backbones import fiber_backbones        # noqa: F401
except ModuleNotFoundError:
    pass  # detectron2 not installed; geometry/morphology modules still available

__version__ = "2.0.0"
__all__ = ["__version__"]
