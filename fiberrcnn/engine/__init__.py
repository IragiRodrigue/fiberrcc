"""FiberRCNN engine: training and inference."""

try:
    from .trainer import FiberTrainer, build_fiber_cfg, set_seed, EarlyStoppingHook, WandbHook
    _has_detectron2 = True
except ModuleNotFoundError:
    _has_detectron2 = False

from .inference import FiberPredictor, FiberInstance, ImagePrediction

__all__ = ["FiberInstance", "ImagePrediction", "FiberPredictor"]
if _has_detectron2:
    __all__ += ["FiberTrainer", "build_fiber_cfg", "set_seed", "EarlyStoppingHook", "WandbHook"]
