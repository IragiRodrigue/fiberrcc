"""FiberRCNN evaluation metrics."""

try:
    from .fiber_evaluator import (
        FiberEvaluator,
        mean_absolute_error,
        root_mean_squared_error,
        mean_angular_error,
        compute_oks,
        compute_pck,
        mask_iou,
    )
    __all__ = [
        "FiberEvaluator", "mean_absolute_error", "root_mean_squared_error",
        "mean_angular_error", "compute_oks", "compute_pck", "mask_iou",
    ]
except ModuleNotFoundError:
    # detectron2 not installed — import individual helpers that don't need it
    from .fiber_evaluator import (  # type: ignore
        mean_absolute_error,
        root_mean_squared_error,
        mean_angular_error,
        compute_oks,
        compute_pck,
        mask_iou,
    )
    __all__ = [
        "mean_absolute_error", "root_mean_squared_error",
        "mean_angular_error", "compute_oks", "compute_pck", "mask_iou",
    ]
