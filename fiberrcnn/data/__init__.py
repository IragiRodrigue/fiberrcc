"""Data loading, registration, and conversion."""

from .converter import LabelMeToCOCOFiber, COCOFiberDataset

try:
    from .dataset_mapper import (
        FiberDatasetMapper,
        register_coco_fiber_dataset,
        load_coco_fiber_json,
    )
    __all__ = [
        "LabelMeToCOCOFiber",
        "COCOFiberDataset",
        "load_coco_fiber_json",
        "FiberDatasetMapper",
        "register_coco_fiber_dataset",
    ]
except ModuleNotFoundError:
    # detectron2 not installed
    __all__ = ["LabelMeToCOCOFiber", "COCOFiberDataset"]
