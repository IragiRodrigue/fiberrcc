"""
COCO-Fiber Dataset Mapper
=========================
Detectron2-compatible dataset registration and mapper for the COCO-Fiber
annotation format.  Extends the standard COCO mapper to also load:

* 40 ordered keypoints (centerline)
* fiber_width / fiber_length / fiber_curvature / fiber_orientation / fiber_tortuosity
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import (
    BitMasks,
    Boxes,
    Instances,
    Keypoints,
    polygons_to_bitmask,
)

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 40
KEYPOINT_NAMES = [f"kp{i:02d}" for i in range(NUM_KEYPOINTS)]
KEYPOINT_FLIP_MAP: list[tuple[str, str]] = []  # fibers have no chirality


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_coco_fiber_json(
    json_file: str | Path,
    image_root: str | Path,
) -> list[dict[str, Any]]:
    """Load a COCO-Fiber JSON annotation file into Detectron2's dataset format.

    Each dict in the returned list corresponds to one image and contains
    ``annotations`` with fiber-specific fields.
    """
    json_file = Path(json_file)
    image_root = Path(image_root)

    with open(json_file, "r", encoding="utf-8") as fh:
        coco_data = json.load(fh)

    # Build id → image map
    id_to_image: dict[int, dict[str, Any]] = {
        img["id"]: img for img in coco_data["images"]
    }

    # Group annotations by image_id
    from collections import defaultdict

    id_to_anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        id_to_anns[ann["image_id"]].append(ann)

    dataset_dicts: list[dict[str, Any]] = []

    for img_id, img_info in id_to_image.items():
        record: dict[str, Any] = {
            "file_name": str(image_root / img_info["file_name"]),
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
        }

        anns = id_to_anns.get(img_id, [])
        objs: list[dict[str, Any]] = []
        for ann in anns:
            obj: dict[str, Any] = {
                "bbox": ann["bbox"],
                "bbox_mode": 1,  # BoxMode.XYWH_ABS
                "segmentation": ann["segmentation"],
                "category_id": 0,  # single class
                "keypoints": ann.get("keypoints", []),
                "fiber_width": ann.get("fiber_width", 0.0),
                "fiber_length": ann.get("fiber_length", 0.0),
                "fiber_curvature": ann.get("fiber_curvature", 0.0),
                "fiber_orientation": ann.get("fiber_orientation", 0.0),
                "fiber_tortuosity": ann.get("fiber_tortuosity", 1.0),
                "iscrowd": ann.get("iscrowd", 0),
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    logger.info(
        f"Loaded {len(dataset_dicts)} images with "
        f"{sum(len(d['annotations']) for d in dataset_dicts)} fiber annotations "
        f"from {json_file}"
    )
    return dataset_dicts


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def register_coco_fiber_dataset(
    name: str,
    json_file: str | Path,
    image_root: str | Path,
) -> None:
    """Register a COCO-Fiber split with Detectron2's global catalog."""
    DatasetCatalog.register(
        name,
        lambda jf=json_file, ir=image_root: load_coco_fiber_json(jf, ir),
    )
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        image_root=str(image_root),
        evaluator_type="coco",
        thing_classes=["fiber"],
        thing_colors=[(0, 200, 83)],
        keypoint_names=KEYPOINT_NAMES,
        keypoint_flip_map=KEYPOINT_FLIP_MAP,
        keypoint_connection_rules=[
            (KEYPOINT_NAMES[i], KEYPOINT_NAMES[i + 1], (255, 128, 0))
            for i in range(NUM_KEYPOINTS - 1)
        ],
    )
    logger.info(f"Registered dataset: {name}")


# ---------------------------------------------------------------------------
# Dataset mapper
# ---------------------------------------------------------------------------

class FiberDatasetMapper:
    """Maps a COCO-Fiber dataset dict to a Detectron2 model input dict.

    Parameters
    ----------
    cfg : CfgNode
        Detectron2 config.
    is_train : bool
        Whether training augmentations should be applied.
    """

    def __init__(self, cfg: CfgNode, is_train: bool = True) -> None:
        self.is_train = is_train
        self.augmentations = self._build_augmentations(cfg, is_train)
        self.image_format: str = cfg.INPUT.FORMAT
        self.use_instance_mask: bool = cfg.MODEL.MASK_ON
        self.use_keypoint: bool = cfg.MODEL.KEYPOINT_ON
        self.keypoint_hflip_indices: list[int] | None = None

        logger.info(
            f"FiberDatasetMapper: is_train={is_train}, "
            f"mask={self.use_instance_mask}, keypoint={self.use_keypoint}"
        )

    @staticmethod
    def _build_augmentations(cfg: CfgNode, is_train: bool) -> list[T.Augmentation]:
        augs: list[T.Augmentation] = []
        if is_train:
            augs.append(
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            )
            if cfg.INPUT.RANDOM_FLIP != "none":
                augs.append(T.RandomFlip(horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal")))
        else:
            augs.append(
                T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST],
                    cfg.INPUT.MAX_SIZE_TEST,
                )
            )
        return augs

    def __call__(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # (H, W)

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        anns = dataset_dict.pop("annotations", [])
        anns = [obj for obj in anns if obj.get("iscrowd", 0) == 0]

        instances = self._annotations_to_instances(anns, image_shape, transforms)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    def _annotations_to_instances(
        self,
        anns: list[dict[str, Any]],
        image_shape: tuple[int, int],
        transforms: T.TransformList,
    ) -> Instances:
        from detectron2.structures import BoxMode

        boxes = [
            BoxMode.convert(obj["bbox"], BoxMode(obj["bbox_mode"]), BoxMode.XYXY_ABS)
            for obj in anns
        ]
        target = Instances(image_shape)

        # Boxes
        boxes_t = transforms.apply_box(np.array(boxes, dtype=np.float32))
        target.gt_boxes = Boxes(boxes_t)

        # Classes
        classes = [int(obj["category_id"]) for obj in anns]
        target.gt_classes = torch.tensor(classes, dtype=torch.int64)

        # Masks
        if self.use_instance_mask:
            masks = [
                transforms.apply_segmentation(
                    polygons_to_bitmask(obj["segmentation"], *image_shape)
                )
                for obj in anns
            ]
            target.gt_masks = BitMasks(
                torch.stack([torch.from_numpy(m) for m in masks])
            )

        # Keypoints (40 per fiber)
        if self.use_keypoint and anns and "keypoints" in anns[0]:
            kps_list = []
            for obj in anns:
                kps_flat = obj["keypoints"]  # [x0,y0,v0, x1,y1,v1, ...]
                kps = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)
                kps = transforms.apply_coords(kps[:, :2])
                kps_list.append(
                    np.concatenate(
                        [kps, np.ones((len(kps), 1), dtype=np.float32) * 2], axis=1
                    )
                )
            target.gt_keypoints = Keypoints(np.stack(kps_list))

        # Fiber regression targets
        for field_name in (
            "fiber_width",
            "fiber_length",
            "fiber_curvature",
            "fiber_orientation",
            "fiber_tortuosity",
        ):
            values = [float(obj.get(field_name, 0.0)) for obj in anns]
            setattr(
                target,
                f"gt_{field_name}",
                torch.tensor(values, dtype=torch.float32),
            )

        return target
