"""
COCO-Fiber Dataset Mapper
=========================
Detectron2-compatible dataset registration and mapper.
Uses polygon segmentations directly (not pre-rasterised masks)
so Detectron2's transforms can handle them correctly.
"""

from __future__ import annotations

import copy
import json
import logging
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
    PolygonMasks,
)

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 40
KEYPOINT_NAMES = [f"kp{i:02d}" for i in range(NUM_KEYPOINTS)]
KEYPOINT_FLIP_MAP: list[tuple[str, str]] = []


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_coco_fiber_json(
    json_file: str | Path,
    image_root: str | Path,
) -> list[dict[str, Any]]:
    """Load a COCO-Fiber JSON into Detectron2's dataset format."""
    from collections import defaultdict

    json_file = Path(json_file)
    image_root = Path(image_root)

    with open(json_file, "r", encoding="utf-8") as fh:
        coco_data = json.load(fh)

    id_to_image = {img["id"]: img for img in coco_data["images"]}
    id_to_anns: dict[int, list] = defaultdict(list)
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

        objs = []
        for ann in id_to_anns.get(img_id, []):
            obj: dict[str, Any] = {
                "bbox": ann["bbox"],
                "bbox_mode": 1,  # XYWH_ABS
                # Keep segmentation as polygon list — NOT pre-rasterised
                "segmentation": ann["segmentation"],
                "category_id": 0,
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
        f"Loaded {len(dataset_dicts)} images, "
        f"{sum(len(d['annotations']) for d in dataset_dicts)} annotations "
        f"from {json_file.name}"
    )
    return dataset_dicts


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_coco_fiber_dataset(
    name: str,
    json_file: str | Path,
    image_root: str | Path,
) -> None:
    DatasetCatalog.register(
        name,
        lambda jf=json_file, ir=image_root: load_coco_fiber_json(jf, ir),
    )
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        image_root=str(image_root),
        evaluator_type="coco",
        thing_classes=["fiber"],
        keypoint_names=KEYPOINT_NAMES,
        keypoint_flip_map=KEYPOINT_FLIP_MAP,
    )
    logger.info(f"Registered dataset: {name}")


# ---------------------------------------------------------------------------
# Dataset mapper
# ---------------------------------------------------------------------------

class FiberDatasetMapper:
    """Maps a COCO-Fiber dataset dict to a Detectron2 model input dict."""

    def __init__(self, cfg: CfgNode, is_train: bool = True) -> None:
        self.is_train = is_train
        self.image_format: str = cfg.INPUT.FORMAT

        if is_train:
            self.augmentations = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                T.RandomFlip(horizontal=True),
            ]
        else:
            self.augmentations = [
                T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                    cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                ),
            ]

    def __call__(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Apply augmentations to image only
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        anns = dataset_dict.pop("annotations", [])
        anns = [a for a in anns if a.get("iscrowd", 0) == 0]
        if not anns:
            dataset_dict["instances"] = Instances(image.shape[:2])
            return dataset_dict

        image_shape = image.shape[:2]  # (H, W) after resize

        # ── Boxes ──────────────────────────────────────────────────────
        from detectron2.structures import BoxMode
        boxes = [
            BoxMode.convert(a["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for a in anns
        ]
        boxes_t = transforms.apply_box(np.array(boxes, dtype=np.float32))
        # Clip to image bounds
        H, W = image_shape
        boxes_t = np.clip(boxes_t, [0, 0, 0, 0], [W, H, W, H])

        target = Instances(image_shape)
        target.gt_boxes = Boxes(torch.tensor(boxes_t, dtype=torch.float32))
        target.gt_classes = torch.zeros(len(anns), dtype=torch.int64)

        # ── Masks — use PolygonMasks, transform coords only ────────────
        polys = []
        for ann in anns:
            seg = ann.get("segmentation", [[]])
            # Each seg is a list of [x0,y0,x1,y1,...] flat arrays
            transformed_polys = []
            for poly_flat in seg:
                pts = np.array(poly_flat, dtype=np.float64).reshape(-1, 2)
                pts = transforms.apply_coords(pts)
                transformed_polys.append(pts.flatten().tolist())
            polys.append(transformed_polys)
        target.gt_masks = PolygonMasks(polys)

        # ── Keypoints ──────────────────────────────────────────────────
        # Keypoints are stored normalised [0,1] in the JSON.
        # We keep them in [0,1] space — NO apply_coords (which would
        # convert back to absolute pixels and cause loss explosion).
        kps_list = []
        has_kps = any(a.get("keypoints") for a in anns)
        if has_kps:
            for ann in anns:
                kps_flat = ann.get("keypoints", [])
                if kps_flat:
                    kps = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)
                    # xy already in [0,1] — just keep as-is
                    kps_list.append(kps)
                else:
                    kps_list.append(np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32))
            target.gt_keypoints = Keypoints(np.stack(kps_list))

        # ── Fiber regression targets ───────────────────────────────────
        # Values are already normalised to ~[0,1] by the converter.
        for field_name in (
            "fiber_width", "fiber_length", "fiber_curvature",
            "fiber_orientation", "fiber_tortuosity",
        ):
            vals = [float(a.get(field_name, 0.0)) for a in anns]
            setattr(target, f"gt_{field_name}",
                    torch.tensor(vals, dtype=torch.float32))

        dataset_dict["instances"] = utils.filter_empty_instances(target)
        return dataset_dict