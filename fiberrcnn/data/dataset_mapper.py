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
    Boxes,
    Instances,
    Keypoints,
    PolygonMasks,
)

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 40
KEYPOINT_NAMES = [f"kp{i:02d}" for i in range(NUM_KEYPOINTS)]
KEYPOINT_FLIP_MAP: list[tuple[str, str]] = []
_DEBUG_TARGETS = os.getenv("FIBERRCNN_DEBUG_TARGETS", "0") == "1"
_DEBUG_TARGET_LIMIT = int(os.getenv("FIBERRCNN_DEBUG_TARGET_LIMIT", "3"))
_debug_target_logs_remaining = _DEBUG_TARGET_LIMIT


def _canonicalize_keypoint_order(keypoints: np.ndarray) -> np.ndarray:
    """Keep centerline keypoints in a stable start-to-end order after augmentation."""
    if keypoints.shape[0] < 2:
        return keypoints

    first_xy = keypoints[0, :2]
    last_xy = keypoints[-1, :2]
    if (first_xy[0] > last_xy[0]) or (
        np.isclose(first_xy[0], last_xy[0]) and first_xy[1] > last_xy[1]
    ):
        return keypoints[::-1].copy()
    return keypoints


def _log_debug_targets(
    dataset_dict: dict[str, Any],
    image_shape: tuple[int, int],
    anns: list[dict[str, Any]],
    polys: list[list[list[float]]],
    kps_list: list[np.ndarray],
) -> None:
    global _debug_target_logs_remaining
    if not _DEBUG_TARGETS or _debug_target_logs_remaining <= 0:
        return

    _debug_target_logs_remaining -= 1
    H, W = image_shape
    logger.warning(
        "DEBUG_TARGETS image=%s transformed_hw=(%s,%s) n_anns=%s",
        dataset_dict.get("file_name", "<unknown>"),
        H,
        W,
        len(anns),
    )
    for idx, ann in enumerate(anns[:3]):
        bbox = ann.get("bbox", [0, 0, 0, 0])
        seg = polys[idx] if idx < len(polys) else []
        kp = kps_list[idx] if idx < len(kps_list) else np.zeros((0, 3), dtype=np.float32)
        n_polys = len(seg)
        n_pts = sum(len(poly) // 2 for poly in seg)
        if kp.size:
            kp_x_span = float(kp[:, 0].max() - kp[:, 0].min())
            kp_y_span = float(kp[:, 1].max() - kp[:, 1].min())
        else:
            kp_x_span = 0.0
            kp_y_span = 0.0
        logger.warning(
            "DEBUG_TARGETS ann=%s bbox_xywh=%s polys=%s pts=%s kp_span_norm=(%.4f, %.4f)",
            idx,
            [round(float(v), 2) for v in bbox],
            n_polys,
            n_pts,
            kp_x_span,
            kp_y_span,
        )


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
            dataset_category_id = int(ann.get("category_id", 1))
            obj: dict[str, Any] = {
                "bbox": ann["bbox"],
                "bbox_mode": 1,  # XYWH_ABS
                "segmentation": ann["segmentation"],
                "category_id": max(dataset_category_id - 1, 0),
                "keypoints": ann.get("keypoints", []),
                "fiber_width": ann.get("fiber_width", 0.0),
                "fiber_length": ann.get("fiber_length", 0.0),
                "fiber_curvature": ann.get("fiber_curvature", 0.0),
                "fiber_orientation": ann.get("fiber_orientation", 0.0),
                "fiber_tortuosity": ann.get("fiber_tortuosity", 1.0),
                "fiber_width_px": ann.get("fiber_width_px"),
                "fiber_length_px": ann.get("fiber_length_px"),
                "fiber_orientation_deg": ann.get("fiber_orientation_deg"),
                "fiber_tortuosity_raw": ann.get("fiber_tortuosity_raw"),
                "has_bead": ann.get("has_bead", False),
                "is_blurry": ann.get("is_blurry", False),
                "is_crossing": ann.get("is_crossing", False),
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
        thing_dataset_id_to_contiguous_id={1: 0},
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

        image_shape = image.shape[:2]
        H, W = image_shape

        from detectron2.structures import BoxMode

        boxes = [
            BoxMode.convert(a["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for a in anns
        ]
        boxes_t = transforms.apply_box(np.array(boxes, dtype=np.float32))
        boxes_t = np.clip(boxes_t, [0, 0, 0, 0], [W, H, W, H])

        target = Instances(image_shape)
        target.gt_boxes = Boxes(torch.tensor(boxes_t, dtype=torch.float32))
        target.gt_classes = torch.zeros(len(anns), dtype=torch.int64)

        polys = []
        for ann in anns:
            seg = ann.get("segmentation", [[]])
            transformed_polys = []
            for poly_flat in seg:
                pts = np.array(poly_flat, dtype=np.float64).reshape(-1, 2)
                pts = transforms.apply_coords(pts)
                transformed_polys.append(pts.flatten().tolist())
            polys.append(transformed_polys)
        target.gt_masks = PolygonMasks(polys)

        kps_list = []
        has_kps = any(a.get("keypoints") for a in anns)
        if has_kps:
            for ann in anns:
                kps_flat = ann.get("keypoints", [])
                if kps_flat:
                    kps = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)
                    abs_xy = kps[:, :2].copy()
                    abs_xy[:, 0] *= max(dataset_dict["width"], 1)
                    abs_xy[:, 1] *= max(dataset_dict["height"], 1)
                    abs_xy = transforms.apply_coords(abs_xy)
                    abs_xy[:, 0] = np.clip(abs_xy[:, 0], 0.0, max(W - 1, 0))
                    abs_xy[:, 1] = np.clip(abs_xy[:, 1], 0.0, max(H - 1, 0))
                    kps[:, :2] = abs_xy
                    kps = _canonicalize_keypoint_order(kps)
                    kps[:, 0] /= max(W, 1)
                    kps[:, 1] /= max(H, 1)
                    kps_list.append(kps)
                else:
                    kps_list.append(np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32))
            target.gt_keypoints = Keypoints(np.stack(kps_list))

        _log_debug_targets(dataset_dict, image_shape, anns, polys, kps_list)

        for field_name in (
            "fiber_width",
            "fiber_length",
            "fiber_curvature",
            "fiber_orientation",
            "fiber_tortuosity",
        ):
            vals = [float(a.get(field_name, 0.0)) for a in anns]
            setattr(target, f"gt_{field_name}", torch.tensor(vals, dtype=torch.float32))

        for field_name in ("has_bead", "is_blurry", "is_crossing"):
            vals = [float(bool(a.get(field_name, False))) for a in anns]
            setattr(target, f"gt_{field_name}", torch.tensor(vals, dtype=torch.float32))

        dataset_dict["instances"] = utils.filter_empty_instances(target)
        return dataset_dict
