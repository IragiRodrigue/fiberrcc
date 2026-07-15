"""
FiberRCNN Evaluation
=====================
Computes all evaluation metrics:

Detection:
    AP, AP50, AP75 (via pycocotools)

Segmentation:
    Mask mAP, IoU

Keypoints:
    OKS, PCK @ 0.05 / 0.10 / 0.20

Regression (width, length, curvature, tortuosity):
    MAE, RMSE

Orientation:
    Mean Angular Error (circular)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

try:
    from detectron2.evaluation import COCOEvaluator
    from detectron2.structures import Instances, polygons_to_bitmask
except ModuleNotFoundError:
    COCOEvaluator = None  # type: ignore
    Instances = None       # type: ignore
    polygons_to_bitmask = None  # type: ignore

logger = logging.getLogger(__name__)


def _instances_len(instances: Instances) -> int:
    """Return the number of elements in a Detectron2 ``Instances`` safely."""
    try:
        return len(instances)
    except NotImplementedError:
        fields = instances.get_fields()
        if not fields:
            return 0
        first_field = next(iter(fields.values()))
        return len(first_field)


def _bbox_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between XYXY boxes."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, p in enumerate(pred_boxes):
        px1, py1, px2, py2 = p
        p_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
        for j, g in enumerate(gt_boxes):
            gx1, gy1, gx2, gy2 = g
            g_area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)

            ix1 = max(px1, gx1)
            iy1 = max(py1, gy1)
            ix2 = min(px2, gx2)
            iy2 = min(py2, gy2)
            inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
            union = p_area + g_area - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious


def _greedy_match_by_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    min_iou: float = 0.1,
) -> list[tuple[int, int, float]]:
    """Greedily match predictions to GT using highest-IoU pairs first."""
    ious = _bbox_iou_matrix(pred_boxes, gt_boxes)
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    while ious.size:
        flat_idx = int(np.argmax(ious))
        best_iou = float(ious.flat[flat_idx])
        if best_iou < min_iou:
            break

        pred_idx, gt_idx = np.unravel_index(flat_idx, ious.shape)
        if pred_idx in used_pred or gt_idx in used_gt:
            ious[pred_idx, gt_idx] = -1.0
            continue

        matches.append((pred_idx, gt_idx, best_iou))
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        ious[pred_idx, :] = -1.0
        ious[:, gt_idx] = -1.0

    return matches


# ---------------------------------------------------------------------------
# Regression metrics helpers
# ---------------------------------------------------------------------------

def mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def root_mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(((pred - target) ** 2).mean()))


def mean_angular_error(pred_deg: np.ndarray, target_deg: np.ndarray) -> float:
    """Mean angular error for [0, 180) orientation values (in degrees)."""
    # Use circular difference on doubled angles
    diff = np.abs(pred_deg - target_deg) % 180.0
    diff = np.minimum(diff, 180.0 - diff)
    return float(diff.mean())


# ---------------------------------------------------------------------------
# Keypoint metrics
# ---------------------------------------------------------------------------

def compute_oks(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    bbox_area: float,
    sigma: float = 0.05,
) -> float:
    """Object Keypoint Similarity for a single instance.

    Parameters
    ----------
    pred_kps : (K, 2) predicted keypoint (x, y)
    gt_kps   : (K, 2) ground-truth keypoint (x, y)
    bbox_area : float, bounding box area in pixels²
    sigma    : per-keypoint standard deviation (default 0.05)

    Returns
    -------
    oks : float in [0, 1]
    """
    if bbox_area <= 0:
        return 0.0
    d_sq = ((pred_kps - gt_kps) ** 2).sum(axis=1)
    s_sq = (2.0 * sigma) ** 2 * (bbox_area + np.spacing(1)) * 2.0
    return float(np.exp(-d_sq / s_sq).mean())


def compute_pck(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    bbox_size: float,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """Percentage of Correct Keypoints at multiple thresholds.

    Parameters
    ----------
    pred_kps, gt_kps : (K, 2)
    bbox_size : reference size (max of bbox side)
    thresholds : fraction of bbox_size
    """
    d = np.sqrt(((pred_kps - gt_kps) ** 2).sum(axis=1))
    pck: dict[str, float] = {}
    for t in thresholds:
        pck[f"PCK@{t:.2f}"] = float((d < t * bbox_size).mean())
    return pck


# ---------------------------------------------------------------------------
# Segmentation IoU
# ---------------------------------------------------------------------------

def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary mask IoU."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / (union + 1e-6))


def _keypoints_xy_array(value: Any) -> np.ndarray | None:
    """Normalize keypoint containers to an ``(N, K, 2)`` numpy array."""
    if value is None:
        return None

    if hasattr(value, "tensor"):
        arr = value.tensor.cpu().numpy()
    else:
        arr = value.cpu().numpy()

    if arr.ndim != 3 or arr.shape[-1] < 2:
        return None
    return arr[:, :, :2]


# ---------------------------------------------------------------------------
# FiberEvaluator
# ---------------------------------------------------------------------------

class FiberEvaluator:
    """Collect predictions and compute all FiberRCNN evaluation metrics.

    Usage
    -----
    >>> evaluator = FiberEvaluator()
    >>> for batch_preds, batch_gts in dataloader:
    ...     evaluator.process(batch_preds, batch_gts)
    >>> results = evaluator.evaluate()
    """

    def __init__(self) -> None:
        self._predictions: list[dict[str, Any]] = []
        self._ground_truths: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._predictions.clear()
        self._ground_truths.clear()

    def process(
        self,
        predictions: list[Instances],
        ground_truths: list[Instances],
        image_ids: list[int] | None = None,
    ) -> None:
        """Accumulate predictions and ground truths for a batch of images.

        Parameters
        ----------
        predictions : list of Instances (one per image)
        ground_truths : list of Instances (one per image)
        image_ids : optional image identifiers
        """
        if image_ids is None:
            image_ids = list(range(len(predictions)))

        for img_id, preds, gts in zip(image_ids, predictions, ground_truths):
            self._predictions.append(
                {
                    "image_id": img_id,
                    "instances": preds,
                }
            )
            self._ground_truths.append(
                {
                    "image_id": img_id,
                    "instances": gts,
                }
            )

    def evaluate(self) -> dict[str, Any]:
        """Compute and return all metrics."""
        results: dict[str, Any] = {}

        # Collect arrays
        gt_widths: list[float] = []
        pred_widths: list[float] = []
        gt_lengths: list[float] = []
        pred_lengths: list[float] = []
        gt_curvs: list[float] = []
        pred_curvs: list[float] = []
        gt_orients: list[float] = []
        pred_orients: list[float] = []
        gt_torts: list[float] = []
        pred_torts: list[float] = []

        oks_scores: list[float] = []
        pck_scores: dict[str, list[float]] = defaultdict(list)
        iou_scores: list[float] = []
        matched_ious: list[float] = []

        for pred_entry, gt_entry in zip(self._predictions, self._ground_truths):
            p_inst: Instances = pred_entry["instances"]
            g_inst: Instances = gt_entry["instances"]

            n_pred = _instances_len(p_inst)
            n_gt = _instances_len(g_inst)
            if n_pred == 0 or n_gt == 0:
                continue

            pred_boxes = p_inst.pred_boxes.tensor.cpu().numpy()[:n_pred]
            gt_boxes = g_inst.gt_boxes.tensor.cpu().numpy()[:n_gt]
            matches = _greedy_match_by_iou(pred_boxes, gt_boxes)
            if not matches:
                continue

            pred_idx = np.array([m[0] for m in matches], dtype=np.int64)
            gt_idx = np.array([m[1] for m in matches], dtype=np.int64)
            matched_ious.extend([m[2] for m in matches])
            n_match = len(matches)

            # ---- Regression ----
            for attr_pred, attr_gt, p_list, g_list in [
                ("pred_fiber_width", "gt_fiber_width", pred_widths, gt_widths),
                ("pred_fiber_length", "gt_fiber_length", pred_lengths, gt_lengths),
                ("pred_fiber_curvature", "gt_fiber_curvature", pred_curvs, gt_curvs),
                ("pred_fiber_orientation", "gt_fiber_orientation", pred_orients, gt_orients),
                ("pred_fiber_tortuosity", "gt_fiber_tortuosity", pred_torts, gt_torts),
            ]:
                if hasattr(p_inst, attr_pred) and hasattr(g_inst, attr_gt):
                    p_vals = getattr(p_inst, attr_pred).cpu().numpy()[pred_idx]
                    g_vals = getattr(g_inst, attr_gt).cpu().numpy()[gt_idx]
                    p_list.extend(p_vals.tolist())
                    g_list.extend(g_vals.tolist())

            # ---- Keypoints ----
            if hasattr(p_inst, "pred_keypoints") and hasattr(g_inst, "gt_keypoints"):
                p_kps = _keypoints_xy_array(p_inst.pred_keypoints)
                g_kps = _keypoints_xy_array(g_inst.gt_keypoints)
                if p_kps is None or g_kps is None:
                    p_kps = None
                    g_kps = None

                if p_kps is not None and g_kps is not None:
                    p_kps = p_kps[pred_idx]
                    g_kps = g_kps[gt_idx]
                    gt_boxes_matched = gt_boxes[gt_idx]
                    bbox_areas = (gt_boxes_matched[:, 2] - gt_boxes_matched[:, 0]) * (
                        gt_boxes_matched[:, 3] - gt_boxes_matched[:, 1]
                    )
                    bbox_sizes = np.maximum(
                        gt_boxes_matched[:, 2] - gt_boxes_matched[:, 0],
                        gt_boxes_matched[:, 3] - gt_boxes_matched[:, 1],
                    )

                    for i in range(n_match):
                        oks = compute_oks(p_kps[i], g_kps[i], bbox_areas[i])
                        oks_scores.append(oks)
                        for k_str, v in compute_pck(p_kps[i], g_kps[i], bbox_sizes[i]).items():
                            pck_scores[k_str].append(v)

            # ---- Mask IoU ----
            if hasattr(p_inst, "pred_masks") and hasattr(g_inst, "gt_masks"):
                p_masks = (p_inst.pred_masks.cpu().numpy() > 0.5)[pred_idx]
                if hasattr(g_inst.gt_masks, "tensor"):
                    g_masks = g_inst.gt_masks.tensor.cpu().numpy()[gt_idx]
                else:
                    image_height, image_width = g_inst.image_size
                    g_masks = np.stack(
                        [
                            polygons_to_bitmask(
                                g_inst.gt_masks.polygons[i],
                                image_height,
                                image_width,
                            )
                            for i in gt_idx.tolist()
                        ]
                    )
                for p_m, g_m in zip(p_masks, g_masks):
                    iou_scores.append(mask_iou(p_m, g_m))

        # ---- Aggregate ----
        def _stat(
            pred_list: list[float], gt_list: list[float], name: str
        ) -> None:
            if pred_list:
                p = np.array(pred_list)
                g = np.array(gt_list)
                results[f"{name}/MAE"] = mean_absolute_error(p, g)
                results[f"{name}/RMSE"] = root_mean_squared_error(p, g)

        _stat(pred_widths, gt_widths, "width")
        _stat(pred_lengths, gt_lengths, "length")
        _stat(pred_curvs, gt_curvs, "curvature")
        _stat(pred_torts, gt_torts, "tortuosity")

        if pred_orients:
            results["orientation/AngularError"] = mean_angular_error(
                np.array(pred_orients), np.array(gt_orients)
            )

        if oks_scores:
            results["keypoints/OKS"] = float(np.mean(oks_scores))

        for k_str, v_list in pck_scores.items():
            results[f"keypoints/{k_str}"] = float(np.mean(v_list))

        if iou_scores:
            results["segmentation/mIoU"] = float(np.mean(iou_scores))
        if matched_ious:
            results["matching/MeanBBoxIoU"] = float(np.mean(matched_ious))
            results["matching/MatchedPairs"] = float(len(matched_ious))

        # Log
        for k, v in sorted(results.items()):
            logger.info(f"  {k:40s} {v:.4f}")

        return results
