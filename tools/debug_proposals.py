#!/usr/bin/env python3
"""
debug_proposals.py
==================
Measure how well RPN proposals and final detections cover GT boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure proposal and detection recall for FiberRCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--dataset_json", type=Path, required=True)
    p.add_argument("--image_root", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated IoU thresholds.",
    )
    p.add_argument("--limit", type=int, default=0, help="0 means all images.")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _summarize_best_ious(best_ious, thresholds):
    n = len(best_ious)
    if n == 0:
        return {
            "count": 0,
            "mean_best_iou": 0.0,
            "median_best_iou": 0.0,
            **{f"recall@{t:.2f}": 0.0 for t in thresholds},
        }
    best_sorted = sorted(best_ious)
    return {
        "count": n,
        "mean_best_iou": float(sum(best_ious) / n),
        "median_best_iou": float(best_sorted[n // 2]),
        **{
            f"recall@{t:.2f}": float(sum(v >= t for v in best_ious) / n)
            for t in thresholds
        },
    }


def main() -> None:
    args = _parse_args()
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())

    import cv2
    import numpy as np
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog
    from detectron2.data import transforms as T
    from detectron2.data import detection_utils as utils
    from detectron2.modeling import build_model
    from detectron2.structures import Boxes, Instances

    import fiberrcnn  # noqa: F401
    from fiberrcnn.data import register_coco_fiber_dataset
    from fiberrcnn.engine.inference import _add_fiber_cfg_defaults
    from fiberrcnn.evaluation.fiber_evaluator import _bbox_iou_matrix

    dataset_name = "fiber_debug_proposals"
    register_coco_fiber_dataset(dataset_name, args.dataset_json, args.image_root)

    cfg = get_cfg()
    _add_fiber_cfg_defaults(cfg)
    cfg.merge_from_file(str(args.config))
    if args.weights is not None:
        cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    resize_aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice",
    )

    dataset_dicts = DatasetCatalog.get(dataset_name)
    if args.limit > 0:
        dataset_dicts = dataset_dicts[: args.limit]

    per_image: list[dict[str, Any]] = []
    all_prop_best: list[float] = []
    all_det_best: list[float] = []

    for sample in dataset_dicts:
        image = utils.read_image(sample["file_name"], format=cfg.INPUT.FORMAT)
        aug_input = T.AugInput(image)
        resize_aug(aug_input)
        image_resized = aug_input.image
        resized_h, resized_w = image_resized.shape[:2]

        boxes_xyxy = []
        for ann in sample.get("annotations", []):
            x, y, w, h = ann["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
        if not boxes_xyxy:
            continue

        gt_boxes_np = aug_input.transform.apply_box(
            np.asarray(boxes_xyxy, dtype=np.float32)
        )
        gt_boxes_np = np.clip(
            gt_boxes_np,
            [0.0, 0.0, 0.0, 0.0],
            [resized_w, resized_h, resized_w, resized_h],
        )

        image_tensor = torch.as_tensor(
            np.ascontiguousarray(image_resized.transpose(2, 0, 1))
        )
        inputs = [
            {
                "image": image_tensor,
                "height": resized_h,
                "width": resized_w,
            }
        ]

        with torch.no_grad():
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features, None)
            pred_instances, _ = model.roi_heads(images, features, proposals, None)

        prop_boxes = proposals[0].proposal_boxes.tensor.detach().cpu().numpy()
        if args.topk > 0:
            prop_boxes = prop_boxes[: args.topk]

        det_boxes = pred_instances[0].pred_boxes.tensor.detach().cpu().numpy()

        prop_iou = _bbox_iou_matrix(prop_boxes, gt_boxes_np)
        det_iou = _bbox_iou_matrix(det_boxes, gt_boxes_np)

        prop_best = prop_iou.max(axis=0).tolist() if prop_iou.size else [0.0] * len(gt_boxes_np)
        det_best = det_iou.max(axis=0).tolist() if det_iou.size else [0.0] * len(gt_boxes_np)

        all_prop_best.extend(prop_best)
        all_det_best.extend(det_best)

        per_image.append(
            {
                "image_id": int(sample["image_id"]),
                "file_name": sample["file_name"],
                "gt_count": int(len(gt_boxes_np)),
                "proposal_count": int(len(prop_boxes)),
                "detection_count": int(len(det_boxes)),
                "proposals": _summarize_best_ious(prop_best, thresholds),
                "detections": _summarize_best_ious(det_best, thresholds),
            }
        )

    per_image.sort(
        key=lambda row: (
            row["detections"].get("recall@0.50", 0.0),
            row["proposals"].get("recall@0.50", 0.0),
            row["proposals"].get("mean_best_iou", 0.0),
        )
    )

    summary = {
        "config": str(args.config),
        "weights": str(args.weights or cfg.MODEL.WEIGHTS),
        "dataset_json": str(args.dataset_json),
        "image_root": str(args.image_root),
        "images_evaluated": len(per_image),
        "topk_proposals": args.topk,
        "thresholds": list(thresholds),
        "global": {
            "proposals": _summarize_best_ious(all_prop_best, thresholds),
            "detections": _summarize_best_ious(all_det_best, thresholds),
        },
        "worst_images": per_image[: min(10, len(per_image))],
        "per_image": per_image,
    }

    print("\n=== Proposal Recall Summary ===")
    for stage in ("proposals", "detections"):
        stats = summary["global"][stage]
        print(f"\n[{stage}]")
        print(f"  count           : {stats['count']}")
        print(f"  mean_best_iou   : {stats['mean_best_iou']:.4f}")
        print(f"  median_best_iou : {stats['median_best_iou']:.4f}")
        for t in thresholds:
            print(f"  recall@{t:.2f}     : {stats[f'recall@{t:.2f}']:.4f}")

    print("\n=== Worst Images (by detection recall@0.50) ===")
    for row in summary["worst_images"]:
        print(
            f"  {Path(row['file_name']).name}: "
            f"gt={row['gt_count']} prop_r50={row['proposals'].get('recall@0.50', 0.0):.3f} "
            f"det_r50={row['detections'].get('recall@0.50', 0.0):.3f} "
            f"prop_mean_iou={row['proposals'].get('mean_best_iou', 0.0):.3f} "
            f"det_mean_iou={row['detections'].get('mean_best_iou', 0.0):.3f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nSaved report -> {args.output}")


if __name__ == "__main__":
    main()
