#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate a trained FiberRCNN model on a COCO-Fiber test set.

Usage:
    python tools/evaluate.py \\
        --config     configs/fiber_rcnn_r50_fpn.yaml \\
        --weights    output/run01/model_final.pth \\
        --test_json  /data/coco_fiber/test.json \\
        --image_root /data/images \\
        --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate FiberRCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--test_json", type=Path, required=True)
    p.add_argument("--image_root", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import fiberrcnn  # register models
    from fiberrcnn.data import register_coco_fiber_dataset, load_coco_fiber_json
    from fiberrcnn.engine.inference import FiberPredictor
    from fiberrcnn.evaluation import FiberEvaluator
    from detectron2.data import DatasetCatalog, MetadataCatalog
    import torch
    import numpy as np

    # Register
    register_coco_fiber_dataset("fiber_test", args.test_json, args.image_root)

    predictor = FiberPredictor.from_config(
        cfg_path=args.config,
        weights_path=args.weights,
        score_thresh=args.threshold,
        device=args.device,
    )

    dataset_dicts = DatasetCatalog.get("fiber_test")
    evaluator = FiberEvaluator()

    logger.info(f"Evaluating on {len(dataset_dicts)} images …")

    import cv2
    from detectron2.structures import Instances, Boxes

    for sample in dataset_dicts:
        bgr = cv2.imread(sample["file_name"])
        if bgr is None:
            logger.warning(f"Could not read: {sample['file_name']}")
            continue

        pred = predictor.predict(bgr, run_morphology=False)
        pred_inst = pred.fiber_instances

        # Build ground-truth Instances from dataset dict
        H, W = sample["height"], sample["width"]
        gt = Instances((H, W))

        anns = sample.get("annotations", [])
        if anns:
            from detectron2.structures import BoxMode
            boxes = [
                BoxMode.convert(a["bbox"], BoxMode(a["bbox_mode"]), BoxMode.XYXY_ABS)
                for a in anns
            ]
            gt.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
            gt.gt_classes = torch.zeros(len(anns), dtype=torch.int64)

            for field_name in ("fiber_width", "fiber_length", "fiber_curvature",
                               "fiber_orientation", "fiber_tortuosity"):
                vals = [a.get(field_name, 0.0) for a in anns]
                setattr(gt, f"gt_{field_name}", torch.tensor(vals, dtype=torch.float32))
        else:
            gt.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            gt.gt_classes = torch.zeros((0,), dtype=torch.int64)

        # Convert pred to Instances for evaluator
        if pred_inst:
            boxes_pred = torch.tensor(
                [[f.bbox[0], f.bbox[1],
                  f.bbox[0]+f.bbox[2], f.bbox[1]+f.bbox[3]]
                 for f in pred_inst],
                dtype=torch.float32,
            )
            p_inst = Instances((H, W))
            p_inst.pred_boxes = Boxes(boxes_pred)
            p_inst.scores = torch.tensor([f.confidence for f in pred_inst])

            for attr in ("fiber_width", "fiber_length", "fiber_curvature",
                         "fiber_orientation", "fiber_tortuosity"):
                setattr(
                    p_inst,
                    f"pred_{attr}",
                    torch.tensor([getattr(f, attr) for f in pred_inst]),
                )
        else:
            p_inst = Instances((H, W))
            p_inst.pred_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            p_inst.scores = torch.zeros((0,), dtype=torch.float32)

        evaluator.process([p_inst], [gt], image_ids=[sample["image_id"]])

    results = evaluator.evaluate()

    # Save results
    results_path = args.output_dir / "metrics.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.success(f"Metrics saved → {results_path}")

    print("\n=== Evaluation Results ===")
    for k, v in sorted(results.items()):
        print(f"  {k:40s} {v:.4f}")


if __name__ == "__main__":
    main()
