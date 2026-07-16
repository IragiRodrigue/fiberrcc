#!/usr/bin/env python3
"""
infer.py
=========
Run FiberRCNN inference on one image or a directory of images.

Single image:
    python tools/infer.py \\
        --config  configs/fiber_rcnn_r50_fpn.yaml \\
        --weights output/run01/model_final.pth \\
        --input   /data/images/sem_001.png \\
        --output_dir ./results

Directory:
    python tools/infer.py \\
        --config     configs/fiber_rcnn_r50_fpn.yaml \\
        --weights    output/run01/model_final.pth \\
        --input_dir  /data/test_images \\
        --output_dir ./results \\
        --threshold  0.6 \\
        --no_morphology
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FiberRCNN inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--input", type=Path, default=None, help="Single image path.")
    p.add_argument("--input_dir", type=Path, default=None, help="Directory of images.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output directory.")
    p.add_argument(
        "--threshold", type=float, default=0.5, help="Detection confidence threshold."
    )
    p.add_argument(
        "--viz_threshold",
        type=float,
        default=None,
        help="Optional stricter threshold used only for saved visualisations.",
    )
    p.add_argument(
        "--no_morphology",
        action="store_true",
        help="Skip image-level morphological analysis.",
    )
    p.add_argument(
        "--no_visualise",
        action="store_true",
        help="Skip saving visualisation images.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--ext",
        nargs="+",
        default=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Image file extensions to process.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    import fiberrcnn  # register models
    from fiberrcnn.engine.inference import FiberPredictor
    from fiberrcnn.visualization import save_visualisation_report

    predictor = FiberPredictor.from_config(
        cfg_path=args.config,
        weights_path=args.weights,
        score_thresh=args.threshold,
        device=args.device,
    )

    image_paths: list[Path] = []
    if args.input is not None:
        image_paths = [args.input]
    elif args.input_dir is not None:
        for ext in args.ext:
            image_paths.extend(sorted(args.input_dir.glob(f"*.{ext}")))
            image_paths.extend(sorted(args.input_dir.glob(f"*.{ext.upper()}")))
    else:
        logger.error("Provide --input or --input_dir.")
        sys.exit(1)

    if not image_paths:
        logger.error("No images found.")
        sys.exit(1)

    logger.info(f"Processing {len(image_paths)} image(s) -> {args.output_dir}")

    for img_path in image_paths:
        logger.info(f"  {img_path.name}")
        pred = predictor.predict(
            img_path,
            run_morphology=not args.no_morphology,
        )
        pred.save_json(args.output_dir / f"{img_path.stem}_prediction.json")

        if not args.no_visualise:
            bgr = cv2.imread(str(img_path))
            if bgr is not None:
                viz_threshold = (
                    args.threshold if args.viz_threshold is None else args.viz_threshold
                )

                masks: list[np.ndarray] = []
                centerlines: list[np.ndarray] = []
                keypoints_list: list[np.ndarray] = []
                widths: list[float] = []
                lengths: list[float] = []
                orientations: list[float] = []
                curvatures: list[float] = []
                tortuosities: list[float] = []

                for i, fi in enumerate(pred.fiber_instances):
                    if fi.confidence < viz_threshold:
                        continue
                    if i >= len(pred.masks):
                        continue

                    mask = np.asarray(pred.masks[i], dtype=bool)
                    if mask.size == 0 or not mask.any():
                        continue

                    masks.append(mask)
                    centerlines.append(
                        np.asarray(pred.centerlines[i], dtype=np.float32)
                        if i < len(pred.centerlines)
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                    keypoints_list.append(
                        np.asarray(fi.keypoints, dtype=np.float32)
                        if fi.keypoints
                        else np.zeros((0, 2), dtype=np.float32)
                    )
                    widths.append(fi.fiber_width)
                    lengths.append(fi.fiber_length)
                    orientations.append(fi.fiber_orientation)
                    curvatures.append(fi.fiber_curvature)
                    tortuosities.append(fi.fiber_tortuosity)

                save_visualisation_report(
                    image=bgr,
                    masks=masks,
                    centerlines=centerlines,
                    keypoints_list=keypoints_list,
                    widths=widths,
                    lengths=lengths,
                    orientations=orientations,
                    curvatures=curvatures,
                    tortuosities=tortuosities,
                    output_dir=args.output_dir / img_path.stem,
                    image_name=img_path.stem,
                )

    logger.success(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
