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
        --no_morphology  # skip image-level metrics for speed
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

    import fiberrcnn  # register models
    from fiberrcnn.engine.inference import FiberPredictor
    from fiberrcnn.visualization import save_visualisation_report
    import cv2

    predictor = FiberPredictor.from_config(
        cfg_path=args.config,
        weights_path=args.weights,
        score_thresh=args.threshold,
        device=args.device,
    )

    # Collect image paths
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

    logger.info(f"Processing {len(image_paths)} image(s) → {args.output_dir}")

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
                import numpy as np

                masks = []
                centerlines = []
                keypoints_list = []

                from fiberrcnn.geometry import extract_centerline, resample_centerline

                for fi in pred.fiber_instances:
                    # Reconstruct a dummy mask from bbox for visualisation
                    # (real mask would come from pred.fiber_instances mask field
                    # if we stored it — here we use a placeholder)
                    H, W = pred.image_height, pred.image_width
                    m = np.zeros((H, W), dtype=bool)
                    x, y, w, h = [int(v) for v in fi.bbox]
                    m[y : y + h, x : x + w] = True
                    masks.append(m)

                    if fi.keypoints:
                        kps = np.array(fi.keypoints, dtype=np.float32)
                        centerlines.append(kps)
                        keypoints_list.append(kps)
                    else:
                        cl = extract_centerline(m)
                        centerlines.append(cl)
                        keypoints_list.append(resample_centerline(cl, 40))

                save_visualisation_report(
                    image=bgr,
                    masks=masks,
                    centerlines=centerlines,
                    keypoints_list=keypoints_list,
                    widths=[f.fiber_width for f in pred.fiber_instances],
                    lengths=[f.fiber_length for f in pred.fiber_instances],
                    orientations=[f.fiber_orientation for f in pred.fiber_instances],
                    curvatures=[f.fiber_curvature for f in pred.fiber_instances],
                    tortuosities=[f.fiber_tortuosity for f in pred.fiber_instances],
                    output_dir=args.output_dir / img_path.stem,
                    image_name=img_path.stem,
                )

    logger.success(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
