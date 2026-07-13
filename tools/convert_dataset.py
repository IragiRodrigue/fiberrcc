#!/usr/bin/env python3
"""
convert_dataset.py
==================
Convert a directory of LabelMe annotations to COCO-Fiber format.

Usage examples
--------------
# Convert all files into a single JSON:
python tools/convert_dataset.py \\
    --labelme_dir /data/raw_annotations \\
    --output      /data/coco_fiber/all.json

# Convert and split into train / val / test:
python tools/convert_dataset.py \\
    --labelme_dir /data/raw_annotations \\
    --output_dir  /data/coco_fiber \\
    --split \\
    --train_ratio 0.8 \\
    --val_ratio   0.1 \\
    --seed        42

# Verify the output:
python tools/convert_dataset.py \\
    --verify /data/coco_fiber/train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LabelMe polygon annotations to COCO-Fiber format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--labelme_dir",
        type=Path,
        default=None,
        help="Directory containing LabelMe JSON files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for a single COCO-Fiber JSON (no split).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory when --split is used.",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Split into train / val / test subsets.",
    )
    p.add_argument("--train_ratio", type=float, default=0.8, help="Fraction for training.")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Fraction for validation.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--n_keypoints",
        type=int,
        default=40,
        help="Number of ordered centerline keypoints per fiber.",
    )
    p.add_argument(
        "--fiber_label",
        type=str,
        default="fiber",
        help="LabelMe label string to process.",
    )
    p.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Path to an existing COCO-Fiber JSON to verify.",
    )
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _verify_json(path: Path) -> None:
    """Print basic statistics of a COCO-Fiber JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    n_images = len(data.get("images", []))
    n_anns = len(data.get("annotations", []))
    cats = [c["name"] for c in data.get("categories", [])]

    logger.info(f"=== Verification: {path} ===")
    logger.info(f"  Images      : {n_images}")
    logger.info(f"  Annotations : {n_anns}")
    logger.info(f"  Categories  : {cats}")

    if n_anns > 0:
        ann = data["annotations"][0]
        logger.info(f"  Sample annotation keys: {list(ann.keys())}")
        logger.info(f"  Keypoints length: {len(ann.get('keypoints', []))}")
        logger.info(f"  fiber_width  : {ann.get('fiber_width')}")
        logger.info(f"  fiber_length : {ann.get('fiber_length')}")
        logger.info(f"  fiber_orientation : {ann.get('fiber_orientation')}")
        logger.info(f"  fiber_tortuosity  : {ann.get('fiber_tortuosity')}")

    logger.success("Verification passed.")


def main() -> None:
    args = _parse_args()

    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Verify mode
    if args.verify is not None:
        if not args.verify.exists():
            logger.error(f"File not found: {args.verify}")
            sys.exit(1)
        _verify_json(args.verify)
        return

    # Require labelme_dir for conversion
    if args.labelme_dir is None:
        logger.error("--labelme_dir is required for conversion.")
        sys.exit(1)

    from fiberrcnn.data.converter import LabelMeToCOCOFiber

    converter = LabelMeToCOCOFiber(
        n_keypoints=args.n_keypoints,
        fiber_label=args.fiber_label,
    )

    if args.split:
        if args.output_dir is None:
            logger.error("--output_dir is required when using --split.")
            sys.exit(1)
        paths = converter.split(
            labelme_dir=args.labelme_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        for split_name, p in paths.items():
            logger.success(f"{split_name:5s} → {p}")
    else:
        if args.output is None:
            logger.error("--output is required when not using --split.")
            sys.exit(1)
        converter.convert(
            labelme_dir=args.labelme_dir,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
