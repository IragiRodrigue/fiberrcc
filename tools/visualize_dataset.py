#!/usr/bin/env python3
"""
visualize_dataset.py
=====================
Inspect and visualise a COCO-Fiber dataset annotation file before training.
Draws instances masks, centerlines, keypoints, and prints per-image statistics.

Usage
-----
# Visualise all images (saved to output directory):
python tools/visualize_dataset.py \\
    --json       /data/coco_fiber/train.json \\
    --image_root /data/images \\
    --output_dir ./viz_train \\
    --max_images 50

# Visualise a single image by ID:
python tools/visualize_dataset.py \\
    --json       /data/coco_fiber/train.json \\
    --image_root /data/images \\
    --image_id   7 \\
    --output_dir ./viz_train

# Print per-image statistics without saving images:
python tools/visualize_dataset.py \\
    --json /data/coco_fiber/train.json \\
    --stats_only

# Show orientation rose plot for the full dataset:
python tools/visualize_dataset.py \\
    --json /data/coco_fiber/train.json \\
    --rose --output_dir ./viz_train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from fiberrcnn.geometry import polygon_to_mask, extract_centerline, resample_centerline
from fiberrcnn.visualization import (
    draw_instance_overlay,
    draw_centerlines,
    plot_histogram,
    plot_rose,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _keypoints_to_array(kps_flat: list[float]) -> np.ndarray:
    """Flatten [x,y,v, x,y,v, ...] → (K, 2)."""
    arr = np.array(kps_flat, dtype=np.float32).reshape(-1, 3)
    return arr[:, :2]


def _print_dataset_stats(data: dict) -> None:
    """Print summary statistics for a COCO-Fiber JSON dataset."""
    images = data.get("images", [])
    anns = data.get("annotations", [])

    n_imgs = len(images)
    n_anns = len(anns)
    anns_per_img = n_anns / max(n_imgs, 1)

    widths = [a.get("fiber_width", 0) for a in anns if a.get("fiber_width", 0) > 0]
    lengths = [a.get("fiber_length", 0) for a in anns if a.get("fiber_length", 0) > 0]
    orients = [a.get("fiber_orientation", 0) for a in anns]
    torts = [a.get("fiber_tortuosity", 1) for a in anns]
    curvs = [a.get("fiber_curvature", 0) for a in anns]

    def _fmt(vals: list[float], name: str) -> str:
        if not vals:
            return f"  {name}: N/A"
        a = np.array(vals)
        return (
            f"  {name:25s} mean={a.mean():.2f}  std={a.std():.2f}  "
            f"min={a.min():.2f}  max={a.max():.2f}"
        )

    print(f"\n{'='*55}")
    print(f"  Dataset Statistics")
    print(f"{'='*55}")
    print(f"  Images      : {n_imgs}")
    print(f"  Annotations : {n_anns}  ({anns_per_img:.1f} fibers/image)")
    print(_fmt(widths, "fiber_width (px)"))
    print(_fmt(lengths, "fiber_length (px)"))
    print(_fmt(orients, "fiber_orientation (°)"))
    print(_fmt(torts, "fiber_tortuosity"))
    print(_fmt(curvs, "fiber_curvature (1/px)"))
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Per-image visualisation
# ---------------------------------------------------------------------------

def visualise_image(
    image_info: dict,
    anns: list[dict],
    image_root: Path,
    output_dir: Path,
) -> None:
    """Draw and save all annotation visualisations for one image."""
    fname = image_root / image_info["file_name"]
    if not fname.exists():
        logger.warning(f"Image not found: {fname}")
        return

    bgr = cv2.imread(str(fname))
    if bgr is None:
        logger.warning(f"Could not read: {fname}")
        return

    H, W = image_info["height"], image_info["width"]
    stem = Path(image_info["file_name"]).stem
    out_sub = output_dir / stem
    out_sub.mkdir(parents=True, exist_ok=True)

    masks: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []
    keypoints_list: list[np.ndarray] = []

    for ann in anns:
        # Mask from polygon
        seg = ann.get("segmentation", [[]])
        pts = np.array(seg[0], dtype=np.float32).reshape(-1, 2).tolist()
        if len(pts) < 3:
            continue
        mask = polygon_to_mask(pts, H, W)
        masks.append(mask)

        # Centerline
        cl = extract_centerline(mask)
        centerlines.append(cl)

        # Keypoints
        kps_flat = ann.get("keypoints", [])
        if kps_flat:
            kps = _keypoints_to_array(kps_flat)
            keypoints_list.append(kps)
        else:
            kps = resample_centerline(cl, 40)
            keypoints_list.append(kps)

    if not masks:
        return

    # 1. Instance overlay
    overlay = draw_instance_overlay(bgr, masks)
    cv2.imwrite(str(out_sub / "overlay.png"), overlay)

    # 2. Centerlines + keypoints
    cl_img = draw_centerlines(bgr, centerlines, keypoints_list)
    cv2.imwrite(str(out_sub / "centerlines.png"), cl_img)

    # 3. Side-by-side comparison
    combined = np.hstack([overlay, cl_img])
    cv2.imwrite(str(out_sub / "combined.png"), combined)

    # 4. Per-image stats on the overlay
    n = len(anns)
    widths = [a.get("fiber_width", 0) for a in anns]
    lengths = [a.get("fiber_length", 0) for a in anns]
    orients = [a.get("fiber_orientation", 0) for a in anns]

    stats_text = [
        f"Fibers: {n}",
        f"W: {np.mean(widths):.1f}px" if widths else "",
        f"L: {np.mean(lengths):.1f}px" if lengths else "",
        f"θ: {np.mean(orients):.1f}°" if orients else "",
    ]
    y_pos = 20
    for line in stats_text:
        if line:
            cv2.putText(overlay, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y_pos += 22
    cv2.imwrite(str(out_sub / "overlay_stats.png"), overlay)

    logger.debug(f"  Saved → {out_sub}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise a COCO-Fiber dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json", type=Path, required=True, help="COCO-Fiber JSON file.")
    p.add_argument("--image_root", type=Path, default=None,
                   help="Directory containing images (required unless --stats_only).")
    p.add_argument("--output_dir", type=Path, default=Path("./viz_output"),
                   help="Output directory for visualisations.")
    p.add_argument("--image_id", type=int, default=None,
                   help="Visualise only this image ID.")
    p.add_argument("--max_images", type=int, default=100,
                   help="Maximum number of images to process.")
    p.add_argument("--stats_only", action="store_true",
                   help="Print statistics only, do not save images.")
    p.add_argument("--rose", action="store_true",
                   help="Save a rose plot for the full dataset orientation distribution.")
    p.add_argument("--histograms", action="store_true",
                   help="Save dataset-level histograms (width, length, orientation).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if not args.json.exists():
        logger.error(f"JSON not found: {args.json}")
        sys.exit(1)

    data = _load_json(args.json)
    _print_dataset_stats(data)

    if args.stats_only:
        return

    if args.image_root is None:
        logger.error("--image_root is required unless --stats_only is set.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build image_id → annotations lookup
    id_to_anns: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        id_to_anns[ann["image_id"]].append(ann)

    images = data.get("images", [])
    if args.image_id is not None:
        images = [img for img in images if img["id"] == args.image_id]
        if not images:
            logger.error(f"Image ID {args.image_id} not found in dataset.")
            sys.exit(1)
    else:
        images = images[: args.max_images]

    logger.info(f"Visualising {len(images)} image(s) → {args.output_dir}")

    for img_info in tqdm(images, desc="Visualising"):
        anns = id_to_anns.get(img_info["id"], [])
        visualise_image(img_info, anns, args.image_root, args.output_dir)

    # Dataset-level plots
    all_anns = data.get("annotations", [])
    all_widths = [a.get("fiber_width", 0) for a in all_anns if a.get("fiber_width", 0) > 0]
    all_lengths = [a.get("fiber_length", 0) for a in all_anns if a.get("fiber_length", 0) > 0]
    all_orients = [a.get("fiber_orientation", 0) for a in all_anns]
    all_torts = [a.get("fiber_tortuosity", 1) for a in all_anns]

    if args.rose and all_orients:
        fig = plot_rose(all_orients, title="Dataset Fiber Orientation",
                        output_path=args.output_dir / "dataset_rose_orientation.png")
        plt.close(fig)
        logger.info("Saved rose plot.")

    if args.histograms:
        for vals, title, xlabel, fname in [
            (all_widths,  "Fiber Width Distribution",   "Width (px)",       "hist_width.png"),
            (all_lengths, "Fiber Length Distribution",  "Length (px)",      "hist_length.png"),
            (all_orients, "Fiber Orientation",          "Orientation (°)",  "hist_orientation.png"),
            (all_torts,   "Fiber Tortuosity",           "Tortuosity",       "hist_tortuosity.png"),
        ]:
            if vals:
                fig = plot_histogram(vals, title, xlabel,
                                     output_path=args.output_dir / fname)
                plt.close(fig)
        logger.info("Saved histograms.")

    logger.success(f"Done. Outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
