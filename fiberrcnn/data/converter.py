"""
LabelMe → COCO-Fiber Converter
================================
Reads a directory of LabelMe JSON annotation files and converts them to a
single COCO-Fiber JSON dataset including:

* instance segmentation masks
* bounding boxes
* 40 ordered keypoints
* fiber_width, fiber_length, fiber_curvature, fiber_orientation, fiber_tortuosity

Usage (CLI):
    python tools/convert_dataset.py \\
        --labelme_dir /data/raw \\
        --output      /data/coco_fiber/annotations.json \\
        --split_ratio 0.8

Usage (Python):
    from fiberrcnn.data.converter import LabelMeToCOCOFiber
    converter = LabelMeToCOCOFiber(n_keypoints=40)
    converter.convert("/data/raw", "/data/coco_fiber/annotations.json")
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from fiberrcnn.geometry import compute_fiber_geometry

# ---------------------------------------------------------------------------
# COCO-Fiber schema helpers
# ---------------------------------------------------------------------------

@dataclass
class COCOFiberDataset:
    info: dict[str, Any] = field(default_factory=dict)
    licenses: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class LabelMeToCOCOFiber:
    """Convert a directory of LabelMe annotations to COCO-Fiber format.

    Parameters
    ----------
    n_keypoints:
        Number of ordered centerline keypoints per fiber (default 40).
    fiber_label:
        LabelMe label string for fiber polygons (default ``"fiber"``).
    """

    CATEGORY: dict[str, Any] = {
        "id": 1,
        "name": "fiber",
        "supercategory": "nanofiber",
        "keypoints": [f"kp{i:02d}" for i in range(40)],
        "skeleton": [[i, i + 1] for i in range(39)],
    }

    def __init__(
        self,
        n_keypoints: int = 40,
        fiber_label: str = "fiber",
    ) -> None:
        self.n_keypoints = n_keypoints
        self.fiber_label = fiber_label

    # ------------------------------------------------------------------

    def convert(
        self,
        labelme_dir: str | Path,
        output_path: str | Path,
    ) -> COCOFiberDataset:
        """Convert all LabelMe JSON files in *labelme_dir*.

        Parameters
        ----------
        labelme_dir:
            Directory containing ``*.json`` LabelMe annotation files and
            their corresponding images.
        output_path:
            Destination path for the COCO-Fiber JSON file.

        Returns
        -------
        dataset : COCOFiberDataset
        """
        labelme_dir = Path(labelme_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_files = sorted(labelme_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {labelme_dir}")

        logger.info(f"Found {len(json_files)} LabelMe files in {labelme_dir}")

        dataset = COCOFiberDataset(
            info={
                "description": "COCO-Fiber nanofiber dataset",
                "version": "2.0",
                "year": 2024,
                "contributor": "FiberRCNN",
                "url": "",
                "date_created": "",
            },
            categories=[self.CATEGORY],
        )

        image_id = 0
        ann_id = 0

        for jf in tqdm(json_files, desc="Converting"):
            try:
                with open(jf, "r", encoding="utf-8") as fh:
                    lm = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Skipping {jf.name}: {exc}")
                continue

            img_h: int = lm.get("imageHeight", 0)
            img_w: int = lm.get("imageWidth", 0)
            img_fname: str = lm.get("imagePath", jf.stem + ".png")

            # Resolve to just the filename (strip any directory prefix)
            img_fname = Path(img_fname).name

            image_id += 1
            dataset.images.append(
                {
                    "id": image_id,
                    "file_name": img_fname,
                    "height": img_h,
                    "width": img_w,
                }
            )

            shapes = lm.get("shapes", [])
            for shape in shapes:
                if shape.get("label", "") != self.fiber_label:
                    continue
                if shape.get("shape_type", "") != "polygon":
                    continue

                points = shape["points"]
                if len(points) < 3:
                    logger.debug(f"Skipping degenerate polygon in {jf.name}")
                    continue

                try:
                    geom = compute_fiber_geometry(points, img_h, img_w, self.n_keypoints)
                except Exception as exc:
                    logger.warning(f"Geometry failed for shape in {jf.name}: {exc}")
                    continue

                ann_id += 1
                # Flatten keypoints to [x0, y0, v0, x1, y1, v1, ...]
                # Normalise (x,y) to [0,1] by image dimensions so that
                # keypoint regression targets are scale-invariant.
                kps = geom.keypoints.copy()  # (40, 3)
                kps[:, 0] /= max(img_w, 1)   # x / W
                kps[:, 1] /= max(img_h, 1)   # y / H
                kps_flat: list[float] = kps.flatten().tolist()

                # Normalise pixel-scale metrics by image diagonal
                img_diag = float((img_h ** 2 + img_w ** 2) ** 0.5) + 1e-6

                annotation: dict[str, Any] = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [round(v, 2) for v in geom.bbox],
                    "area": geom.area,
                    "segmentation": geom.segmentation,
                    "iscrowd": 0,
                    "num_keypoints": self.n_keypoints,
                    "keypoints": [round(v, 6) for v in kps_flat],
                    # All regression targets normalised to ~[0, 1]
                    "fiber_width": round(geom.fiber_width / img_diag, 6),
                    "fiber_length": round(geom.fiber_length / img_diag, 6),
                    "fiber_curvature": round(geom.fiber_curvature, 6),
                    "fiber_orientation": round(geom.fiber_orientation / 180.0, 6),
                    "fiber_tortuosity": round(geom.fiber_tortuosity - 1.0, 6),
                    # Raw values kept for reference (used in morphology post-processing)
                    "fiber_width_px": round(geom.fiber_width, 4),
                    "fiber_length_px": round(geom.fiber_length, 4),
                    "fiber_orientation_deg": round(geom.fiber_orientation, 4),
                    "fiber_tortuosity_raw": round(geom.fiber_tortuosity, 4),
                }
                dataset.annotations.append(annotation)

        logger.info(
            f"Converted {len(dataset.images)} images, "
            f"{len(dataset.annotations)} fiber annotations"
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(dataset.to_dict(), fh, indent=2)

        logger.success(f"Saved COCO-Fiber dataset → {output_path}")
        return dataset

    # ------------------------------------------------------------------

    def split(
        self,
        labelme_dir: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> dict[str, Path]:
        """Convert and split into train / val / test subsets.

        Returns
        -------
        paths : {"train": Path, "val": Path, "test": Path}
        """
        labelme_dir = Path(labelme_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(labelme_dir.glob("*.json"))
        random.seed(seed)
        random.shuffle(json_files)

        n = len(json_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": json_files[:n_train],
            "val": json_files[n_train : n_train + n_val],
            "test": json_files[n_train + n_val :],
        }

        import tempfile, shutil

        out_paths: dict[str, Path] = {}
        for split_name, files in splits.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                for f in files:
                    shutil.copy(f, tmp_dir / f.name)
                dst = output_dir / f"{split_name}.json"
                self.convert(tmp_dir, dst)
                out_paths[split_name] = dst

        return out_paths
