"""
Unit tests for fiberrcnn.data.converter
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fiberrcnn.data.converter import LabelMeToCOCOFiber


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_labelme_json(
    tmp_dir: Path,
    fname: str = "sem_001.json",
    n_fibers: int = 3,
    img_h: int = 256,
    img_w: int = 256,
) -> Path:
    """Write a synthetic LabelMe JSON file with n_fibers horizontal fibers."""
    shapes = []
    for i in range(n_fibers):
        y_center = 50.0 + i * 50.0
        points = [
            [10.0, y_center - 3],
            [240.0, y_center - 3],
            [240.0, y_center + 3],
            [10.0, y_center + 3],
        ]
        shapes.append({
            "label": "fiber",
            "shape_type": "polygon",
            "points": points,
            "flags": {},
        })

    data = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": fname.replace(".json", ".png"),
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    path = tmp_dir / fname
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLabelMeToCOCOFiber:
    def test_basic_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _make_labelme_json(tmp_dir, n_fibers=3)
            converter = LabelMeToCOCOFiber(n_keypoints=40)
            out = tmp_dir / "output.json"
            dataset = converter.convert(tmp_dir, out)

        assert len(dataset.images) == 1
        assert len(dataset.annotations) == 3

    def test_annotation_fields_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _make_labelme_json(tmp_dir, n_fibers=2)
            converter = LabelMeToCOCOFiber(n_keypoints=40)
            out = tmp_dir / "output.json"
            dataset = converter.convert(tmp_dir, out)

        ann = dataset.annotations[0]
        required = [
            "bbox", "segmentation", "keypoints",
            "fiber_width", "fiber_length",
            "fiber_curvature", "fiber_orientation", "fiber_tortuosity",
        ]
        for key in required:
            assert key in ann, f"Missing key: {key}"

    def test_keypoints_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _make_labelme_json(tmp_dir, n_fibers=1)
            converter = LabelMeToCOCOFiber(n_keypoints=40)
            out = tmp_dir / "output.json"
            dataset = converter.convert(tmp_dir, out)

        # 40 keypoints × 3 values (x, y, v) = 120
        assert len(dataset.annotations[0]["keypoints"]) == 120

    def test_positive_geometric_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _make_labelme_json(tmp_dir, n_fibers=2)
            converter = LabelMeToCOCOFiber()
            out = tmp_dir / "output.json"
            dataset = converter.convert(tmp_dir, out)

        for ann in dataset.annotations:
            assert ann["fiber_width"] > 0, "Width must be positive"
            assert ann["fiber_length"] > 0, "Length must be positive"
            assert ann["fiber_tortuosity"] >= 1.0, "Tortuosity must be ≥ 1"
            assert 0.0 <= ann["fiber_orientation"] < 180.0

    def test_json_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            _make_labelme_json(tmp_dir)
            out = tmp_dir / "output.json"
            LabelMeToCOCOFiber().convert(tmp_dir, out)
            assert out.exists()

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                LabelMeToCOCOFiber().convert(Path(tmp) / "empty", Path(tmp) / "out.json")

    def test_split_creates_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Need at least 10 files for a meaningful split
            for i in range(10):
                _make_labelme_json(tmp_dir, fname=f"img_{i:03d}.json", n_fibers=2)
            out_dir = tmp_dir / "split"
            converter = LabelMeToCOCOFiber()
            paths = converter.split(tmp_dir, out_dir, train_ratio=0.7, val_ratio=0.2)

            assert "train" in paths
            assert "val" in paths
            assert "test" in paths
            for name, p in paths.items():
                assert p.exists(), f"{name} file missing"

    def test_non_fiber_labels_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            data = {
                "shapes": [
                    {"label": "noise", "shape_type": "polygon",
                     "points": [[0,0],[10,0],[10,10],[0,10]], "flags": {}},
                    {"label": "fiber", "shape_type": "polygon",
                     "points": [[10,47],[110,47],[110,52],[10,52]], "flags": {}},
                ],
                "imagePath": "test.png",
                "imageHeight": 100,
                "imageWidth": 120,
            }
            p = tmp_dir / "test.json"
            with open(p, "w") as fh:
                json.dump(data, fh)

            out = tmp_dir / "out.json"
            dataset = LabelMeToCOCOFiber().convert(tmp_dir, out)

        assert len(dataset.annotations) == 1  # only the fiber
