"""
Unit tests for fiberrcnn.engine.inference (pure Python parts).

The FiberPredictor class itself requires Detectron2 and a trained model,
so we test:
  * FiberInstance dataclass serialisation
  * ImagePrediction serialisation / JSON I/O
  * _instances_to_fiber_list helper (mocked)
  * Post-processing and morphological integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fiberrcnn.engine.inference import (
    FiberInstance,
    ImagePrediction,
)


# ---------------------------------------------------------------------------
# FiberInstance
# ---------------------------------------------------------------------------

class TestFiberInstance:
    def _make(self, idx: int = 0) -> FiberInstance:
        return FiberInstance(
            instance_id=idx,
            bbox=[10.0, 20.0, 50.0, 30.0],
            confidence=0.92,
            fiber_width=5.3,
            fiber_length=112.4,
            fiber_curvature=0.0012,
            fiber_orientation=45.0,
            fiber_tortuosity=1.02,
            has_bead=False,
            is_blurry=False,
            is_crossing=True,
            keypoints=[[float(i), float(i), 2.0] for i in range(40)],
        )

    def test_to_dict_keys(self):
        fi = self._make()
        d = fi.to_dict()
        required = [
            "instance_id", "bbox", "confidence",
            "fiber_width", "fiber_length", "fiber_curvature",
            "fiber_orientation", "fiber_tortuosity",
            "has_bead", "is_blurry", "is_crossing", "keypoints",
        ]
        for k in required:
            assert k in d, f"Missing key: {k}"

    def test_confidence_range(self):
        fi = self._make()
        assert 0.0 <= fi.confidence <= 1.0

    def test_tortuosity_ge_one(self):
        fi = self._make()
        assert fi.fiber_tortuosity >= 1.0

    def test_orientation_range(self):
        fi = self._make()
        assert 0.0 <= fi.fiber_orientation < 180.0

    def test_keypoints_shape(self):
        fi = self._make()
        kps = fi.keypoints
        assert len(kps) == 40


# ---------------------------------------------------------------------------
# ImagePrediction
# ---------------------------------------------------------------------------

class TestImagePrediction:
    def _make(self, n_fibers: int = 5) -> ImagePrediction:
        fibers = [
            FiberInstance(
                instance_id=i,
                bbox=[float(i * 20), 10.0, 15.0, 8.0],
                confidence=0.8 + i * 0.02,
                fiber_width=4.0 + i * 0.5,
                fiber_length=80.0 + i * 5.0,
                fiber_curvature=0.001,
                fiber_orientation=float(i * 18),
                fiber_tortuosity=1.0 + i * 0.01,
            )
            for i in range(n_fibers)
        ]
        return ImagePrediction(
            image_path="/data/sem_001.png",
            image_height=512,
            image_width=512,
            fiber_instances=fibers,
            image_metrics={
                "porosity": 0.72,
                "coverage_ratio": 0.28,
                "fiber_density": 3.2,
                "alignment_score": 0.65,
                "mean_pore_size": 22.0,
            },
        )

    def test_to_dict_structure(self):
        pred = self._make()
        d = pred.to_dict()
        assert "image_path" in d
        assert "image_height" in d
        assert "image_width" in d
        assert "fiber_instances" in d
        assert "image_metrics" in d
        assert len(d["fiber_instances"]) == 5

    def test_save_and_load_json(self):
        pred = self._make()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pred.json"
            pred.save_json(p)
            assert p.exists()

            with open(p) as fh:
                reloaded = json.load(fh)

        assert reloaded["image_path"] == "/data/sem_001.png"
        assert len(reloaded["fiber_instances"]) == 5
        assert reloaded["image_metrics"]["porosity"] == pytest.approx(0.72)

    def test_json_is_valid_json(self):
        pred = self._make()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pred.json"
            pred.save_json(p)
            text = p.read_text(encoding="utf-8")
            parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_empty_instances(self):
        pred = ImagePrediction(
            image_path="test.png",
            image_height=100,
            image_width=100,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.json"
            pred.save_json(p)
            d = json.loads(p.read_text())
        assert d["fiber_instances"] == []
        assert d["image_metrics"] == {}

    def test_fiber_count(self):
        pred = self._make(n_fibers=10)
        assert len(pred.fiber_instances) == 10


# ---------------------------------------------------------------------------
# _instances_to_fiber_list (unit test with mocked Instances)
# ---------------------------------------------------------------------------

class TestInstancesToFiberList:
    def test_basic_conversion(self):
        try:
            from detectron2.structures import Instances, Boxes
        except ImportError:
            pytest.skip("detectron2 not available")

        from fiberrcnn.engine.inference import _instances_to_fiber_list

        H, W = 200, 300
        inst = Instances((H, W))
        boxes = torch.tensor(
            [[10, 20, 80, 60], [100, 50, 200, 120]],
            dtype=torch.float32,
        )
        inst.pred_boxes = Boxes(boxes)
        inst.scores = torch.tensor([0.9, 0.75])

        # Fake masks
        masks = torch.zeros(2, H, W)
        masks[0, 20:60, 10:80] = 1.0
        masks[1, 50:120, 100:200] = 1.0
        inst.pred_masks = masks

        # Fake scalar predictions
        inst.pred_fiber_width = torch.tensor([0.05, 0.06])
        inst.pred_fiber_length = torch.tensor([0.10, 0.20])
        inst.pred_fiber_curvature = torch.tensor([5.0, 6.0])
        inst.pred_fiber_orientation = torch.tensor([0.25, 0.50])
        inst.pred_fiber_tortuosity = torch.tensor([0.05, 0.10])

        # Fake normalized keypoints
        inst.pred_keypoints = torch.tensor(
            [
                [[0.25, 0.50], [0.50, 0.75]],
                [[0.10, 0.20], [0.90, 0.80]],
            ],
            dtype=torch.float32,
        )

        # Fake quality flags
        inst.pred_has_bead    = torch.tensor([0.1, 0.9])
        inst.pred_is_blurry   = torch.tensor([0.2, 0.3])
        inst.pred_is_crossing = torch.tensor([0.8, 0.4])

        fibers, masks_np, centerlines = _instances_to_fiber_list(inst)

        assert len(fibers) == 2
        assert len(masks_np) == 2
        assert len(centerlines) == 2

        # Second fiber has has_bead > 0.5
        assert fibers[1].has_bead is True
        assert fibers[0].has_bead is False

        # Confidence values
        assert fibers[0].confidence == pytest.approx(0.9)
        assert fibers[1].confidence == pytest.approx(0.75)
        assert fibers[0].fiber_orientation == pytest.approx(45.0)
        assert fibers[1].fiber_orientation == pytest.approx(90.0)
        assert fibers[0].fiber_tortuosity == pytest.approx(1.05)
        assert fibers[1].fiber_tortuosity == pytest.approx(1.10)
        assert fibers[0].keypoints[0] == pytest.approx([75.0, 100.0])
        assert fibers[1].keypoints[1] == pytest.approx([270.0, 160.0])
        assert fibers[0].fiber_width > 0.0
        assert fibers[1].fiber_length > fibers[0].fiber_length

    def test_empty_instances(self):
        try:
            from detectron2.structures import Instances, Boxes
        except ImportError:
            pytest.skip("detectron2 not available")

        from fiberrcnn.engine.inference import _instances_to_fiber_list

        inst = Instances((100, 100))
        fibers, masks_np, cls = _instances_to_fiber_list(inst)
        assert fibers == []
        assert masks_np == []
        assert cls == []


# ---------------------------------------------------------------------------
# Morphological integration
# ---------------------------------------------------------------------------

class TestInferenceMorphologyIntegration:
    """Test that morphological metrics are correctly populated in ImagePrediction."""

    def test_morph_metrics_populated(self):
        from fiberrcnn.morphology import compute_image_morphology

        H, W = 200, 300
        mask = np.zeros((H, W), dtype=bool)
        mask[80:90, 20:280] = True
        cl = np.column_stack([np.linspace(20, 280, 40), np.full(40, 85.0)])

        morph = compute_image_morphology(
            masks=[mask],
            centerlines=[cl],
            widths=[10.0],
            lengths=[260.0],
            curvatures=[0.001],
            orientations=[0.0],
            tortuosities=[1.01],
            image_height=H,
            image_width=W,
        )
        result_dict = morph.to_dict()

        assert "porosity" in result_dict
        assert "coverage_ratio" in result_dict
        assert "alignment_score" in result_dict
        assert result_dict["porosity"] + result_dict["coverage_ratio"] == pytest.approx(1.0, abs=1e-5)
        assert result_dict["alignment_score"] > 0.9  # near-horizontal → aligned
