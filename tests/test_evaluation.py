"""
Unit tests for fiberrcnn.evaluation.fiber_evaluator
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fiberrcnn.evaluation.fiber_evaluator import (
    FiberEvaluator,
    compute_oks,
    compute_pck,
    mask_iou,
    mean_absolute_error,
    mean_angular_error,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# Regression metric helpers
# ---------------------------------------------------------------------------

class TestMAE:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(x, x) == pytest.approx(0.0)

    def test_known_error(self):
        pred = np.array([1.0, 2.0, 3.0])
        tgt  = np.array([2.0, 2.0, 2.0])
        assert mean_absolute_error(pred, tgt) == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_symmetry(self):
        a = np.random.rand(20)
        b = np.random.rand(20)
        assert mean_absolute_error(a, b) == pytest.approx(mean_absolute_error(b, a))


class TestRMSE:
    def test_zero_error(self):
        x = np.array([1.0, 2.0, 3.0])
        assert root_mean_squared_error(x, x) == pytest.approx(0.0)

    def test_rmse_ge_mae(self):
        rng = np.random.default_rng(0)
        p = rng.normal(0, 1, 100)
        t = rng.normal(0, 1, 100)
        assert root_mean_squared_error(p, t) >= mean_absolute_error(p, t)


class TestAngularError:
    def test_identical_angles(self):
        angles = np.array([0.0, 45.0, 90.0, 135.0])
        assert mean_angular_error(angles, angles) == pytest.approx(0.0)

    def test_180_and_0_equivalent(self):
        """Orientation is periodic at 180°."""
        pred = np.array([0.0])
        tgt  = np.array([180.0])
        assert mean_angular_error(pred, tgt) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_max_error(self):
        """90° apart = maximum angular error of 90°."""
        pred = np.array([0.0])
        tgt  = np.array([90.0])
        assert mean_angular_error(pred, tgt) == pytest.approx(90.0, abs=1e-6)

    def test_non_negative(self):
        pred = np.random.uniform(0, 180, 50)
        tgt  = np.random.uniform(0, 180, 50)
        assert mean_angular_error(pred, tgt) >= 0.0


# ---------------------------------------------------------------------------
# Keypoint metrics
# ---------------------------------------------------------------------------

class TestOKS:
    def test_perfect_match(self):
        kps = np.zeros((40, 2))
        oks = compute_oks(kps, kps, bbox_area=100.0 * 100.0)
        assert oks == pytest.approx(1.0, abs=1e-3)

    def test_zero_area(self):
        kps = np.zeros((40, 2))
        assert compute_oks(kps, kps, bbox_area=0.0) == 0.0

    def test_large_displacement_low_oks(self):
        pred = np.zeros((40, 2))
        tgt  = np.ones((40, 2)) * 100.0
        oks  = compute_oks(pred, tgt, bbox_area=50.0 * 50.0)
        assert oks < 0.1


class TestPCK:
    def test_perfect_match_all_ones(self):
        kps = np.zeros((40, 2))
        pck = compute_pck(kps, kps, bbox_size=100.0)
        for val in pck.values():
            assert val == pytest.approx(1.0)

    def test_keys_present(self):
        pck = compute_pck(np.zeros((40, 2)), np.ones((40, 2)), bbox_size=10.0)
        assert "PCK@0.05" in pck
        assert "PCK@0.10" in pck
        assert "PCK@0.20" in pck

    def test_larger_threshold_higher_pck(self):
        pred = np.zeros((40, 2))
        tgt  = np.ones((40, 2)) * 5.0
        pck  = compute_pck(pred, tgt, bbox_size=100.0)
        assert pck["PCK@0.20"] >= pck["PCK@0.10"]
        assert pck["PCK@0.10"] >= pck["PCK@0.05"]


# ---------------------------------------------------------------------------
# Mask IoU
# ---------------------------------------------------------------------------

class TestMaskIoU:
    def test_identical_masks(self):
        m = np.zeros((100, 100), dtype=bool)
        m[10:50, 10:50] = True
        assert mask_iou(m, m) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[:50, :] = True
        m2[50:, :] = True
        assert mask_iou(m1, m2) == pytest.approx(0.0, abs=1e-4)

    def test_half_overlap(self):
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[0:50, 0:100] = True
        m2[25:75, 0:100] = True
        iou = mask_iou(m1, m2)
        # intersection = 25×100 = 2500; union = 75×100 = 7500 → IoU ≈ 0.333
        assert 0.3 < iou < 0.4

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        m1 = rng.integers(0, 2, (50, 50)).astype(bool)
        m2 = rng.integers(0, 2, (50, 50)).astype(bool)
        assert mask_iou(m1, m2) == pytest.approx(mask_iou(m2, m1))


# ---------------------------------------------------------------------------
# FiberEvaluator integration
# ---------------------------------------------------------------------------

class TestFiberEvaluator:
    def _make_instances(self, n: int = 4, H: int = 200, W: int = 300):
        """Create dummy Detectron2 Instances for testing."""
        from detectron2.structures import Instances, Boxes

        inst = Instances((H, W))
        boxes = torch.tensor(
            [[10 + i * 20, 10, 40 + i * 20, 40] for i in range(n)],
            dtype=torch.float32,
        )
        inst.pred_boxes = Boxes(boxes)
        inst.gt_boxes = Boxes(boxes)
        inst.scores = torch.ones(n)
        inst.gt_classes = torch.zeros(n, dtype=torch.int64)

        for attr in ("fiber_width", "fiber_length", "fiber_curvature",
                     "fiber_orientation", "fiber_tortuosity"):
            vals = torch.rand(n) * 10 + 1
            setattr(inst, f"pred_{attr}", vals)
            setattr(inst, f"gt_{attr}", vals.clone())   # perfect prediction

        # Keypoints
        kps = torch.rand(n, 40, 2) * 50
        inst.pred_keypoints = kps
        inst.gt_keypoints = type("KPS", (), {"tensor": torch.cat(
            [kps, torch.ones(n, 40, 1) * 2], dim=2
        )})()

        # Masks
        masks = torch.zeros(n, H, W)
        for i in range(n):
            x1, y1, x2, y2 = boxes[i].int().tolist()
            masks[i, y1:y2, x1:x2] = 1.0
        inst.pred_masks = masks
        inst.gt_masks = type("M", (), {"tensor": masks > 0})()

        return inst

    def test_evaluate_perfect_predictions(self):
        try:
            from detectron2.structures import Instances
        except ImportError:
            pytest.skip("detectron2 not available")

        evaluator = FiberEvaluator()
        inst = self._make_instances(n=4)
        evaluator.process([inst], [inst], image_ids=[1])
        results = evaluator.evaluate()

        # Perfect prediction → MAE ≈ 0 for all regression targets
        for key in ("width/MAE", "length/MAE", "curvature/MAE",
                    "tortuosity/MAE"):
            assert results[key] == pytest.approx(0.0, abs=1e-4), f"{key} should be 0"

        assert results["keypoints/OKS"] == pytest.approx(1.0, abs=1e-2)
        assert results["segmentation/mIoU"] == pytest.approx(1.0, abs=1e-3)

    def test_evaluate_empty_predictions(self):
        try:
            from detectron2.structures import Instances
        except ImportError:
            pytest.skip("detectron2 not available")

        evaluator = FiberEvaluator()
        empty = Instances((100, 100))
        evaluator.process([empty], [empty], image_ids=[1])
        results = evaluator.evaluate()
        # No instances → empty results dict
        assert isinstance(results, dict)

    def test_reset_clears_state(self):
        evaluator = FiberEvaluator()
        try:
            inst = self._make_instances(n=2)
            evaluator.process([inst], [inst], image_ids=[1])
        except ImportError:
            pass
        evaluator.reset()
        assert len(evaluator._predictions) == 0
        assert len(evaluator._ground_truths) == 0
