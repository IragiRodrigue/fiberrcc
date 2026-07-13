"""
Unit tests for fiberrcnn.modeling.heads
"""

from __future__ import annotations

import pytest
import torch

from fiberrcnn.modeling.heads.fiber_heads import (
    FiberCurvatureHead,
    FiberKeypointHead,
    FiberLengthHead,
    FiberMaskHead,
    FiberOrientationHead,
    FiberQualityHead,
    FiberTortuosityHead,
    FiberWidthHead,
)

BATCH = 4
IN_CH = 256
ROI = 14


def _feat(n=BATCH, c=IN_CH, h=ROI, w=ROI) -> torch.Tensor:
    return torch.randn(n, c, h, w)


# ---------------------------------------------------------------------------
# FiberMaskHead
# ---------------------------------------------------------------------------

class TestFiberMaskHead:
    def test_inference_shape(self):
        head = FiberMaskHead(input_channels=IN_CH)
        pred, loss = head(_feat(), targets=None)
        assert pred.shape[0] == BATCH
        assert pred.shape[1] == 1
        assert loss is None

    def test_training_loss(self):
        head = FiberMaskHead(input_channels=IN_CH, output_size=14)
        feat = _feat()
        tgt = (torch.rand(BATCH, ROI, ROI) > 0.5).float()
        _, loss = head(feat, targets=tgt)
        assert loss is not None
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# FiberKeypointHead
# ---------------------------------------------------------------------------

class TestFiberKeypointHead:
    def test_inference_shape(self):
        head = FiberKeypointHead(input_channels=IN_CH, num_keypoints=40, roi_size=ROI)
        pred, loss = head(_feat())
        assert pred.shape == (BATCH, 40, 2)
        assert loss is None

    def test_training_loss(self):
        head = FiberKeypointHead(input_channels=IN_CH, num_keypoints=40, roi_size=ROI)
        tgt = torch.rand(BATCH, 40, 2)
        _, loss = head(_feat(), targets=tgt)
        assert loss is not None and loss.item() > 0


# ---------------------------------------------------------------------------
# Scalar regression heads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("HeadClass", [
    FiberWidthHead,
    FiberLengthHead,
    FiberCurvatureHead,
    FiberTortuosityHead,
])
class TestScalarRegressionHeads:
    def test_inference_shape(self, HeadClass):
        head = HeadClass(input_channels=IN_CH)
        pred, loss = head(_feat())
        assert pred.shape == (BATCH,)
        assert loss is None

    def test_positive_output(self, HeadClass):
        """Width, length, curvature, tortuosity must be positive."""
        head = HeadClass(input_channels=IN_CH)
        pred, _ = head(_feat())
        assert (pred > 0).all()

    def test_training_loss(self, HeadClass):
        head = HeadClass(input_channels=IN_CH)
        tgt = torch.abs(torch.randn(BATCH))
        _, loss = head(_feat(), targets=tgt)
        assert loss is not None and loss.item() >= 0


# ---------------------------------------------------------------------------
# FiberOrientationHead
# ---------------------------------------------------------------------------

class TestFiberOrientationHead:
    def test_output_in_range(self):
        head = FiberOrientationHead(input_channels=IN_CH)
        pred, _ = head(_feat())
        assert pred.shape == (BATCH,)
        assert (pred >= 0).all()
        assert (pred <= 180.0).all()

    def test_training_loss(self):
        head = FiberOrientationHead(input_channels=IN_CH)
        tgt = torch.rand(BATCH) * 180.0
        _, loss = head(_feat(), targets=tgt)
        assert loss is not None and loss.item() >= 0


# ---------------------------------------------------------------------------
# FiberQualityHead
# ---------------------------------------------------------------------------

class TestFiberQualityHead:
    def test_output_shape(self):
        head = FiberQualityHead(input_channels=IN_CH)
        pred, loss = head(_feat())
        assert pred.shape == (BATCH, 3)
        assert loss is None

    def test_training_loss(self):
        head = FiberQualityHead(input_channels=IN_CH)
        tgt = (torch.rand(BATCH, 3) > 0.5).float()
        _, loss = head(_feat(), targets=tgt)
        assert loss is not None and loss.item() > 0
