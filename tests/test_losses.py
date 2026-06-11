"""
Unit tests for fiberrcnn.modeling.losses
"""

from __future__ import annotations

import math

import pytest
import torch

from fiberrcnn.modeling.losses.fiber_losses import (
    BinaryDiceLoss,
    CircularOrientationLoss,
    KeypointRegressionLoss,
    MaskLoss,
    QualityLoss,
)


# ---------------------------------------------------------------------------
# BinaryDiceLoss
# ---------------------------------------------------------------------------

class TestBinaryDiceLoss:
    def test_perfect_prediction_near_zero(self):
        loss_fn = BinaryDiceLoss()
        pred = torch.ones(4, 28, 28) * 10.0  # high logits → sigmoid ≈ 1
        tgt = torch.ones(4, 28, 28)
        loss = loss_fn(pred, tgt)
        assert loss.item() < 0.02

    def test_wrong_prediction_near_one(self):
        loss_fn = BinaryDiceLoss()
        pred = torch.ones(4, 28, 28) * 10.0
        tgt = torch.zeros(4, 28, 28)
        loss = loss_fn(pred, tgt)
        assert loss.item() > 0.9

    def test_scalar_output(self):
        loss_fn = BinaryDiceLoss()
        pred = torch.randn(2, 14, 14)
        tgt = (torch.randn(2, 14, 14) > 0).float()
        loss = loss_fn(pred, tgt)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# MaskLoss
# ---------------------------------------------------------------------------

class TestMaskLoss:
    def test_output_scalar(self):
        fn = MaskLoss()
        pred = torch.randn(2, 28, 28)
        tgt = (torch.randn(2, 28, 28) > 0).float()
        loss = fn(pred, tgt)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_perfect_prediction_lower_than_random(self):
        fn = MaskLoss()
        pred_good = torch.ones(2, 28, 28) * 8.0
        pred_bad = torch.randn(2, 28, 28)
        tgt = torch.ones(2, 28, 28)
        assert fn(pred_good, tgt).item() < fn(pred_bad, tgt).item()


# ---------------------------------------------------------------------------
# CircularOrientationLoss
# ---------------------------------------------------------------------------

class TestCircularOrientationLoss:
    def test_zero_error_for_identical(self):
        fn = CircularOrientationLoss()
        pred = torch.tensor([45.0, 90.0, 135.0])
        tgt = pred.clone()
        loss = fn(pred, tgt)
        assert loss.item() < 1e-6

    def test_180_and_0_are_equivalent(self):
        """Angles 0° and 180° should have near-zero loss (same orientation)."""
        fn = CircularOrientationLoss()
        pred = torch.tensor([0.0])
        tgt = torch.tensor([180.0])
        loss = fn(pred, tgt)
        # cos(2 * (0 - π)) = cos(-2π) = 1 → loss = 1 - 1 = 0
        assert loss.item() < 1e-5

    def test_orthogonal_orientations_max_loss(self):
        """45° vs 135° should give maximum circular loss."""
        fn = CircularOrientationLoss()
        pred = torch.tensor([45.0])
        tgt = torch.tensor([135.0])
        loss = fn(pred, tgt)
        # cos(2*(45-135)°) = cos(-180°) = -1 → loss = 2
        assert loss.item() > 1.9


# ---------------------------------------------------------------------------
# KeypointRegressionLoss
# ---------------------------------------------------------------------------

class TestKeypointRegressionLoss:
    def test_zero_error(self):
        fn = KeypointRegressionLoss()
        pred = torch.rand(4, 40, 2)
        loss = fn(pred, pred.clone())
        assert loss.item() < 1e-6

    def test_with_visibility_weights(self):
        fn = KeypointRegressionLoss()
        pred = torch.rand(2, 40, 2)
        tgt = torch.zeros(2, 40, 2)
        weights = torch.zeros(2, 40)  # all invisible
        loss_zero_weight = fn(pred, tgt, weights)
        loss_no_weight = fn(pred, tgt, None)
        # Zero weights should suppress the loss
        assert loss_zero_weight.item() < loss_no_weight.item()


# ---------------------------------------------------------------------------
# QualityLoss
# ---------------------------------------------------------------------------

class TestQualityLoss:
    def test_output_scalar(self):
        fn = QualityLoss()
        pred = torch.randn(8, 3)
        tgt = (torch.rand(8, 3) > 0.5).float()
        loss = fn(pred, tgt)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        fn = QualityLoss()
        # all positives
        pred = torch.ones(4, 3) * 10.0
        tgt = torch.ones(4, 3)
        loss = fn(pred, tgt)
        assert loss.item() < 0.01
