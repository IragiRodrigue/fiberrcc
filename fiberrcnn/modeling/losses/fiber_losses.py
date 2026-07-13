"""
FiberRCNN Custom Loss Functions
================================
All losses follow the Detectron2 convention of returning a scalar tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Dice Loss (binary)
# ---------------------------------------------------------------------------

class BinaryDiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation masks.

    Parameters
    ----------
    smooth : float
        Laplace smoothing to avoid zero-division.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : (N,) or (N, H, W) — logits or probabilities
        target : same shape as *pred*, float in [0, 1]
        """
        pred = torch.sigmoid(pred)
        pred_flat = pred.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1).float()

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# BCE + Dice combined mask loss
# ---------------------------------------------------------------------------

class MaskLoss(nn.Module):
    """Combined BCE + Dice loss for mask prediction.

    Parameters
    ----------
    bce_weight : float
        Weight for the BCE term (0–1).
    dice_weight : float
        Weight for the Dice term (0–1).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = BinaryDiceLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : (N, H, W) logits
        target : (N, H, W) float in [0, 1]
        """
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="mean")
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Circular Orientation Loss
# ---------------------------------------------------------------------------

class CircularOrientationLoss(nn.Module):
    """Loss for orientation prediction in degrees [0, 180).

    Uses a circular distance so that 0° and 179° are treated as close.
    Maps to [0, π) radians, computes 1 − cos(2Δθ) which is periodic
    with period π.

    Parameters
    ----------
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : (N,) predicted orientation in degrees
        target : (N,) ground-truth orientation in degrees
        """
        # Convert to radians
        deg_to_rad = math.pi / 180.0
        pred_rad = pred * deg_to_rad
        target_rad = target * deg_to_rad
        # Circular distance for [0, π) periodicity
        loss = 1.0 - torch.cos(2.0 * (pred_rad - target_rad))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Keypoint loss (SmoothL1 + L1 combined)
# ---------------------------------------------------------------------------

class KeypointRegressionLoss(nn.Module):
    """Per-keypoint regression loss combining SmoothL1 and L1.

    Parameters
    ----------
    smooth_l1_beta : float
        β for SmoothL1 (huber loss threshold).
    l1_weight : float
        Weight for the L1 term.
    smooth_l1_weight : float
        Weight for the SmoothL1 term.
    """

    def __init__(
        self,
        smooth_l1_beta: float = 1.0,
        l1_weight: float = 0.5,
        smooth_l1_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.beta = smooth_l1_beta
        self.l1_w = l1_weight
        self.sl1_w = smooth_l1_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        pred : (N, K, 2) predicted keypoint coordinates
        target : (N, K, 2) ground-truth keypoint coordinates
        weights : (N, K) optional per-keypoint visibility weights
        """
        smooth_l1 = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        l1 = F.l1_loss(pred, target, reduction="none")  # (N, K, 2)

        combined = self.sl1_w * smooth_l1 + self.l1_w * l1  # (N, K, 2)

        if weights is not None:
            combined = combined * weights.unsqueeze(-1)

        return combined.mean()


# ---------------------------------------------------------------------------
# Quality head loss
# ---------------------------------------------------------------------------

class QualityLoss(nn.Module):
    """BCEWithLogitsLoss for multi-label fiber quality flags.

    Predicts: has_bead, is_blurry, is_crossing.
    """

    def __init__(self, pos_weight: Tensor | None = None) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="mean"
        )

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : (N, 3) logits
        target : (N, 3) binary labels
        """
        return self.criterion(pred, target.float())
