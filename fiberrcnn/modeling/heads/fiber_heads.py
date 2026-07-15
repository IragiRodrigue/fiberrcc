"""
FiberRCNN Prediction Heads
===========================
Implements lightweight MLP / Conv heads for each prediction target:

1.  BoxHead          → bounding-box regression (delegated to Detectron2)
2.  MaskHead         → instance mask  (BCE + Dice)
3.  KeypointHead     → 40 ordered centerline keypoints
4.  WidthHead        → mean fiber width (SmoothL1)
5.  LengthHead       → fiber arc-length  (SmoothL1)
6.  CurvatureHead    → mean absolute curvature  (SmoothL1)
7.  OrientationHead  → principal orientation in degrees (Circular loss)
8.  TortuosityHead   → arc / end-to-end ratio (SmoothL1)
9.  QualityHead      → has_bead / is_blurry / is_crossing  (BCE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fiberrcnn.modeling.losses.fiber_losses import (
    CircularOrientationLoss,
    KeypointRegressionLoss,
    MaskLoss,
    QualityLoss,
)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _conv_bn_relu(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _fc_bn_relu(in_feat: int, out_feat: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_feat, out_feat, bias=False),
        nn.BatchNorm1d(out_feat),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# 2. Mask Head
# ---------------------------------------------------------------------------

class FiberMaskHead(nn.Module):
    """FCN mask head with BCE + Dice loss.

    Parameters
    ----------
    input_channels : int
        Number of feature channels from the ROI pooler (default 256).
    num_conv : int
        Number of 3×3 convolutions before the final 1×1 predictor.
    output_size : int
        Side length of the predicted mask (default 28 → 28×28).
    """

    def __init__(
        self,
        input_channels: int = 256,
        num_conv: int = 4,
        output_size: int = 28,
    ) -> None:
        super().__init__()
        convs = []
        in_ch = input_channels
        for _ in range(num_conv):
            convs.append(_conv_bn_relu(in_ch, 256))
            in_ch = 256
        self.convs = nn.Sequential(*convs)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.predictor = nn.Conv2d(256, 1, kernel_size=1)
        self.loss_fn = MaskLoss()

    def forward(self, features: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        features : (N, C, H, W)
        targets  : (N, H_out, W_out) float binary masks, or None at inference

        Returns
        -------
        pred_masks : (N, 1, H_out, W_out) logits
        loss       : scalar or None
        """
        x = self.convs(features)
        x = F.relu(self.deconv(x))
        pred = self.predictor(x)  # (N, 1, H, W)

        loss = None
        if targets is not None:
            # Resize targets to match prediction
            tgt = F.interpolate(
                targets.unsqueeze(1).float(),
                size=pred.shape[-2:],
                mode="nearest",
            )
            loss = self.loss_fn(pred.squeeze(1), tgt.squeeze(1))

        return pred, loss


# ---------------------------------------------------------------------------
# 3. Keypoint Head
# ---------------------------------------------------------------------------

class FiberKeypointHead(nn.Module):
    """Predicts 40 ordered centerline keypoints.

    Uses a shared convolutional trunk followed by a per-keypoint MLP head.

    Parameters
    ----------
    input_channels : int
    num_keypoints : int
        Default 40.
    roi_size : int
        Side length of ROI pooled features.
    """

    def __init__(
        self,
        input_channels: int = 256,
        num_keypoints: int = 40,
        roi_size: int = 14,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints

        self.conv_trunk = nn.Sequential(
            _conv_bn_relu(input_channels, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256),
        )
        flat_dim = 256 * roi_size * roi_size
        self.fc_head = nn.Sequential(
            _fc_bn_relu(flat_dim, 1024),
            nn.Dropout(0.3),
            _fc_bn_relu(1024, 512),
        )
        self.predictor = nn.Linear(512, num_keypoints * 2)
        self.loss_fn = KeypointRegressionLoss()

        nn.init.xavier_normal_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        features : (N, C, roi_size, roi_size)
        targets  : (N, K, 2) ground-truth keypoint coords, or None
        weights  : (N, K) visibility weights, or None

        Returns
        -------
        pred_kps : (N, K, 2)
        loss     : scalar or None
        """
        x = self.conv_trunk(features)
        x = x.flatten(1)
        x = self.fc_head(x)
        # Predict ROI-local coordinates in [0, 1] for numerical stability.
        pred = torch.sigmoid(self.predictor(x).view(-1, self.num_keypoints, 2))

        loss = None
        if targets is not None:
            loss = self.loss_fn(pred, targets, weights)

        return pred, loss


# ---------------------------------------------------------------------------
# Shared scalar regression head
# ---------------------------------------------------------------------------

class _ScalarRegressionHead(nn.Module):
    """Generic pooling + MLP head that predicts a single scalar per instance.

    Parameters
    ----------
    input_channels : int
    hidden_dim : int
    activation : optional post-processing of the prediction (e.g. F.softplus)
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            _fc_bn_relu(input_channels, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """(N, C, H, W) → (N,)"""
        x = self.pool(features).flatten(1)
        return self.mlp(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. Width Head
# ---------------------------------------------------------------------------

class FiberWidthHead(nn.Module):
    """Predicts mean fiber width (positive scalar) using SmoothL1 loss."""

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.head = _ScalarRegressionHead(input_channels)

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        pred = F.softplus(self.head(features))  # ensure positive
        loss = None
        if targets is not None:
            loss = F.smooth_l1_loss(pred, targets)
        return pred, loss


# ---------------------------------------------------------------------------
# 5. Length Head
# ---------------------------------------------------------------------------

class FiberLengthHead(nn.Module):
    """Predicts fiber arc-length (positive scalar) using SmoothL1 loss."""

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.head = _ScalarRegressionHead(input_channels)

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        pred = F.softplus(self.head(features))
        loss = None
        if targets is not None:
            loss = F.smooth_l1_loss(pred, targets)
        return pred, loss


# ---------------------------------------------------------------------------
# 6. Curvature Head
# ---------------------------------------------------------------------------

class FiberCurvatureHead(nn.Module):
    """Predicts mean absolute curvature (non-negative) using SmoothL1 loss."""

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.head = _ScalarRegressionHead(input_channels)

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        pred = F.softplus(self.head(features))
        loss = None
        if targets is not None:
            loss = F.smooth_l1_loss(pred, targets)
        return pred, loss


# ---------------------------------------------------------------------------
# 7. Orientation Head
# ---------------------------------------------------------------------------

class FiberOrientationHead(nn.Module):
    """Predicts fiber principal orientation in degrees [0, 180).

    Uses a circular loss (1 − cos(2Δθ)) to respect the π-periodicity of
    undirected orientation.
    """

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.head = _ScalarRegressionHead(input_channels)
        self.loss_fn = CircularOrientationLoss()

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # Map raw prediction to [0, 180) via sigmoid
        pred = torch.sigmoid(self.head(features)) * 180.0
        loss = None
        if targets is not None:
            loss = self.loss_fn(pred, targets)
        return pred, loss


# ---------------------------------------------------------------------------
# 8. Tortuosity Head
# ---------------------------------------------------------------------------

class FiberTortuosityHead(nn.Module):
    """Predicts fiber tortuosity (≥ 1.0) using SmoothL1 loss."""

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.head = _ScalarRegressionHead(input_channels)

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # Tortuosity ≥ 1: use softplus + 1
        pred = F.softplus(self.head(features)) + 1.0
        loss = None
        if targets is not None:
            loss = F.smooth_l1_loss(pred, targets)
        return pred, loss


# ---------------------------------------------------------------------------
# 9. Quality Head
# ---------------------------------------------------------------------------

class FiberQualityHead(nn.Module):
    """Predicts per-instance quality flags: has_bead, is_blurry, is_crossing.

    Uses BCEWithLogitsLoss.
    """

    NUM_FLAGS = 3  # has_bead, is_blurry, is_crossing

    def __init__(self, input_channels: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            _fc_bn_relu(input_channels, 256),
            nn.Linear(256, self.NUM_FLAGS),
        )
        self.loss_fn = QualityLoss()

    def forward(
        self,
        features: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Returns
        -------
        pred  : (N, 3) logits
        loss  : scalar or None
        """
        x = self.pool(features).flatten(1)
        pred = self.mlp(x)
        loss = None
        if targets is not None:
            loss = self.loss_fn(pred, targets)
        return pred, loss
