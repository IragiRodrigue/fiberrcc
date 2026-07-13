"""
Backbone Registration
=====================
Registers ConvNeXt-Tiny, Swin-T, and Swin-S backbones (via torchvision) with
Detectron2's BACKBONE_REGISTRY so they can be used via the config file.

ResNet50-FPN and ResNet101-FPN are already built into Detectron2 and do not
require registration here.

Usage in config:
    MODEL:
      BACKBONE:
        NAME: build_convnext_tiny_fpn_backbone   # or build_swin_t_fpn_backbone etc.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, FPN, Backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConvNeXt wrapper
# ---------------------------------------------------------------------------

class ConvNeXtWrapper(Backbone):
    """Wraps ``torchvision.models.convnext_tiny`` as a Detectron2 backbone.

    Exposes four FPN-compatible feature stages: ``res2``, ``res3``, ``res4``,
    ``res5`` (following Detectron2 naming convention).
    """

    _STAGE_CHANNELS = {
        "convnext_tiny": [96, 192, 384, 768],
        "convnext_small": [96, 192, 384, 768],
    }

    _STAGE_STRIDES = [4, 8, 16, 32]

    def __init__(self, variant: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        if variant == "convnext_tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_tiny(weights=weights)
        elif variant == "convnext_small":
            weights = tvm.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = tvm.convnext_small(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

        features = model.features
        self.stage0 = features[0]          # stem
        self.stage1 = nn.Sequential(features[1], features[2])
        self.stage2 = nn.Sequential(features[3], features[4])
        self.stage3 = nn.Sequential(features[5], features[6])
        self.stage4 = features[7]

        channels = self._STAGE_CHANNELS[variant]
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {
            k: v for k, v in zip(self._out_features, channels)
        }
        self._out_feature_strides = {
            k: v for k, v in zip(self._out_features, self._STAGE_STRIDES)
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stage0(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return {"res2": c2, "res3": c3, "res4": c4, "res5": c5}

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


# ---------------------------------------------------------------------------
# Swin Transformer wrapper
# ---------------------------------------------------------------------------

class SwinWrapper(Backbone):
    """Wraps ``torchvision.models.swin_t / swin_s`` as a Detectron2 backbone."""

    _CONFIGS = {
        "swin_t": {
            "channels": [96, 192, 384, 768],
            "builder": "swin_t",
            "weights_cls": "Swin_T_Weights",
        },
        "swin_s": {
            "channels": [96, 192, 384, 768],
            "builder": "swin_s",
            "weights_cls": "Swin_S_Weights",
        },
    }

    def __init__(self, variant: str = "swin_t", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        cfg = self._CONFIGS[variant]
        weights_cls = getattr(tvm, cfg["weights_cls"])
        weights = weights_cls.DEFAULT if pretrained else None
        model = getattr(tvm, cfg["builder"])(weights=weights)

        # Swin torchvision has .features with alternating patch merging + blocks
        self.stem = model.features[0]       # PatchMerging stem
        self.stage1 = model.features[1]     # SwinTransformerBlock × 2
        self.down1 = model.features[2]      # PatchMerging
        self.stage2 = model.features[3]
        self.down2 = model.features[4]
        self.stage3 = model.features[5]
        self.down3 = model.features[6]
        self.stage4 = model.features[7]

        channels = cfg["channels"]
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = dict(zip(self._out_features, channels))
        self._out_feature_strides = {
            "res2": 4, "res3": 8, "res4": 16, "res5": 32
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x).permute(0, 3, 1, 2)
        x = self.down1(self.stage1(x))
        c3 = self.stage2(x).permute(0, 3, 1, 2)
        x = self.down2(self.stage2(x))
        c4 = self.stage3(x).permute(0, 3, 1, 2)
        x = self.down3(self.stage3(x))
        c5 = self.stage4(x).permute(0, 3, 1, 2)
        return {"res2": c2, "res3": c3, "res4": c4, "res5": c5}

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


# ---------------------------------------------------------------------------
# FPN builder helpers (registered with Detectron2)
# ---------------------------------------------------------------------------

def _build_fpn(cfg, backbone: Backbone) -> FPN:
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    return FPN(
        bottom_up=backbone,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )


@BACKBONE_REGISTRY.register()
def build_convnext_tiny_fpn_backbone(cfg, input_shape: ShapeSpec) -> FPN:
    return _build_fpn(cfg, ConvNeXtWrapper("convnext_tiny", pretrained=True))


@BACKBONE_REGISTRY.register()
def build_convnext_small_fpn_backbone(cfg, input_shape: ShapeSpec) -> FPN:
    return _build_fpn(cfg, ConvNeXtWrapper("convnext_small", pretrained=True))


@BACKBONE_REGISTRY.register()
def build_swin_t_fpn_backbone(cfg, input_shape: ShapeSpec) -> FPN:
    return _build_fpn(cfg, SwinWrapper("swin_t", pretrained=True))


@BACKBONE_REGISTRY.register()
def build_swin_s_fpn_backbone(cfg, input_shape: ShapeSpec) -> FPN:
    return _build_fpn(cfg, SwinWrapper("swin_s", pretrained=True))
