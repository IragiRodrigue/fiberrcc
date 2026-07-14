"""
FiberROIHeads
=============
Extends Detectron2's StandardROIHeads with fiber-specific prediction heads.
All box matching, sampling, and proposal logic is delegated to StandardROIHeads.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from detectron2.config import CfgNode
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import ImageList, Instances
from torch import Tensor

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
from detectron2.modeling.poolers import ROIPooler

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class FiberROIHeads(StandardROIHeads):
    """StandardROIHeads + fiber-specific heads (mask, keypoint, regression).

    Registered as ``FiberROIHeads`` in Detectron2's ROI_HEADS_REGISTRY.
    All proposal matching/sampling is handled by the parent class.
    """

    def __init__(self, cfg: CfgNode, input_shape: dict[str, ShapeSpec]) -> None:
        super().__init__(cfg, input_shape)

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = 14
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_ch = input_shape[in_features[0]].channels

        # Shared feature pooler for fiber heads
        self.fiber_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        # Fiber-specific heads
        self.fiber_mask_head      = FiberMaskHead(input_channels=in_ch)
        self.fiber_keypoint_head  = FiberKeypointHead(
            input_channels=in_ch, num_keypoints=40, roi_size=pooler_resolution
        )
        self.fiber_width_head      = FiberWidthHead(input_channels=in_ch)
        self.fiber_length_head     = FiberLengthHead(input_channels=in_ch)
        self.fiber_curvature_head  = FiberCurvatureHead(input_channels=in_ch)
        self.fiber_orientation_head = FiberOrientationHead(input_channels=in_ch)
        self.fiber_tortuosity_head = FiberTortuosityHead(input_channels=in_ch)
        self.fiber_quality_head    = FiberQualityHead(input_channels=in_ch)

    # ------------------------------------------------------------------
    # Training: called by StandardROIHeads after matching
    # ------------------------------------------------------------------

    def _forward_mask(self, features, instances):
        """Override parent mask head with our FiberMaskHead."""
        if not self.training:
            return self._inference_fiber_heads(features, instances)
        return {}

    def forward(
        self,
        images: ImageList,
        features: dict[str, Tensor],
        proposals: list[Instances],
        targets: list[Instances] | None = None,
    ) -> tuple[list[Instances], dict[str, Tensor]]:
        """Forward pass delegating box/class to parent, fiber heads added on top."""

        # Let StandardROIHeads handle box prediction + matching
        # We temporarily disable its mask/keypoint heads
        del images  # unused

        if self.training:
            assert targets is not None
            proposals = self.label_and_sample_proposals(proposals, targets)

        in_features = self.box_in_features
        feature_list = [features[f] for f in in_features]

        losses: dict[str, Tensor] = {}

        # ── Box head (from parent) ────────────────────────────────────
        # During inference proposals come from RPN and have proposal_boxes.
        # During training they are already sampled and also have proposal_boxes.
        box_features = self.box_pooler(
            feature_list,
            [x.proposal_boxes for x in proposals],
        )
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses.update(self.box_predictor.losses(predictions, proposals))
            # ── Fiber heads ───────────────────────────────────────────
            losses.update(
                self._forward_fiber_heads_train(feature_list, proposals)
            )
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self._forward_fiber_heads_inference(
                feature_list, pred_instances
            )
            return pred_instances, {}

    # ------------------------------------------------------------------
    # Fiber heads — training
    # ------------------------------------------------------------------

    def _forward_fiber_heads_train(
        self,
        feature_list: list[Tensor],
        proposals: list[Instances],
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        # Only use foreground proposals
        fg_proposals = [
            p[p.gt_classes >= 0] for p in proposals
            if hasattr(p, "gt_classes") and len(p) > 0
        ]
        if not fg_proposals or not any(len(p) > 0 for p in fg_proposals):
            return losses

        # Filter to truly foreground (class == 0, not background == num_classes)
        fg_only = []
        for p in proposals:
            if not hasattr(p, "gt_classes") or len(p) == 0:
                continue
            mask = p.gt_classes == 0  # class 0 = fiber
            if mask.any():
                fg_only.append(p[mask])

        if not fg_only:
            return losses

        boxes = [p.proposal_boxes for p in fg_only]
        feats = self.fiber_pooler(feature_list, boxes)

        if feats.shape[0] == 0:
            return losses

        # ── Mask ──────────────────────────────────────────────────────
        if all(hasattr(p, "gt_masks") for p in fg_only):
            gt_masks_list = []
            for p in fg_only:
                crop = p.gt_masks.crop_and_resize(
                    p.proposal_boxes.tensor, 28
                ).float()
                gt_masks_list.append(crop)
            gt_masks = torch.cat(gt_masks_list)
            if gt_masks.shape[0] > 0:
                _, loss = self.fiber_mask_head(feats, gt_masks)
                if loss is not None:
                    losses["loss_fiber_mask"] = loss

        # ── Keypoints (already normalised [0,1] by converter) ────────
        if all(hasattr(p, "gt_keypoints") for p in fg_only):
            gt_kps = torch.cat([p.gt_keypoints.tensor for p in fg_only])
            if gt_kps.shape[0] > 0:
                _, loss = self.fiber_keypoint_head(
                    feats, gt_kps[:, :, :2], weights=gt_kps[:, :, 2]
                )
                if loss is not None:
                    losses["loss_keypoints"] = loss

        # ── Scalar regression ─────────────────────────────────────────
        for attr, head, key in [
            ("gt_fiber_width",      self.fiber_width_head,      "loss_width"),
            ("gt_fiber_length",     self.fiber_length_head,     "loss_length"),
            ("gt_fiber_curvature",  self.fiber_curvature_head,  "loss_curvature"),
            ("gt_fiber_orientation",self.fiber_orientation_head,"loss_orientation"),
            ("gt_fiber_tortuosity", self.fiber_tortuosity_head, "loss_tortuosity"),
        ]:
            if all(hasattr(p, attr) for p in fg_only):
                gt_vals = torch.cat([getattr(p, attr) for p in fg_only])
                if gt_vals.shape[0] > 0:
                    _, loss = head(feats, gt_vals)
                    if loss is not None:
                        losses[key] = loss

        quality_attrs = ("gt_has_bead", "gt_is_blurry", "gt_is_crossing")
        if all(all(hasattr(p, attr) for attr in quality_attrs) for p in fg_only):
            gt_quality = torch.stack(
                [
                    torch.cat([getattr(p, attr) for p in fg_only])
                    for attr in quality_attrs
                ],
                dim=1,
            )
            if gt_quality.shape[0] > 0:
                _, loss = self.fiber_quality_head(feats, gt_quality)
                if loss is not None:
                    losses["loss_quality"] = loss

        return losses

    # ------------------------------------------------------------------
    # Fiber heads — inference
    # ------------------------------------------------------------------

    def _forward_fiber_heads_inference(
        self,
        feature_list: list[Tensor],
        pred_instances: list[Instances],
    ) -> list[Instances]:
        if not any(len(p) > 0 for p in pred_instances):
            return pred_instances

        boxes = [p.pred_boxes for p in pred_instances]
        feats = self.fiber_pooler(feature_list, boxes)

        if feats.shape[0] == 0:
            return pred_instances

        pred_masks,   _ = self.fiber_mask_head(feats)
        pred_kps,     _ = self.fiber_keypoint_head(feats)
        pred_width,   _ = self.fiber_width_head(feats)
        pred_length,  _ = self.fiber_length_head(feats)
        pred_curv,    _ = self.fiber_curvature_head(feats)
        pred_orient,  _ = self.fiber_orientation_head(feats)
        pred_tort,    _ = self.fiber_tortuosity_head(feats)
        pred_quality, _ = self.fiber_quality_head(feats)

        # Detectron2 / COCO evaluators expect keypoints as (x, y, score).
        pred_kps_scores = torch.ones(
            (*pred_kps.shape[:2], 1),
            dtype=pred_kps.dtype,
            device=pred_kps.device,
        )
        pred_kps = torch.cat([pred_kps, pred_kps_scores], dim=2)

        offset = 0
        for inst in pred_instances:
            n = len(inst)
            if n == 0:
                continue
            s = slice(offset, offset + n)
            inst.pred_masks = torch.sigmoid(pred_masks[s])  # keep (N,1,H,W) for Detectron2 postprocess
            inst.pred_keypoints          = pred_kps[s]
            inst.pred_fiber_width        = pred_width[s]
            inst.pred_fiber_length       = pred_length[s]
            inst.pred_fiber_curvature    = pred_curv[s]
            inst.pred_fiber_orientation  = pred_orient[s]
            inst.pred_fiber_tortuosity   = pred_tort[s]
            q = torch.sigmoid(pred_quality[s])
            inst.pred_has_bead    = q[:, 0]
            inst.pred_is_blurry   = q[:, 1]
            inst.pred_is_crossing = q[:, 2]
            offset += n

        return pred_instances
