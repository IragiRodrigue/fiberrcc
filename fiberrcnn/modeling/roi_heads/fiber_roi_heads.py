"""
FiberROIHeads
=============
Custom Detectron2 ROI head that wires together all fiber-specific prediction
heads (box, mask, keypoint, width, length, curvature, orientation, tortuosity,
quality) and returns unified losses during training / predictions at inference.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from detectron2.config import CfgNode, configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, ImageList, Instances
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

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class FiberROIHeads(nn.Module):
    """Multi-head ROI module for nanofiber instance analysis.

    Registered under the key ``"FiberROIHeads"`` in Detectron2's
    ``ROI_HEADS_REGISTRY`` so it can be selected via:

    .. code-block:: yaml

       MODEL:
         ROI_HEADS:
           NAME: FiberROIHeads

    Parameters
    ----------
    cfg : CfgNode
    input_shape : dict[str, ShapeSpec]
        Feature-map shape specs from the backbone / FPN.
    """

    def __init__(self, cfg: CfgNode, input_shape: dict[str, ShapeSpec]) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes: int = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.batch_size_per_image: int = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_fraction: float = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.score_thresh: float = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.nms_thresh: float = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Pooler parameters -----
        in_features: list[str] = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution: int = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION  # 7
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio: int = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type: str = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head_in_channels = input_shape[in_features[0]].channels
        box_head_shape = ShapeSpec(channels=box_head_in_channels, height=pooler_resolution, width=pooler_resolution)

        self.box_head = build_box_head(cfg, box_head_shape)
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

        # ---- Mask pooler ----
        mask_resolution: int = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION  # 14
        self.mask_pooler = ROIPooler(
            output_size=mask_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = FiberMaskHead(
            input_channels=box_head_in_channels,
            output_size=mask_resolution * 2,
        )

        # ---- Keypoint / regression pooler (shared) ----
        feat_resolution: int = getattr(cfg.MODEL, "FIBER_ROI_POOLER_RESOLUTION", 14)
        self.feat_pooler = ROIPooler(
            output_size=feat_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        in_ch = box_head_in_channels

        self.keypoint_head = FiberKeypointHead(
            input_channels=in_ch, num_keypoints=40, roi_size=feat_resolution
        )
        self.width_head = FiberWidthHead(input_channels=in_ch)
        self.length_head = FiberLengthHead(input_channels=in_ch)
        self.curvature_head = FiberCurvatureHead(input_channels=in_ch)
        self.orientation_head = FiberOrientationHead(input_channels=in_ch)
        self.tortuosity_head = FiberTortuosityHead(input_channels=in_ch)
        self.quality_head = FiberQualityHead(input_channels=in_ch)

        # ---- Proposal sampler ----
        from detectron2.modeling.proposal_generator.proposal_utils import (
            add_ground_truth_to_proposals,
        )
        from detectron2.modeling.sampling import subsample_labels

        self._add_gt_to_proposals = add_ground_truth_to_proposals
        self._subsample_labels = subsample_labels

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Proposal matching
    # ------------------------------------------------------------------

    def _match_and_sample(
        self,
        proposals: list[Instances],
        targets: list[Instances],
    ) -> tuple[list[Instances], list[int]]:
        """Match proposals to GT and subsample for training."""
        from detectron2.modeling.matcher import Matcher
        from detectron2.modeling.box_regression import Box2BoxTransform

        if not hasattr(self, "_matcher"):
            self._matcher = Matcher(
                thresholds=self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                labels=self.cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            )

        proposals = self._add_gt_to_proposals(proposals, targets)
        proposals_with_gt: list[Instances] = []
        num_fg_samples: list[int] = []

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            gt_boxes = targets_per_image.gt_boxes if has_gt else Boxes(
                torch.zeros((0, 4), device=proposals_per_image.proposal_boxes.device)
            )

            match_quality_matrix = targets_per_image.gt_boxes.tensor.new_zeros(
                (len(gt_boxes), len(proposals_per_image))
            )
            if has_gt:
                from torchvision.ops import box_iou
                match_quality_matrix = box_iou(
                    gt_boxes.tensor, proposals_per_image.proposal_boxes.tensor
                )

            matched_idxs, proposal_labels = self._matcher(match_quality_matrix)

            sampled_idxs, gt_classes = self._subsample_labels(
                proposal_labels,
                self.batch_size_per_image,
                self.positive_fraction,
                self.num_classes,
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                matched_gt = targets_per_image[matched_idxs[sampled_idxs]]
                for field_name in targets_per_image.get_fields():
                    if field_name.startswith("gt_"):
                        setattr(
                            proposals_per_image,
                            field_name,
                            getattr(matched_gt, field_name),
                        )

            num_fg_samples.append((gt_classes >= 0).sum().item())
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt, num_fg_samples

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: ImageList,
        features: dict[str, Tensor],
        proposals: list[Instances],
        targets: list[Instances] | None = None,
    ) -> tuple[list[Instances], dict[str, Tensor]]:
        """
        Returns
        -------
        (instances, losses)
            During training: empty list and dict of scalar losses.
            During inference: list of predicted Instances and empty dict.
        """
        in_features = self.cfg.MODEL.ROI_HEADS.IN_FEATURES
        feature_list = [features[f] for f in in_features]

        if self.training:
            assert targets is not None
            proposals, _ = self._match_and_sample(proposals, targets)

        boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in proposals]

        # ---- Box head ----
        box_features = self.box_pooler(feature_list, boxes)
        box_features_fc = self.box_head(box_features)
        predictions = self.box_predictor(box_features_fc)

        # ---- Shared pooled features (mask / keypoint / regression) ----
        feat_features = self.feat_pooler(feature_list, boxes)
        mask_features = self.mask_pooler(feature_list, boxes)

        if self.training:
            losses = self._training_losses(
                proposals, predictions, feat_features, mask_features
            )
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self._forward_inference(pred_instances, feature_list)
            return pred_instances, {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _training_losses(
        self,
        proposals: list[Instances],
        predictions: tuple[Tensor, Tensor],
        feat_features: Tensor,
        mask_features: Tensor,
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        # Box classification + regression
        box_losses = self.box_predictor.losses(predictions, proposals)
        losses.update(box_losses)

        # Gather gt across all proposals in batch
        gt_classes = torch.cat([p.gt_classes for p in proposals])
        fg_mask = (gt_classes >= 0) & (gt_classes < self.num_classes)

        if fg_mask.any():
            fg_feat = feat_features[fg_mask]
            fg_mask_feat = mask_features[fg_mask]

            # Mask
            gt_masks = torch.cat([p.gt_masks.tensor for p in proposals])[fg_mask]
            _, mask_loss = self.mask_head(fg_mask_feat, gt_masks.float())
            losses["loss_mask"] = mask_loss

            # Keypoints
            if hasattr(proposals[0], "gt_keypoints"):
                gt_kps = torch.cat(
                    [p.gt_keypoints.tensor for p in proposals]
                )[fg_mask]  # (N, K, 3)
                _, kp_loss = self.keypoint_head(
                    fg_feat,
                    gt_kps[:, :, :2],
                    weights=gt_kps[:, :, 2],
                )
                losses["loss_keypoint"] = kp_loss

            # Scalar regression heads
            for attr, head, key in [
                ("gt_fiber_width", self.width_head, "loss_width"),
                ("gt_fiber_length", self.length_head, "loss_length"),
                ("gt_fiber_curvature", self.curvature_head, "loss_curvature"),
                ("gt_fiber_orientation", self.orientation_head, "loss_orientation"),
                ("gt_fiber_tortuosity", self.tortuosity_head, "loss_tortuosity"),
            ]:
                if hasattr(proposals[0], attr):
                    gt_vals = torch.cat([getattr(p, attr) for p in proposals])[fg_mask]
                    _, reg_loss = head(fg_feat, gt_vals)
                    losses[key] = reg_loss

            # Quality head (no GT available by default — skip gracefully)
            if hasattr(proposals[0], "gt_quality"):
                gt_quality = torch.cat([p.gt_quality for p in proposals])[fg_mask]
                _, q_loss = self.quality_head(fg_feat, gt_quality)
                losses["loss_quality"] = q_loss

        return losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _forward_inference(
        self,
        pred_instances: list[Instances],
        feature_list: list[Tensor],
    ) -> list[Instances]:
        """Enrich predicted instances with fiber-specific attributes."""
        if not any(len(p) > 0 for p in pred_instances):
            return pred_instances

        boxes = [p.pred_boxes for p in pred_instances]
        feat_features = self.feat_pooler(feature_list, boxes)
        mask_features = self.mask_pooler(feature_list, boxes)

        # Masks
        pred_masks, _ = self.mask_head(mask_features)
        pred_masks = torch.sigmoid(pred_masks).squeeze(1)  # (N, H, W)

        # Keypoints
        pred_kps, _ = self.keypoint_head(feat_features)  # (N, 40, 2)

        # Scalar heads
        pred_width, _ = self.width_head(feat_features)
        pred_length, _ = self.length_head(feat_features)
        pred_curv, _ = self.curvature_head(feat_features)
        pred_orient, _ = self.orientation_head(feat_features)
        pred_tort, _ = self.tortuosity_head(feat_features)
        pred_quality, _ = self.quality_head(feat_features)

        # Distribute back across per-image instances
        offset = 0
        for instances in pred_instances:
            n = len(instances)
            if n == 0:
                continue
            slc = slice(offset, offset + n)
            instances.pred_masks = pred_masks[slc]
            instances.pred_keypoints = pred_kps[slc]
            instances.pred_fiber_width = pred_width[slc]
            instances.pred_fiber_length = pred_length[slc]
            instances.pred_fiber_curvature = pred_curv[slc]
            instances.pred_fiber_orientation = pred_orient[slc]
            instances.pred_fiber_tortuosity = pred_tort[slc]

            quality_sigmoid = torch.sigmoid(pred_quality[slc])
            instances.pred_has_bead = quality_sigmoid[:, 0]
            instances.pred_is_blurry = quality_sigmoid[:, 1]
            instances.pred_is_crossing = quality_sigmoid[:, 2]

            offset += n

        return pred_instances
