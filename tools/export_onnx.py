#!/usr/bin/env python3
"""
export_onnx.py
==============
Export a trained FiberRCNN backbone+FPN to ONNX format for TensorRT deployment.

Usage:
    python tools/export_onnx.py \\
        --config  configs/fiber_rcnn_r50_fpn.yaml \\
        --weights output/run01/model_final.pth \\
        --output  output/fiberrcnn.onnx \\
        --input_size 800 1333

Notes
-----
* Only the backbone + FPN feature extraction is exported (the full
  Detectron2 Mask R-CNN graph includes custom ops that ONNX does not
  support natively). For full end-to-end deployment with TensorRT,
  use Detectron2's ``d2go`` or ``torch2trt`` pipelines.
* The exported model accepts a single ``float32`` BCHW tensor and
  returns a dict of feature maps ``{p2, p3, p4, p5, p6}``.
* Set ``--dynamic`` to export with dynamic height/width axes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export FiberRCNN backbone+FPN to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("fiberrcnn_backbone.onnx"))
    p.add_argument(
        "--input_size",
        nargs=2,
        type=int,
        default=[800, 1333],
        metavar=("H", "W"),
        help="Input image size for the dummy tensor.",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic spatial axes.",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    return p.parse_args()


class BackboneFPNWrapper(nn.Module):
    """Thin wrapper that exposes backbone+FPN as a plain nn.Module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        # Return feature maps in a fixed order so ONNX can handle them
        return tuple(features[k] for k in sorted(features.keys()))


def main() -> None:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    import fiberrcnn  # register models
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    cfg = get_cfg()
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.WEIGHTS = str(args.weights)
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(args.weights))
    model.eval()

    wrapper = BackboneFPNWrapper(model).to(args.device)
    wrapper.eval()

    H, W = args.input_size
    dummy = torch.zeros(1, 3, H, W, device=args.device)

    dynamic_axes: dict | None = None
    if args.dynamic:
        dynamic_axes = {
            "image": {0: "batch", 2: "height", 3: "width"},
        }
        for i in range(5):  # p2..p6
            dynamic_axes[f"feature_{i}"] = {0: "batch", 2: "h", 3: "w"}

    output_names = [f"feature_{i}" for i in range(5)]

    logger.info(f"Exporting to {args.output} (opset={args.opset}) …")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(args.output),
            input_names=["image"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            export_params=True,
        )

    logger.success(f"ONNX model saved → {args.output}")

    # Verify
    try:
        import onnx

        model_onnx = onnx.load(str(args.output))
        onnx.checker.check_model(model_onnx)
        logger.success("ONNX model passed checker validation.")
    except ImportError:
        logger.warning("onnx not installed — skipping validation.")

    # Optional ORT verification
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(
            str(args.output),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        dummy_np = np.random.randn(1, 3, H, W).astype(np.float32)
        ort_outs = sess.run(None, {input_name: dummy_np})
        logger.success(
            f"ORT inference OK — {len(ort_outs)} feature maps, "
            f"shapes: {[o.shape for o in ort_outs]}"
        )
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ORT verification.")


if __name__ == "__main__":
    main()
