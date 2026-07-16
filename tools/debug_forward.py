#!/usr/bin/env python3
"""
debug_forward.py
================
Run one forward pass and print compact diagnostics for masks and keypoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug one FiberRCNN forward pass")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np

    import fiberrcnn  # noqa: F401
    from fiberrcnn.engine.inference import FiberPredictor

    predictor = FiberPredictor.from_config(
        cfg_path=args.config,
        weights_path=args.weights,
        score_thresh=args.threshold,
        device=args.device,
    )
    pred = predictor.predict(args.input, run_morphology=False)

    print(f"image={args.input.name}")
    print(f"instances={len(pred.fiber_instances)} masks={len(pred.masks)} centerlines={len(pred.centerlines)}")

    if pred.masks:
        occ = [float(np.asarray(m, dtype=np.float32).mean()) for m in pred.masks]
        print(
            "mask_occ",
            {
                "min": round(min(occ), 6),
                "median": round(sorted(occ)[len(occ) // 2], 6),
                "max": round(max(occ), 6),
            },
        )

    for idx, fi in enumerate(pred.fiber_instances[: args.topk]):
        mask_occ = float(np.asarray(pred.masks[idx], dtype=np.float32).mean()) if idx < len(pred.masks) else 0.0
        kps = np.asarray(fi.keypoints, dtype=np.float32) if fi.keypoints else np.zeros((0, 2), dtype=np.float32)
        if kps.size:
            kp_span_x = float(kps[:, 0].max() - kps[:, 0].min())
            kp_span_y = float(kps[:, 1].max() - kps[:, 1].min())
        else:
            kp_span_x = 0.0
            kp_span_y = 0.0
        print(
            {
                "idx": idx,
                "score": round(fi.confidence, 4),
                "bbox_xywh": [round(float(v), 1) for v in fi.bbox],
                "mask_occ": round(mask_occ, 6),
                "kp_span_x": round(kp_span_x, 3),
                "kp_span_y": round(kp_span_y, 3),
            }
        )


if __name__ == "__main__":
    main()
