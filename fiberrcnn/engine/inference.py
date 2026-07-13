"""
FiberRCNN Inference Engine
===========================
Wraps Detectron2's ``DefaultPredictor`` with:

* Fiber-specific post-processing
* Morphological analysis
* Structured JSON output
* Optional ONNX runtime backend

Typical usage::

    from fiberrcnn.engine.inference import FiberPredictor

    predictor = FiberPredictor.from_config("configs/fiber_rcnn_r50_fpn.yaml",
                                           "output/model_final.pth")
    result = predictor.predict("path/to/sem_image.png")
    print(result.image_metrics)          # porosity, density, …
    print(result.fiber_instances[0])     # per-fiber dict
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.structures import Instances
except ModuleNotFoundError:
    get_cfg = None          # type: ignore
    DefaultPredictor = None  # type: ignore
    Instances = None         # type: ignore

from fiberrcnn.geometry import (
    extract_centerline,
    resample_centerline,
)
from fiberrcnn.morphology import compute_image_morphology, ImageMorphologyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class FiberInstance:
    """Per-fiber prediction result."""

    instance_id: int
    bbox: list[float]
    confidence: float
    fiber_width: float = 0.0
    fiber_length: float = 0.0
    fiber_curvature: float = 0.0
    fiber_orientation: float = 0.0
    fiber_tortuosity: float = 0.0
    has_bead: bool = False
    is_blurry: bool = False
    is_crossing: bool = False
    keypoints: list[list[float]] = field(default_factory=list)
    # Mask is excluded from JSON by default (too large)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImagePrediction:
    """Full prediction result for one image."""

    image_path: str
    image_height: int
    image_width: int
    fiber_instances: list[FiberInstance] = field(default_factory=list)
    image_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "fiber_instances": [f.to_dict() for f in self.fiber_instances],
            "image_metrics": self.image_metrics,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info(f"Saved prediction JSON → {path}")


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _instances_to_fiber_list(
    instances: Instances,
    mask_threshold: float = 0.5,
) -> tuple[list[FiberInstance], list[np.ndarray], list[np.ndarray]]:
    """Convert Detectron2 Instances to FiberInstance list.

    Returns
    -------
    fiber_instances : list of FiberInstance
    masks_np : list of (H, W) bool arrays
    centerlines : list of (N, 2) centerline arrays
    """
    fiber_instances: list[FiberInstance] = []
    masks_np: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []

    n = len(instances)
    if n == 0:
        return fiber_instances, masks_np, centerlines

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy() if hasattr(instances, "scores") else np.ones(n)

    for i in range(n):
        # Mask
        if hasattr(instances, "pred_masks"):
            mask = (instances.pred_masks[i].cpu().numpy() > mask_threshold)
        else:
            mask = np.zeros(
                (int(boxes[i, 3] - boxes[i, 1]), int(boxes[i, 2] - boxes[i, 0])),
                dtype=bool,
            )
        masks_np.append(mask)

        # Centerline from mask
        cl = extract_centerline(mask) if mask.any() else np.zeros((2, 2))
        centerlines.append(cl)

        # Keypoints
        kps_flat: list[list[float]] = []
        if hasattr(instances, "pred_keypoints"):
            kps = instances.pred_keypoints[i].cpu().numpy()  # (K, 2)
            kps_flat = kps.tolist()

        # Quality flags
        has_bead = False
        is_blurry = False
        is_crossing = False
        if hasattr(instances, "pred_has_bead"):
            has_bead = bool(instances.pred_has_bead[i].item() > 0.5)
            is_blurry = bool(instances.pred_is_blurry[i].item() > 0.5)
            is_crossing = bool(instances.pred_is_crossing[i].item() > 0.5)

        def _attr(name: str, default: float = 0.0) -> float:
            if hasattr(instances, name):
                return float(getattr(instances, name)[i].item())
            return default

        box_xyxy = boxes[i].tolist()
        bbox_xywh = [
            box_xyxy[0],
            box_xyxy[1],
            box_xyxy[2] - box_xyxy[0],
            box_xyxy[3] - box_xyxy[1],
        ]

        fi = FiberInstance(
            instance_id=i,
            bbox=bbox_xywh,
            confidence=float(scores[i]),
            fiber_width=_attr("pred_fiber_width"),
            fiber_length=_attr("pred_fiber_length"),
            fiber_curvature=_attr("pred_fiber_curvature"),
            fiber_orientation=_attr("pred_fiber_orientation"),
            fiber_tortuosity=_attr("pred_fiber_tortuosity"),
            has_bead=has_bead,
            is_blurry=is_blurry,
            is_crossing=is_crossing,
            keypoints=kps_flat,
        )
        fiber_instances.append(fi)

    return fiber_instances, masks_np, centerlines


# ---------------------------------------------------------------------------
# FiberPredictor
# ---------------------------------------------------------------------------

class FiberPredictor:
    """High-level inference wrapper for FiberRCNN.

    Parameters
    ----------
    cfg_path : str | Path
        YAML config file.
    weights_path : str | Path
        Model checkpoint path.
    score_thresh : float
        Override detection threshold.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        cfg_path: str | Path,
        weights_path: str | Path,
        score_thresh: float = 0.5,
        device: str = "cuda",
    ) -> None:
        from detectron2.config import get_cfg

        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = device if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        self._predictor = DefaultPredictor(cfg)
        logger.info(f"FiberPredictor ready — device: {cfg.MODEL.DEVICE}")

    @classmethod
    def from_config(
        cls,
        cfg_path: str | Path,
        weights_path: str | Path,
        score_thresh: float = 0.5,
        device: str = "cuda",
    ) -> "FiberPredictor":
        return cls(cfg_path, weights_path, score_thresh, device)

    def predict(
        self,
        image_input: str | Path | np.ndarray,
        run_morphology: bool = True,
    ) -> ImagePrediction:
        """Run inference on a single image.

        Parameters
        ----------
        image_input : file path or (H, W, 3) BGR numpy array
        run_morphology : compute image-level morphological metrics

        Returns
        -------
        ImagePrediction
        """
        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
        else:
            bgr = image_input
            img_path = "<array>"

        H, W = bgr.shape[:2]

        # Run model
        with torch.no_grad():
            output = self._predictor(bgr)
        instances: Instances = output["instances"].to("cpu")

        fiber_instances, masks_np, centerlines = _instances_to_fiber_list(instances)

        image_metrics: dict[str, Any] = {}
        if run_morphology and len(masks_np) > 0:
            morph = compute_image_morphology(
                masks=[m for m in masks_np],
                centerlines=centerlines,
                widths=[f.fiber_width for f in fiber_instances],
                lengths=[f.fiber_length for f in fiber_instances],
                curvatures=[f.fiber_curvature for f in fiber_instances],
                orientations=[f.fiber_orientation for f in fiber_instances],
                tortuosities=[f.fiber_tortuosity for f in fiber_instances],
                image_height=H,
                image_width=W,
            )
            image_metrics = morph.to_dict()
            # Flatten pore_stats into top-level
            pore = image_metrics.pop("pore_stats", {})
            image_metrics.update(pore)

        return ImagePrediction(
            image_path=img_path,
            image_height=H,
            image_width=W,
            fiber_instances=fiber_instances,
            image_metrics=image_metrics,
        )

    def predict_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path | None = None,
        run_morphology: bool = True,
        save_json: bool = True,
    ) -> list[ImagePrediction]:
        """Run inference on a list of images.

        Parameters
        ----------
        image_paths : list of file paths
        output_dir : directory for JSON outputs (optional)
        run_morphology : compute image-level metrics
        save_json : write per-image JSON files
        """
        from tqdm import tqdm

        results: list[ImagePrediction] = []
        for p in tqdm(image_paths, desc="Inference"):
            pred = self.predict(p, run_morphology=run_morphology)
            results.append(pred)
            if save_json and output_dir is not None:
                stem = Path(str(p)).stem
                pred.save_json(Path(output_dir) / f"{stem}_prediction.json")
        return results
