"""
FiberRCNN Inference Engine
==========================
Wraps Detectron2's ``DefaultPredictor`` with:

* Fiber-specific post-processing
* Morphological analysis
* Structured JSON output

Typical usage::

    from fiberrcnn.engine.inference import FiberPredictor

    predictor = FiberPredictor.from_config(
        "configs/fiber_rcnn_r50_fpn.yaml",
        "output/model_final.pth",
    )
    result = predictor.predict("path/to/sem_image.png")
    print(result.image_metrics)
    print(result.fiber_instances[0])
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances

from fiberrcnn.geometry import extract_centerline
from fiberrcnn.morphology import compute_image_morphology

logger = logging.getLogger(__name__)


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
            "fiber_instances": [fiber.to_dict() for fiber in self.fiber_instances],
            "image_metrics": self.image_metrics,
        }

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        logger.info("Saved prediction JSON to %s", output_path)


def _instances_to_fiber_list(
    instances: Instances,
    mask_threshold: float = 0.5,
) -> tuple[list[FiberInstance], list[np.ndarray], list[np.ndarray]]:
    """Convert Detectron2 ``Instances`` to serializable FiberRCNN outputs."""

    fiber_instances: list[FiberInstance] = []
    masks_np: list[np.ndarray] = []
    centerlines: list[np.ndarray] = []

    n_instances = len(instances)
    if n_instances == 0:
        return fiber_instances, masks_np, centerlines

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy() if hasattr(instances, "scores") else np.ones(n_instances)

    for index in range(n_instances):
        if hasattr(instances, "pred_masks"):
            mask = instances.pred_masks[index].cpu().numpy() > mask_threshold
        else:
            mask = np.zeros(
                (
                    int(boxes[index, 3] - boxes[index, 1]),
                    int(boxes[index, 2] - boxes[index, 0]),
                ),
                dtype=bool,
            )
        masks_np.append(mask)

        centerline = extract_centerline(mask) if mask.any() else np.zeros((2, 2))
        centerlines.append(centerline)

        keypoints: list[list[float]] = []
        if hasattr(instances, "pred_keypoints"):
            keypoints = instances.pred_keypoints[index].cpu().numpy().tolist()

        has_bead = False
        is_blurry = False
        is_crossing = False
        if hasattr(instances, "pred_has_bead"):
            has_bead = bool(instances.pred_has_bead[index].item() > 0.5)
            is_blurry = bool(instances.pred_is_blurry[index].item() > 0.5)
            is_crossing = bool(instances.pred_is_crossing[index].item() > 0.5)

        def _get_scalar_attr(name: str, default: float = 0.0) -> float:
            if hasattr(instances, name):
                return float(getattr(instances, name)[index].item())
            return default

        box_xyxy = boxes[index].tolist()
        bbox_xywh = [
            box_xyxy[0],
            box_xyxy[1],
            box_xyxy[2] - box_xyxy[0],
            box_xyxy[3] - box_xyxy[1],
        ]

        fiber_instances.append(
            FiberInstance(
                instance_id=index,
                bbox=bbox_xywh,
                confidence=float(scores[index]),
                fiber_width=_get_scalar_attr("pred_fiber_width"),
                fiber_length=_get_scalar_attr("pred_fiber_length"),
                fiber_curvature=_get_scalar_attr("pred_fiber_curvature"),
                fiber_orientation=_get_scalar_attr("pred_fiber_orientation"),
                fiber_tortuosity=_get_scalar_attr("pred_fiber_tortuosity"),
                has_bead=has_bead,
                is_blurry=is_blurry,
                is_crossing=is_crossing,
                keypoints=keypoints,
            )
        )

    return fiber_instances, masks_np, centerlines


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
        cfg = get_cfg()
        cfg.merge_from_file(str(cfg_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = device if torch.cuda.is_available() else "cpu"
        cfg.freeze()

        self._predictor = DefaultPredictor(cfg)
        logger.info("FiberPredictor ready on device %s", cfg.MODEL.DEVICE)

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
        image_input : str | Path | np.ndarray
            File path or ``(H, W, 3)`` BGR NumPy array.
        run_morphology : bool
            Whether to compute image-level morphological metrics.

        Returns
        -------
        ImagePrediction
        """

        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            bgr = cv2.imread(image_path)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
        else:
            bgr = image_input
            image_path = "<array>"

        height, width = bgr.shape[:2]

        with torch.no_grad():
            output = self._predictor(bgr)
        instances: Instances = output["instances"].to("cpu")

        fiber_instances, masks_np, centerlines = _instances_to_fiber_list(instances)

        image_metrics: dict[str, Any] = {}
        if run_morphology and masks_np:
            morphology = compute_image_morphology(
                masks=masks_np,
                centerlines=centerlines,
                widths=[fiber.fiber_width for fiber in fiber_instances],
                lengths=[fiber.fiber_length for fiber in fiber_instances],
                curvatures=[fiber.fiber_curvature for fiber in fiber_instances],
                orientations=[fiber.fiber_orientation for fiber in fiber_instances],
                tortuosities=[fiber.fiber_tortuosity for fiber in fiber_instances],
                image_height=height,
                image_width=width,
            )
            image_metrics = morphology.to_dict()
            pore_stats = image_metrics.pop("pore_stats", {})
            image_metrics.update(pore_stats)

        return ImagePrediction(
            image_path=image_path,
            image_height=height,
            image_width=width,
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
        image_paths : list[str | Path]
            Input image paths.
        output_dir : str | Path | None
            Directory where JSON outputs are written.
        run_morphology : bool
            Whether to compute image-level morphological metrics.
        save_json : bool
            Whether to write one JSON file per image.
        """

        from tqdm import tqdm

        results: list[ImagePrediction] = []
        for image_path in tqdm(image_paths, desc="Inference"):
            prediction = self.predict(image_path, run_morphology=run_morphology)
            results.append(prediction)
            if save_json and output_dir is not None:
                stem = Path(str(image_path)).stem
                prediction.save_json(Path(output_dir) / f"{stem}_prediction.json")
        return results
