"""
Miscellaneous utilities for FiberRCNN.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_deterministic(seed: int = 42) -> None:
    """Set all random seeds for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    output_dir: str | Path,
    filename: str = "checkpoint.pth",
) -> None:
    """Save a training checkpoint; copy to ``model_best.pth`` if *is_best*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / filename
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, out / "model_best.pth")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> int:
    """Load checkpoint into model (and optionally optimizer).

    Returns
    -------
    start_epoch : int
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


# ---------------------------------------------------------------------------
# Config serialisation
# ---------------------------------------------------------------------------

def save_config_json(cfg: Any, path: str | Path) -> None:
    """Serialise a Detectron2 CfgNode to a JSON file."""
    from detectron2.config import CfgNode

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cfg, CfgNode):
        data = cfg
    else:
        data = cfg
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(data), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------

def md5(path: str | Path, chunk: int = 1 << 20) -> str:
    """Compute MD5 of a file (useful for dataset integrity checks)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while buf := fh.read(chunk):
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Scale / unit conversion
# ---------------------------------------------------------------------------

def pixels_to_nm(pixels: float, nm_per_pixel: float) -> float:
    """Convert pixel measurements to nanometres."""
    return pixels * nm_per_pixel


def nm_to_pixels(nm: float, nm_per_pixel: float) -> float:
    return nm / nm_per_pixel


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load image as BGR uint8 array; raises if not found."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def pad_to_stride(image: np.ndarray, stride: int = 32) -> np.ndarray:
    """Pad (H, W, C) image so that H and W are multiples of *stride*."""
    H, W = image.shape[:2]
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")


# ---------------------------------------------------------------------------
# AMP context
# ---------------------------------------------------------------------------

def amp_autocast(enabled: bool = True):
    """Return a ``torch.cuda.amp.autocast`` context if CUDA is available."""
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast()
    import contextlib
    return contextlib.nullcontext()
