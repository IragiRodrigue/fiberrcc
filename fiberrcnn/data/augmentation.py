"""
SEM-Specific Augmentation Pipeline
=====================================
Custom Albumentations-based transforms tuned for Scanning Electron Microscopy
images of nanofibers.

SEM images differ from natural photos in several key ways:
  * Greyscale (or pseudo-colour) with high local contrast
  * Gaussian detector noise + shot noise
  * Charging artefacts (bright halos / dark bands)
  * Depth-of-field blur on fibers at different planes
  * No colour information — colour jitter is inappropriate
  * Consistent horizontal/vertical structure (fibers, scale bar)

This module provides:
  * ``build_sem_train_transforms()`` — strong augmentations for training
  * ``build_sem_val_transforms()``   — minimal resize-only for validation
  * Individual transform classes for targeted use

All transforms are compatible with Detectron2's ``T.AugmentationList``.
"""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Low-level SEM noise / distortion primitives (pure numpy, no albumentations dep)
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 8.0,
) -> np.ndarray:
    """Add Gaussian noise to simulate SEM detector noise.

    Parameters
    ----------
    image : (H, W, C) or (H, W) uint8
    mean, std : noise distribution parameters

    Returns
    -------
    noisy image, same dtype
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Add Poisson (shot) noise to simulate low-beam-current SEM images."""
    img_f = image.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_f / scale) * scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def add_charging_artefact(
    image: np.ndarray,
    intensity: float = 40.0,
    n_bands: int = 2,
) -> np.ndarray:
    """Simulate horizontal bright/dark charging artefact bands.

    Parameters
    ----------
    image : (H, W, C) uint8
    intensity : absolute pixel shift (positive = bright, negative = dark)
    n_bands : number of bands to add
    """
    H = image.shape[0]
    out = image.copy().astype(np.float32)
    for _ in range(n_bands):
        y = random.randint(0, H - 1)
        band_h = random.randint(2, max(3, H // 20))
        bright = random.choice([1, -1]) * intensity
        y1 = max(0, y - band_h // 2)
        y2 = min(H, y + band_h // 2)
        out[y1:y2] = np.clip(out[y1:y2] + bright, 0, 255)
    return out.astype(np.uint8)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Simulate scan-line motion blur along a given angle."""
    rad = np.deg2rad(angle_deg)
    kx = int(np.round(np.cos(rad) * kernel_size))
    ky = int(np.round(np.sin(rad) * kernel_size))
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cx, cy = kernel_size // 2, kernel_size // 2
    cv2.line(k, (cx, cy), (cx + kx, cy + ky), 1.0, 1)
    k /= k.sum() + 1e-8
    return cv2.filter2D(image, -1, k)


def apply_depth_blur(
    image: np.ndarray,
    sigma: float = 1.5,
    region_fraction: float = 0.3,
) -> np.ndarray:
    """Blur a random vertical strip to simulate depth-of-field variation."""
    H, W = image.shape[:2]
    out = image.copy()
    strip_w = int(W * region_fraction)
    x0 = random.randint(0, max(0, W - strip_w))
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    out[:, x0 : x0 + strip_w] = blurred[:, x0 : x0 + strip_w]
    return out


def adjust_brightness_contrast(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Linear brightness/contrast: out = clip(alpha * in + beta)."""
    out = image.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_histogram_equalisation(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE (contrast-limited adaptive histogram equalization) per channel."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    if image.ndim == 2:
        return clahe.apply(image)
    # Apply per channel for BGR
    result = image.copy()
    for c in range(image.shape[2]):
        result[:, :, c] = clahe.apply(image[:, :, c])
    return result


# ---------------------------------------------------------------------------
# Detectron2-compatible augmentation wrappers
# ---------------------------------------------------------------------------

try:
    from detectron2.data.transforms import Augmentation, AugInput, Transform, NoOpTransform

    class _NumpyTransform(Transform):
        """Wrap a pure-numpy image function as a Detectron2 Transform."""

        def __init__(self, fn, **kwargs: Any) -> None:
            self._fn = fn
            self._kwargs = kwargs

        def apply_image(self, img: np.ndarray) -> np.ndarray:
            return self._fn(img, **self._kwargs)

        def apply_coords(self, coords: np.ndarray) -> np.ndarray:
            return coords  # pixel coords unchanged

        def apply_segmentation(self, seg: np.ndarray) -> np.ndarray:
            return seg  # masks unchanged

    class RandomGaussianNoise(Augmentation):
        """Random Gaussian noise augmentation for SEM images."""

        def __init__(self, std_range: tuple[float, float] = (3.0, 15.0), p: float = 0.5) -> None:
            self.std_range = std_range
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            std = random.uniform(*self.std_range)
            return _NumpyTransform(add_gaussian_noise, std=std)

    class RandomPoissonNoise(Augmentation):
        """Random Poisson (shot) noise augmentation."""

        def __init__(self, scale_range: tuple[float, float] = (0.5, 2.0), p: float = 0.3) -> None:
            self.scale_range = scale_range
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            scale = random.uniform(*self.scale_range)
            return _NumpyTransform(add_poisson_noise, scale=scale)

    class RandomChargingArtefact(Augmentation):
        """Random SEM charging artefact (bright/dark horizontal bands)."""

        def __init__(self, intensity: float = 30.0, p: float = 0.2) -> None:
            self.intensity = intensity
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            return _NumpyTransform(add_charging_artefact, intensity=self.intensity)

    class RandomDepthBlur(Augmentation):
        """Random depth-of-field blur on a vertical strip."""

        def __init__(
            self,
            sigma_range: tuple[float, float] = (0.8, 2.5),
            region_fraction: float = 0.25,
            p: float = 0.3,
        ) -> None:
            self.sigma_range = sigma_range
            self.region_fraction = region_fraction
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            sigma = random.uniform(*self.sigma_range)
            return _NumpyTransform(
                apply_depth_blur, sigma=sigma, region_fraction=self.region_fraction
            )

    class RandomBrightnessContrast(Augmentation):
        """Random linear brightness and contrast adjustment."""

        def __init__(
            self,
            alpha_range: tuple[float, float] = (0.75, 1.25),
            beta_range: tuple[float, float] = (-20.0, 20.0),
            p: float = 0.5,
        ) -> None:
            self.alpha_range = alpha_range
            self.beta_range = beta_range
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            alpha = random.uniform(*self.alpha_range)
            beta = random.uniform(*self.beta_range)
            return _NumpyTransform(adjust_brightness_contrast, alpha=alpha, beta=beta)

    class RandomCLAHE(Augmentation):
        """Random CLAHE histogram equalization."""

        def __init__(
            self,
            clip_limit_range: tuple[float, float] = (1.0, 4.0),
            p: float = 0.3,
        ) -> None:
            self.clip_limit_range = clip_limit_range
            self.p = p

        def get_transform(self, image: np.ndarray) -> Transform:
            if random.random() > self.p:
                return NoOpTransform()
            clip = random.uniform(*self.clip_limit_range)
            return _NumpyTransform(apply_histogram_equalisation, clip_limit=clip)

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def build_sem_train_transforms(
        min_size: int = 800,
        max_size: int = 1333,
    ) -> list[Augmentation]:
        """Build the full SEM training augmentation pipeline.

        Augmentations applied (in order):
        1. Multi-scale resize
        2. Random horizontal flip
        3. Random vertical flip (fibers have no preferred vertical direction)
        4. Random 90° rotation
        5. Random brightness/contrast
        6. Random CLAHE
        7. Random Gaussian noise
        8. Random Poisson noise
        9. Random depth-of-field blur
        10. Random charging artefact bands

        Returns
        -------
        list of Detectron2 Augmentation objects
        """
        from detectron2.data import transforms as T

        return [
            T.ResizeShortestEdge(
                short_edge_length=[min_size, int(min_size * 1.05),
                                   int(min_size * 1.1), int(min_size * 1.15)],
                max_size=max_size,
                sample_style="choice",
            ),
            T.RandomFlip(horizontal=True),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation(angle=[-90, -45, 0, 45, 90], expand=False),
            RandomBrightnessContrast(p=0.6),
            RandomCLAHE(p=0.3),
            RandomGaussianNoise(p=0.5),
            RandomPoissonNoise(p=0.25),
            RandomDepthBlur(p=0.3),
            RandomChargingArtefact(p=0.15),
        ]

    def build_sem_val_transforms(
        min_size: int = 800,
        max_size: int = 1333,
    ) -> list[Augmentation]:
        """Minimal validation transforms (resize only)."""
        from detectron2.data import transforms as T

        return [
            T.ResizeShortestEdge(
                short_edge_length=[min_size],
                max_size=max_size,
            ),
        ]

except ImportError:
    # detectron2 not available — only raw numpy functions are accessible
    def build_sem_train_transforms(**kwargs):  # type: ignore
        raise ImportError("detectron2 is required for build_sem_train_transforms()")

    def build_sem_val_transforms(**kwargs):  # type: ignore
        raise ImportError("detectron2 is required for build_sem_val_transforms()")
