"""
Morphological Analysis Module
==============================
Computes image-level structural metrics from predicted masks and centerlines.

IMPORTANT: These metrics are NOT learned by neural network heads.
They are computed deterministically during post-processing.

All functions accept plain numpy arrays and return plain Python
scalars / dicts so they can be called without a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndlabel, maximum_filter
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PoreSizeStats:
    mean_pore_size: float = 0.0
    median_pore_size: float = 0.0
    max_pore_size: float = 0.0
    pore_size_std: float = 0.0
    pore_count: int = 0


@dataclass
class ImageMorphologyResult:
    """All image-level morphological metrics."""

    # Coverage
    porosity: float = 0.0
    coverage_ratio: float = 0.0
    fiber_density: float = 0.0       # fibers per 100×100 px area

    # Width
    mean_fiber_width: float = 0.0
    width_distribution: list[float] = field(default_factory=list)

    # Length
    mean_fiber_length: float = 0.0
    length_distribution: list[float] = field(default_factory=list)

    # Curvature
    mean_curvature: float = 0.0
    curvature_distribution: list[float] = field(default_factory=list)

    # Tortuosity
    mean_tortuosity: float = 0.0
    tortuosity_distribution: list[float] = field(default_factory=list)

    # Orientation
    mean_orientation: float = 0.0
    orientation_distribution: list[float] = field(default_factory=list)  # degrees
    alignment_score: float = 0.0       # 0 = random, 1 = perfectly aligned

    # Crossings
    intersection_count: int = 0
    junction_density: float = 0.0     # junctions per 100×100 px area

    # Pores
    pore_stats: PoreSizeStats = field(default_factory=PoreSizeStats)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not isinstance(v, list)}
        d["pore_stats"] = self.pore_stats.__dict__
        return d


# ---------------------------------------------------------------------------
# 1. Porosity & Coverage
# ---------------------------------------------------------------------------

def compute_porosity_coverage(combined_mask: np.ndarray) -> tuple[float, float]:
    """Compute porosity and coverage from a combined fiber mask.

    Parameters
    ----------
    combined_mask : (H, W) bool — True where any fiber is present

    Returns
    -------
    (porosity, coverage_ratio)
    """
    total = combined_mask.size
    fiber_px = int(combined_mask.sum())
    bg_px = total - fiber_px
    porosity = bg_px / total
    coverage_ratio = fiber_px / total
    return float(porosity), float(coverage_ratio)


# ---------------------------------------------------------------------------
# 2. Fiber Density
# ---------------------------------------------------------------------------

def compute_fiber_density(n_fibers: int, image_height: int, image_width: int) -> float:
    """Fibers per 10 000 px² (i.e. per 100×100 px tile)."""
    area = image_height * image_width
    return n_fibers / area * 10_000.0


# ---------------------------------------------------------------------------
# 3. Alignment Score
# ---------------------------------------------------------------------------

def compute_alignment_score(orientations_deg: list[float]) -> float:
    """Compute a scalar alignment score in [0, 1].

    Uses circular statistics on doubled angles so that 0° ≡ 180°.

    Parameters
    ----------
    orientations_deg : list of angles in degrees [0, 180)

    Returns
    -------
    score : 0 = random, 1 = perfectly aligned
    """
    if len(orientations_deg) < 2:
        return 1.0 if len(orientations_deg) == 1 else 0.0

    angles = np.deg2rad(np.asarray(orientations_deg) * 2.0)  # double angles
    mean_cos = float(np.cos(angles).mean())
    mean_sin = float(np.sin(angles).mean())
    r = math.sqrt(mean_cos ** 2 + mean_sin ** 2)  # resultant length
    return float(r)


# ---------------------------------------------------------------------------
# 4. Fiber Intersection Detection
# ---------------------------------------------------------------------------

def count_intersections(
    centerlines: list[np.ndarray],
    tolerance: float = 3.0,
) -> int:
    """Count approximate crossings between fiber centerlines.

    Uses a nearest-neighbour approach: for each pair of centerlines, detect
    if any point from one is within *tolerance* pixels of any point from the
    other at a location that is not near the endpoints (which are natural
    near-misses).

    Parameters
    ----------
    centerlines : list of (N_i, 2) arrays in (x, y) convention
    tolerance : distance threshold in pixels

    Returns
    -------
    intersection_count : int
    """
    n = len(centerlines)
    if n < 2:
        return 0

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci = centerlines[i]
            cj = centerlines[j]
            if len(ci) < 3 or len(cj) < 3:
                continue
            # Exclude endpoints (first / last 10% of each centerline)
            trim_i = max(1, len(ci) // 10)
            trim_j = max(1, len(cj) // 10)
            ci_mid = ci[trim_i:-trim_i]
            cj_mid = cj[trim_j:-trim_j]

            # Vectorised distance check
            diff = ci_mid[:, None, :] - cj_mid[None, :, :]  # (Ni, Nj, 2)
            dists = np.sqrt((diff ** 2).sum(axis=2))         # (Ni, Nj)
            if (dists < tolerance).any():
                count += 1
    return count


def compute_junction_density(
    intersection_count: int, image_height: int, image_width: int
) -> float:
    """Junctions per 10 000 px²."""
    area = image_height * image_width
    return intersection_count / area * 10_000.0


# ---------------------------------------------------------------------------
# 5. Pore Size Distribution
# ---------------------------------------------------------------------------

def compute_pore_size_distribution(
    combined_mask: np.ndarray,
    min_pore_radius: float = 1.0,
) -> PoreSizeStats:
    """Estimate pore size distribution using the distance transform.

    Background pixels are transformed by EDT; local maxima in the EDT of the
    background give the radius of the largest inscribed circle in each pore.

    Parameters
    ----------
    combined_mask : (H, W) bool — True where fibers are present
    min_pore_radius : minimum EDT value to consider a peak

    Returns
    -------
    PoreSizeStats
    """
    background = ~combined_mask
    if not background.any():
        return PoreSizeStats()

    edt = distance_transform_edt(background)

    # Local maxima in background EDT
    neighborhood = np.ones((5, 5), dtype=bool)
    local_max = (edt == maximum_filter(edt, footprint=neighborhood)) & background
    peak_values = edt[local_max & (edt >= min_pore_radius)]

    if len(peak_values) == 0:
        return PoreSizeStats()

    # Pore diameter = 2 × inscribed radius
    diameters = peak_values * 2.0

    return PoreSizeStats(
        mean_pore_size=float(diameters.mean()),
        median_pore_size=float(np.median(diameters)),
        max_pore_size=float(diameters.max()),
        pore_size_std=float(diameters.std()),
        pore_count=int(len(diameters)),
    )


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

def compute_image_morphology(
    masks: list[np.ndarray],
    centerlines: list[np.ndarray],
    widths: list[float],
    lengths: list[float],
    curvatures: list[float],
    orientations: list[float],
    tortuosities: list[float],
    image_height: int,
    image_width: int,
) -> ImageMorphologyResult:
    """Compute all image-level morphological metrics.

    Parameters
    ----------
    masks : list of (H, W) bool masks, one per fiber instance
    centerlines : list of (N_i, 2) centerline arrays in (x, y)
    widths, lengths, curvatures, orientations, tortuosities :
        Per-fiber scalar values (from geometry or neural network predictions)
    image_height, image_width : image dimensions

    Returns
    -------
    ImageMorphologyResult
    """
    result = ImageMorphologyResult()
    n = len(masks)

    if n == 0:
        return result

    # Combined mask
    combined = np.zeros((image_height, image_width), dtype=bool)
    for m in masks:
        combined |= m

    result.porosity, result.coverage_ratio = compute_porosity_coverage(combined)
    result.fiber_density = compute_fiber_density(n, image_height, image_width)

    # Width
    result.width_distribution = [float(w) for w in widths]
    result.mean_fiber_width = float(np.mean(widths)) if widths else 0.0

    # Length
    result.length_distribution = [float(l) for l in lengths]
    result.mean_fiber_length = float(np.mean(lengths)) if lengths else 0.0

    # Curvature
    result.curvature_distribution = [float(c) for c in curvatures]
    result.mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0

    # Tortuosity
    result.tortuosity_distribution = [float(t) for t in tortuosities]
    result.mean_tortuosity = float(np.mean(tortuosities)) if tortuosities else 0.0

    # Orientation
    result.orientation_distribution = [float(o) for o in orientations]
    result.mean_orientation = float(np.mean(orientations)) if orientations else 0.0
    result.alignment_score = compute_alignment_score(orientations)

    # Intersections
    result.intersection_count = count_intersections(centerlines)
    result.junction_density = compute_junction_density(
        result.intersection_count, image_height, image_width
    )

    # Pore size
    result.pore_stats = compute_pore_size_distribution(combined)

    return result
