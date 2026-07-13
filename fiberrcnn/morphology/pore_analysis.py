"""
Advanced Pore Analysis
=======================
Dedicated module for detailed pore characterisation in nanofiber mats.

Goes beyond simple distance-transform statistics to provide:

* Individual pore segmentation (connected components in background)
* Per-pore shape descriptors (area, perimeter, circularity, elongation)
* Pore size distribution fitting (log-normal, gamma)
* Pore network topology (connectivity, tortuosity)
* Radial distribution function (RDF) of pore centres
* Export to structured dict / DataFrame

Example
-------
>>> from fiberrcnn.morphology.pore_analysis import PoreAnalyzer
>>> combined_mask = ...  # (H, W) bool, True where fibers are
>>> analyzer = PoreAnalyzer(combined_mask, pixel_size_nm=10.5)
>>> result = analyzer.analyze()
>>> print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label as ndlabel,
    maximum_filter,
    binary_fill_holes,
)
from scipy.stats import lognorm, gamma as gamma_dist
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PoreDescriptor:
    """Shape and size descriptors for a single pore."""

    pore_id: int
    area_px: float           # pixel area
    area_nm2: float          # physical area (nm²), 0 if pixel_size unknown
    perimeter_px: float
    circularity: float       # 4π·area / perimeter²  (1.0 = circle)
    elongation: float        # major_axis / minor_axis (1.0 = circle)
    equivalent_diameter_px: float  # diameter of equal-area circle
    equivalent_diameter_nm: float
    centroid_x: float        # column
    centroid_y: float        # row
    inscribed_radius_px: float     # from distance transform


@dataclass
class PoreNetworkResult:
    """Full pore analysis result for one image."""

    n_pores: int = 0

    # Size statistics (pixels)
    mean_area_px: float = 0.0
    std_area_px: float = 0.0
    mean_diameter_px: float = 0.0
    std_diameter_px: float = 0.0
    median_diameter_px: float = 0.0
    max_diameter_px: float = 0.0

    # Physical units (nm) — populated if pixel_size_nm is given
    mean_diameter_nm: float = 0.0
    std_diameter_nm: float = 0.0

    # Shape
    mean_circularity: float = 0.0
    mean_elongation: float = 0.0

    # Distribution fit
    fit_distribution: str = "none"     # "lognormal" | "gamma"
    fit_params: dict[str, float] = field(default_factory=dict)
    fit_ks_pvalue: float = 0.0

    # Topology
    mean_nn_distance_px: float = 0.0   # mean nearest-neighbour distance
    rdf_radii: list[float] = field(default_factory=list)
    rdf_values: list[float] = field(default_factory=list)

    # Per-pore descriptors (not serialised to summary)
    pores: list[PoreDescriptor] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Pore Analysis Summary ===",
            f"  N pores          : {self.n_pores}",
            f"  Mean diameter    : {self.mean_diameter_px:.1f} px"
            + (f"  ({self.mean_diameter_nm:.1f} nm)" if self.mean_diameter_nm > 0 else ""),
            f"  Std diameter     : {self.std_diameter_px:.1f} px",
            f"  Median diameter  : {self.median_diameter_px:.1f} px",
            f"  Max diameter     : {self.max_diameter_px:.1f} px",
            f"  Mean circularity : {self.mean_circularity:.3f}",
            f"  Mean elongation  : {self.mean_elongation:.2f}",
            f"  Mean NN dist     : {self.mean_nn_distance_px:.1f} px",
            f"  Fit distribution : {self.fit_distribution} (p={self.fit_ks_pvalue:.3f})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "pores"}
        d["pores"] = [p.__dict__ for p in self.pores]
        return d


# ---------------------------------------------------------------------------
# PoreAnalyzer
# ---------------------------------------------------------------------------

class PoreAnalyzer:
    """Analyse the pore structure of a nanofiber mat.

    Parameters
    ----------
    combined_mask : (H, W) bool — True where fibers are present
    pixel_size_nm : nm per pixel for physical unit conversion (0 = skip)
    min_pore_area_px : minimum pore area in pixels (noise filter)
    """

    def __init__(
        self,
        combined_mask: np.ndarray,
        pixel_size_nm: float = 0.0,
        min_pore_area_px: int = 10,
    ) -> None:
        self.mask = combined_mask.astype(bool)
        self.pixel_size_nm = pixel_size_nm
        self.min_pore_area_px = min_pore_area_px
        self._H, self._W = combined_mask.shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> PoreNetworkResult:
        """Run the complete pore analysis pipeline."""
        background = ~self.mask

        if not background.any():
            return PoreNetworkResult()

        # Distance transform for inscribed radii
        edt = distance_transform_edt(background)

        # Segment individual pores
        labeled, n_labels = ndlabel(background)
        if n_labels == 0:
            return PoreNetworkResult()

        props = regionprops(labeled)
        pores: list[PoreDescriptor] = []

        for prop in props:
            if prop.area < self.min_pore_area_px:
                continue

            # Inscribed radius: max EDT in this pore
            pore_mask_local = labeled == prop.label
            edt_vals = edt[pore_mask_local]
            inscribed_r = float(edt_vals.max()) if len(edt_vals) > 0 else 0.0

            # Shape descriptors
            perim = prop.perimeter + 1e-6
            circ = 4.0 * math.pi * prop.area / (perim ** 2)
            axes = prop.axis_major_length, prop.axis_minor_length
            elong = axes[0] / max(axes[1], 1e-3)

            eq_diam_px = prop.equivalent_diameter_axis_length if hasattr(
                prop, "equivalent_diameter_axis_length"
            ) else math.sqrt(4.0 * prop.area / math.pi)

            eq_diam_nm = eq_diam_px * self.pixel_size_nm if self.pixel_size_nm > 0 else 0.0
            area_nm2 = prop.area * (self.pixel_size_nm ** 2) if self.pixel_size_nm > 0 else 0.0

            cy, cx = prop.centroid
            pores.append(PoreDescriptor(
                pore_id=prop.label,
                area_px=float(prop.area),
                area_nm2=area_nm2,
                perimeter_px=float(prop.perimeter),
                circularity=float(circ),
                elongation=float(elong),
                equivalent_diameter_px=float(eq_diam_px),
                equivalent_diameter_nm=float(eq_diam_nm),
                centroid_x=float(cx),
                centroid_y=float(cy),
                inscribed_radius_px=float(inscribed_r),
            ))

        if not pores:
            return PoreNetworkResult()

        diameters = np.array([p.equivalent_diameter_px for p in pores])
        areas = np.array([p.area_px for p in pores])
        circs = np.array([p.circularity for p in pores])
        elongs = np.array([p.elongation for p in pores])

        result = PoreNetworkResult(
            n_pores=len(pores),
            mean_area_px=float(areas.mean()),
            std_area_px=float(areas.std()),
            mean_diameter_px=float(diameters.mean()),
            std_diameter_px=float(diameters.std()),
            median_diameter_px=float(np.median(diameters)),
            max_diameter_px=float(diameters.max()),
            mean_diameter_nm=float(diameters.mean() * self.pixel_size_nm) if self.pixel_size_nm > 0 else 0.0,
            std_diameter_nm=float(diameters.std() * self.pixel_size_nm) if self.pixel_size_nm > 0 else 0.0,
            mean_circularity=float(circs.mean()),
            mean_elongation=float(elongs.mean()),
            pores=pores,
        )

        # Distribution fitting
        result.fit_distribution, result.fit_params, result.fit_ks_pvalue = (
            self._fit_distribution(diameters)
        )

        # Nearest-neighbour distances
        centroids = np.array([[p.centroid_x, p.centroid_y] for p in pores])
        result.mean_nn_distance_px = float(self._mean_nn_distance(centroids))

        # Radial distribution function
        result.rdf_radii, result.rdf_values = self._compute_rdf(centroids)

        return result

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_distribution(
        values: np.ndarray,
    ) -> tuple[str, dict[str, float], float]:
        """Fit log-normal and gamma distributions; return best fit."""
        from scipy.stats import kstest

        if len(values) < 5 or values.std() < 1e-6:
            return "none", {}, 0.0

        best_name = "none"
        best_params: dict[str, float] = {}
        best_p = 0.0

        # Log-normal
        try:
            s, loc, scale = lognorm.fit(values, floc=0)
            stat, p = kstest(values, "lognorm", args=(s, loc, scale))
            if p > best_p:
                best_p = float(p)
                best_name = "lognormal"
                best_params = {"s": float(s), "loc": float(loc), "scale": float(scale)}
        except Exception:
            pass

        # Gamma
        try:
            a, loc, scale = gamma_dist.fit(values, floc=0)
            stat, p = kstest(values, "gamma", args=(a, loc, scale))
            if p > best_p:
                best_p = float(p)
                best_name = "gamma"
                best_params = {"a": float(a), "loc": float(loc), "scale": float(scale)}
        except Exception:
            pass

        return best_name, best_params, float(best_p)

    @staticmethod
    def _mean_nn_distance(centroids: np.ndarray) -> float:
        """Mean distance from each pore to its nearest neighbour."""
        if len(centroids) < 2:
            return 0.0
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)  # k=2: self + 1 neighbour
        return float(dists[:, 1].mean())

    def _compute_rdf(
        self,
        centroids: np.ndarray,
        n_bins: int = 50,
        r_max: float | None = None,
    ) -> tuple[list[float], list[float]]:
        """Compute the radial distribution function g(r).

        Returns
        -------
        (radii, g_values) — lists of floats
        """
        if len(centroids) < 3:
            return [], []

        if r_max is None:
            r_max = min(self._H, self._W) / 2.0

        from scipy.spatial.distance import pdist
        dists = pdist(centroids)
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        counts, _ = np.histogram(dists, bins=bin_edges)

        # Normalise by ideal gas expectation: ρ · 2πr·Δr
        area = self._H * self._W
        rho = len(centroids) / area
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        dr = bin_edges[1] - bin_edges[0]
        expected = rho * 2.0 * math.pi * bin_centres * dr * len(centroids)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(expected > 0, counts / expected, 0.0)

        return bin_centres.tolist(), g.tolist()
