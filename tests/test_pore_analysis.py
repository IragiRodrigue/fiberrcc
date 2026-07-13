"""
Unit tests for fiberrcnn.morphology.pore_analysis
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fiberrcnn.morphology.pore_analysis import (
    PoreAnalyzer,
    PoreDescriptor,
    PoreNetworkResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_fiber_mask():
    """A 200×200 grid of horizontal and vertical fiber strips leaving
    square pores in between."""
    mask = np.zeros((200, 200), dtype=bool)
    # Horizontal fibers every 40 px, 5 px wide
    for y in range(0, 200, 40):
        mask[y : y + 5, :] = True
    # Vertical fibers every 40 px, 5 px wide
    for x in range(0, 200, 40):
        mask[:, x : x + 5] = True
    return mask


@pytest.fixture
def single_pore_mask():
    """A 100×100 image with one large square pore in the centre."""
    mask = np.ones((100, 100), dtype=bool)
    mask[20:80, 20:80] = False  # 60×60 pore
    return mask


# ---------------------------------------------------------------------------
# PoreAnalyzer
# ---------------------------------------------------------------------------

class TestPoreAnalyzerEmpty:
    def test_fully_covered_no_pores(self):
        mask = np.ones((50, 50), dtype=bool)
        analyzer = PoreAnalyzer(mask)
        result = analyzer.analyze()
        assert result.n_pores == 0

    def test_empty_mask_one_pore(self):
        """All background = one giant pore."""
        mask = np.zeros((50, 50), dtype=bool)
        analyzer = PoreAnalyzer(mask)
        result = analyzer.analyze()
        assert result.n_pores == 1
        assert result.mean_area_px > 0


class TestPoreAnalyzerSinglePore:
    def test_detects_single_pore(self, single_pore_mask):
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        assert result.n_pores == 1

    def test_pore_area_approx(self, single_pore_mask):
        """The 60×60 pore has area ≈ 3600 px²."""
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        assert abs(result.mean_area_px - 3600.0) < 100.0

    def test_pore_circularity_less_than_one(self, single_pore_mask):
        """Square pore → circularity < 1 (a circle would give exactly 1)."""
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        # Square: π·(60/2)²/(4·60) perimeter relation ≈ 0.785
        assert 0.5 < result.mean_circularity <= 1.0

    def test_physical_units(self, single_pore_mask):
        analyzer = PoreAnalyzer(single_pore_mask, pixel_size_nm=10.0, min_pore_area_px=50)
        result = analyzer.analyze()
        # 60 px × 10 nm/px = 600 nm equivalent diameter
        assert result.mean_diameter_nm == pytest.approx(
            result.mean_diameter_px * 10.0, rel=0.01
        )


class TestPoreAnalyzerGrid:
    def test_multiple_pores_detected(self, grid_fiber_mask):
        analyzer = PoreAnalyzer(grid_fiber_mask, min_pore_area_px=100)
        result = analyzer.analyze()
        assert result.n_pores >= 4  # grid creates multiple pores

    def test_mean_nn_distance_positive(self, grid_fiber_mask):
        analyzer = PoreAnalyzer(grid_fiber_mask, min_pore_area_px=100)
        result = analyzer.analyze()
        if result.n_pores >= 2:
            assert result.mean_nn_distance_px > 0

    def test_rdf_computed(self, grid_fiber_mask):
        analyzer = PoreAnalyzer(grid_fiber_mask, min_pore_area_px=100)
        result = analyzer.analyze()
        if result.n_pores >= 3:
            assert len(result.rdf_radii) > 0
            assert len(result.rdf_values) == len(result.rdf_radii)


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

class TestDistributionFitting:
    def test_lognormal_fit(self):
        rng = np.random.default_rng(42)
        # Log-normally distributed pore sizes
        samples = rng.lognormal(mean=3.5, sigma=0.4, size=200)
        name, params, p = PoreAnalyzer._fit_distribution(samples)
        assert name in ("lognormal", "gamma")
        assert p > 0

    def test_too_few_samples(self):
        samples = np.array([10.0, 12.0])  # only 2 values
        name, params, p = PoreAnalyzer._fit_distribution(samples)
        assert name == "none"

    def test_constant_array(self):
        samples = np.ones(100) * 5.0
        name, params, p = PoreAnalyzer._fit_distribution(samples)
        assert name == "none"


# ---------------------------------------------------------------------------
# PoreNetworkResult
# ---------------------------------------------------------------------------

class TestPoreNetworkResult:
    def test_summary_prints(self, single_pore_mask):
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        summary = result.summary()
        assert "Pore Analysis" in summary
        assert "N pores" in summary

    def test_to_dict_excludes_nothing_critical(self, single_pore_mask):
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        d = result.to_dict()
        required = ["n_pores", "mean_diameter_px", "mean_circularity",
                    "fit_distribution", "mean_nn_distance_px"]
        for k in required:
            assert k in d

    def test_pore_descriptors_in_dict(self, single_pore_mask):
        analyzer = PoreAnalyzer(single_pore_mask, min_pore_area_px=50)
        result = analyzer.analyze()
        d = result.to_dict()
        assert "pores" in d
        if d["n_pores"] > 0:
            pore = d["pores"][0]
            assert "area_px" in pore
            assert "circularity" in pore
            assert "centroid_x" in pore


# ---------------------------------------------------------------------------
# Nearest-neighbour distances
# ---------------------------------------------------------------------------

class TestNearestNeighbour:
    def test_two_points(self):
        centroids = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = PoreAnalyzer._mean_nn_distance(centroids)
        assert d == pytest.approx(5.0)

    def test_single_point(self):
        centroids = np.array([[10.0, 10.0]])
        assert PoreAnalyzer._mean_nn_distance(centroids) == 0.0

    def test_empty(self):
        assert PoreAnalyzer._mean_nn_distance(np.zeros((0, 2))) == 0.0
