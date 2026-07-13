"""
Unit tests for fiberrcnn.morphology
"""

from __future__ import annotations

import numpy as np
import pytest

from fiberrcnn.morphology.fiber_morphology import (
    PoreSizeStats,
    compute_alignment_score,
    compute_fiber_density,
    compute_image_morphology,
    compute_junction_density,
    compute_pore_size_distribution,
    compute_porosity_coverage,
    count_intersections,
)


# ---------------------------------------------------------------------------
# compute_porosity_coverage
# ---------------------------------------------------------------------------

class TestPorosityCoverage:
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        p, c = compute_porosity_coverage(mask)
        assert p == pytest.approx(1.0)
        assert c == pytest.approx(0.0)

    def test_full_mask(self):
        mask = np.ones((100, 100), dtype=bool)
        p, c = compute_porosity_coverage(mask)
        assert p == pytest.approx(0.0)
        assert c == pytest.approx(1.0)

    def test_half_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :] = True
        p, c = compute_porosity_coverage(mask)
        assert p == pytest.approx(0.5, abs=1e-3)
        assert c == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# compute_fiber_density
# ---------------------------------------------------------------------------

class TestFiberDensity:
    def test_zero_fibers(self):
        assert compute_fiber_density(0, 100, 100) == 0.0

    def test_known_density(self):
        # 10 fibers in 100×100 = 10 / 10000 * 10000 = 10 per 100×100 tile
        d = compute_fiber_density(10, 100, 100)
        assert d == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_alignment_score
# ---------------------------------------------------------------------------

class TestAlignmentScore:
    def test_perfectly_aligned(self):
        # All pointing same direction
        score = compute_alignment_score([45.0] * 20)
        assert score > 0.99

    def test_random_orientations(self):
        rng = np.random.default_rng(0)
        angles = rng.uniform(0, 180, 100).tolist()
        score = compute_alignment_score(angles)
        # Random should be low
        assert score < 0.3

    def test_single_fiber(self):
        score = compute_alignment_score([30.0])
        assert score == pytest.approx(1.0)

    def test_empty_list(self):
        score = compute_alignment_score([])
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# count_intersections
# ---------------------------------------------------------------------------

class TestCountIntersections:
    def test_no_crossing_parallel_lines(self):
        cl1 = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        cl2 = np.column_stack([np.linspace(0, 100, 50), np.ones(50) * 20])
        assert count_intersections([cl1, cl2]) == 0

    def test_crossing_lines(self):
        # X-pattern crossing
        cl1 = np.column_stack([np.linspace(0, 100, 50), np.linspace(0, 100, 50)])
        cl2 = np.column_stack([np.linspace(0, 100, 50), np.linspace(100, 0, 50)])
        count = count_intersections([cl1, cl2])
        assert count >= 1

    def test_single_centerline(self):
        cl = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        assert count_intersections([cl]) == 0


# ---------------------------------------------------------------------------
# compute_pore_size_distribution
# ---------------------------------------------------------------------------

class TestPoreSizeDistribution:
    def test_fully_covered_no_pores(self):
        mask = np.ones((50, 50), dtype=bool)
        stats = compute_pore_size_distribution(mask)
        assert stats.pore_count == 0

    def test_uniform_background(self):
        # Thin vertical strip of fiber — large background
        mask = np.zeros((100, 100), dtype=bool)
        mask[:, 49:51] = True
        stats = compute_pore_size_distribution(mask)
        assert stats.pore_count > 0
        assert stats.mean_pore_size > 0
        assert stats.max_pore_size >= stats.mean_pore_size


# ---------------------------------------------------------------------------
# compute_junction_density
# ---------------------------------------------------------------------------

class TestJunctionDensity:
    def test_zero_intersections(self):
        assert compute_junction_density(0, 100, 100) == 0.0

    def test_known_density(self):
        d = compute_junction_density(5, 100, 100)
        assert d == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_image_morphology (integration)
# ---------------------------------------------------------------------------

class TestComputeImageMorphology:
    def _make_horizontal_fiber_mask(self, H=100, W=200):
        mask = np.zeros((H, W), dtype=bool)
        mask[47:53, 10:190] = True
        return mask

    def test_basic_integration(self):
        H, W = 100, 200
        mask = self._make_horizontal_fiber_mask(H, W)
        cl = np.column_stack([np.linspace(10, 190, 40), np.full(40, 50.0)])

        result = compute_image_morphology(
            masks=[mask],
            centerlines=[cl],
            widths=[6.0],
            lengths=[180.0],
            curvatures=[0.001],
            orientations=[0.0],
            tortuosities=[1.01],
            image_height=H,
            image_width=W,
        )

        assert 0 < result.coverage_ratio < 1
        assert 0 < result.porosity < 1
        assert result.porosity + result.coverage_ratio == pytest.approx(1.0, abs=1e-6)
        assert result.mean_fiber_width == pytest.approx(6.0)
        assert result.mean_fiber_length == pytest.approx(180.0)
        assert result.alignment_score > 0.9   # horizontal fiber → aligned

    def test_empty_input(self):
        result = compute_image_morphology(
            masks=[], centerlines=[], widths=[], lengths=[],
            curvatures=[], orientations=[], tortuosities=[],
            image_height=100, image_width=100,
        )
        assert result.coverage_ratio == 0.0
        assert result.porosity == 0.0
