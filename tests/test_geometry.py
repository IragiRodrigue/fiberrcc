"""
Unit tests for fiberrcnn.geometry
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fiberrcnn.geometry.fiber_geometry import (
    compute_curvature,
    compute_fiber_geometry,
    compute_length,
    compute_orientation,
    compute_tortuosity,
    estimate_width,
    extract_centerline,
    generate_keypoints,
    polygon_to_mask,
    resample_centerline,
    skeletonize_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def straight_rect_polygon():
    """A thin horizontal rectangle ~ 5 px tall × 100 px wide."""
    return [[10.0, 47.0], [110.0, 47.0], [110.0, 52.0], [10.0, 52.0]]


@pytest.fixture
def straight_rect_mask(straight_rect_polygon):
    return polygon_to_mask(straight_rect_polygon, 100, 120)


# ---------------------------------------------------------------------------
# polygon_to_mask
# ---------------------------------------------------------------------------

class TestPolygonToMask:
    def test_shape(self, straight_rect_mask):
        assert straight_rect_mask.shape == (100, 120)
        assert straight_rect_mask.dtype == bool

    def test_coverage(self, straight_rect_mask):
        filled = straight_rect_mask.sum()
        # Rough check: 100 px wide × 5 px tall ≈ 500 px
        assert 400 < filled < 700

    def test_out_of_bounds_clipped(self):
        # Polygon slightly outside the canvas must not raise
        poly = [[-5.0, -5.0], [200.0, -5.0], [200.0, 200.0], [-5.0, 200.0]]
        mask = polygon_to_mask(poly, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.all()  # entire canvas should be filled

    def test_degenerate_polygon(self):
        # Triangle with all points at same location → very small area
        poly = [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]]
        mask = polygon_to_mask(poly, 100, 100)
        assert mask.sum() <= 1


# ---------------------------------------------------------------------------
# skeletonize_mask
# ---------------------------------------------------------------------------

class TestSkeletonize:
    def test_returns_bool_array(self, straight_rect_mask):
        skel = skeletonize_mask(straight_rect_mask)
        assert skel.dtype == bool
        assert skel.shape == straight_rect_mask.shape

    def test_thinner_than_mask(self, straight_rect_mask):
        skel = skeletonize_mask(straight_rect_mask)
        # Skeleton should have fewer pixels than the full mask
        assert skel.sum() < straight_rect_mask.sum()


# ---------------------------------------------------------------------------
# extract_centerline
# ---------------------------------------------------------------------------

class TestExtractCenterline:
    def test_returns_2d_array(self, straight_rect_mask):
        cl = extract_centerline(straight_rect_mask)
        assert cl.ndim == 2
        assert cl.shape[1] == 2

    def test_horizontal_fiber_orientation(self, straight_rect_mask):
        cl = extract_centerline(straight_rect_mask)
        # For a horizontal rectangle the x-extent should be >> y-extent
        x_range = cl[:, 0].max() - cl[:, 0].min()
        y_range = cl[:, 1].max() - cl[:, 1].min()
        assert x_range > 4 * y_range


# ---------------------------------------------------------------------------
# resample_centerline
# ---------------------------------------------------------------------------

class TestResampleCenterline:
    def test_output_shape(self):
        pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        resampled = resample_centerline(pts, n_points=40)
        assert resampled.shape == (40, 2)

    def test_single_point_input(self):
        pts = np.array([[50.0, 50.0]])
        resampled = resample_centerline(pts, n_points=40)
        assert resampled.shape == (40, 2)
        assert np.allclose(resampled, 50.0)


# ---------------------------------------------------------------------------
# generate_keypoints
# ---------------------------------------------------------------------------

class TestGenerateKeypoints:
    def test_shape_and_visibility(self):
        pts = np.column_stack([np.linspace(0, 100, 40), np.zeros(40)])
        kps = generate_keypoints(pts)
        assert kps.shape == (40, 3)
        assert (kps[:, 2] == 2).all()

    def test_wrong_input_raises(self):
        pts = np.zeros((10, 2))
        with pytest.raises(AssertionError):
            generate_keypoints(pts)


# ---------------------------------------------------------------------------
# estimate_width
# ---------------------------------------------------------------------------

class TestEstimateWidth:
    def test_positive_width(self, straight_rect_mask):
        cl = extract_centerline(straight_rect_mask)
        w = estimate_width(straight_rect_mask, cl)
        assert w > 0

    def test_width_approx_correct(self, straight_rect_mask):
        cl = extract_centerline(straight_rect_mask)
        w = estimate_width(straight_rect_mask, cl)
        # Rectangle is ~5 px tall so width ≈ 5 px
        assert 3.0 < w < 10.0


# ---------------------------------------------------------------------------
# compute_length
# ---------------------------------------------------------------------------

class TestComputeLength:
    def test_straight_line(self):
        pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        length = compute_length(pts)
        assert abs(length - 100.0) < 2.0

    def test_single_point_returns_zero(self):
        pts = np.array([[5.0, 5.0]])
        assert compute_length(pts) == 0.0


# ---------------------------------------------------------------------------
# compute_curvature
# ---------------------------------------------------------------------------

class TestComputeCurvature:
    def test_straight_line_near_zero(self):
        pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        k = compute_curvature(pts)
        assert k < 1e-4

    def test_circle_positive_curvature(self):
        theta = np.linspace(0, 2 * math.pi, 100)
        r = 50.0
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        k = compute_curvature(pts)
        # Curvature of a circle = 1/r ≈ 0.02
        assert 0.005 < k < 0.05


# ---------------------------------------------------------------------------
# compute_orientation
# ---------------------------------------------------------------------------

class TestComputeOrientation:
    def test_horizontal_fiber(self):
        pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        angle = compute_orientation(pts)
        assert abs(angle) < 5.0 or abs(angle - 180.0) < 5.0

    def test_vertical_fiber(self):
        pts = np.column_stack([np.zeros(50), np.linspace(0, 100, 50)])
        angle = compute_orientation(pts)
        assert abs(angle - 90.0) < 5.0


# ---------------------------------------------------------------------------
# compute_tortuosity
# ---------------------------------------------------------------------------

class TestComputeTortuosity:
    def test_straight_line_is_one(self):
        pts = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        t = compute_tortuosity(pts)
        assert abs(t - 1.0) < 0.02

    def test_tortuous_path_greater_than_one(self):
        # Sinusoidal path
        x = np.linspace(0, 100, 200)
        y = 10 * np.sin(x / 10.0)
        pts = np.column_stack([x, y])
        t = compute_tortuosity(pts)
        assert t > 1.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestComputeFiberGeometry:
    def test_full_pipeline(self, straight_rect_polygon):
        geom = compute_fiber_geometry(straight_rect_polygon, 100, 120)

        assert len(geom.bbox) == 4
        assert geom.bbox[2] > 0 and geom.bbox[3] > 0
        assert geom.keypoints.shape == (40, 3)
        assert geom.fiber_width > 0
        assert geom.fiber_length > 0
        assert geom.fiber_tortuosity >= 1.0
        assert 0.0 <= geom.fiber_orientation < 180.0
        assert geom.area > 0

    def test_tiny_polygon_does_not_crash(self):
        poly = [[50.0, 50.0], [51.0, 50.0], [51.0, 51.0], [50.0, 51.0]]
        geom = compute_fiber_geometry(poly, 100, 100)
        assert geom is not None
