"""
Unit tests for fiberrcnn.visualization
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from fiberrcnn.visualization.fiber_viz import (
    draw_centerlines,
    draw_instance_overlay,
    draw_orientation_map,
    draw_pore_map,
    draw_porosity_map,
    draw_width_map,
    plot_histogram,
    plot_rose,
    save_visualisation_report,
)


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def sample_masks():
    masks = []
    m1 = np.zeros((200, 300), dtype=bool)
    m1[50:60, 20:280] = True
    m2 = np.zeros((200, 300), dtype=bool)
    m2[100:110, 20:280] = True
    return [m1, m2]


@pytest.fixture
def sample_centerlines():
    cl1 = np.column_stack([np.linspace(20, 280, 40), np.full(40, 55.0)])
    cl2 = np.column_stack([np.linspace(20, 280, 40), np.full(40, 105.0)])
    return [cl1, cl2]


# ---------------------------------------------------------------------------

class TestDrawInstanceOverlay:
    def test_output_shape(self, sample_image, sample_masks):
        out = draw_instance_overlay(sample_image, sample_masks)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_with_boxes_and_scores(self, sample_image, sample_masks):
        boxes = [[20, 50, 280, 60], [20, 100, 280, 110]]
        scores = [0.9, 0.7]
        out = draw_instance_overlay(sample_image, sample_masks, boxes=boxes, scores=scores)
        assert out.shape == sample_image.shape


class TestDrawCenterlines:
    def test_output_shape(self, sample_image, sample_centerlines):
        out = draw_centerlines(sample_image, sample_centerlines)
        assert out.shape == sample_image.shape

    def test_with_keypoints(self, sample_image, sample_centerlines):
        kps = [cl for cl in sample_centerlines]
        out = draw_centerlines(sample_image, sample_centerlines, keypoints=kps)
        assert out.shape == sample_image.shape


class TestDrawWidthMap:
    def test_output_shape(self, sample_masks):
        wmap = draw_width_map(sample_masks, [6.0, 8.0], (200, 300))
        assert wmap.shape == (200, 300, 3)
        assert wmap.dtype == np.uint8


class TestDrawOrientationMap:
    def test_output_shape(self, sample_masks):
        omap = draw_orientation_map(sample_masks, [0.0, 45.0], (200, 300))
        assert omap.shape == (200, 300, 3)
        assert omap.dtype == np.uint8


class TestDrawPoreMap:
    def test_output_shape(self, sample_masks):
        combined = np.zeros((200, 300), dtype=bool)
        for m in sample_masks:
            combined |= m
        pmap = draw_pore_map(combined)
        assert pmap.shape == (200, 300, 3)

    def test_fully_covered_returns_black(self):
        full = np.ones((50, 50), dtype=bool)
        pmap = draw_pore_map(full)
        assert pmap.sum() == 0


class TestDrawPorosityMap:
    def test_output_shape(self, sample_masks):
        combined = np.zeros((200, 300), dtype=bool)
        for m in sample_masks:
            combined |= m
        pmap = draw_porosity_map(combined)
        assert pmap.shape == (200, 300, 3)


class TestPlotHistogram:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        values = list(np.random.randn(100))
        fig = plot_histogram(values, "Test", "Value")
        assert fig is not None
        plt.close(fig)

    def test_saves_to_disk(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "hist.png"
            fig = plot_histogram(list(np.random.randn(50)), "T", "X", output_path=p)
            plt.close(fig)
            assert p.exists()


class TestPlotRose:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        angles = list(np.random.uniform(0, 180, 50))
        fig = plot_rose(angles)
        assert fig is not None
        plt.close(fig)


class TestSaveVisualisationReport:
    def test_creates_output_files(self, sample_image, sample_masks, sample_centerlines):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            save_visualisation_report(
                image=sample_image,
                masks=sample_masks,
                centerlines=sample_centerlines,
                keypoints_list=sample_centerlines,
                widths=[6.0, 8.0],
                lengths=[260.0, 260.0],
                orientations=[0.0, 5.0],
                curvatures=[0.001, 0.002],
                tortuosities=[1.01, 1.02],
                output_dir=tmp,
                image_name="test",
            )
            out = Path(tmp)
            assert (out / "test_overlay.png").exists()
            assert (out / "test_centerlines.png").exists()
        plt.close("all")
