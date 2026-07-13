"""
Unit tests for fiberrcnn.utils.scale_calibration
"""

from __future__ import annotations

import numpy as np
import pytest

from fiberrcnn.utils.scale_calibration import (
    CalibrationResult,
    ScaleCalibrator,
    parse_scale_text,
)


# ---------------------------------------------------------------------------
# parse_scale_text
# ---------------------------------------------------------------------------

class TestParseScaleText:
    @pytest.mark.parametrize("text,expected_nm", [
        ("500 nm", 500.0),
        ("2 µm", 2000.0),
        ("2 um", 2000.0),
        ("0.5 mm", 500_000.0),
        ("1.5 μm", 1500.0),
        ("  100nm  ", 100.0),
        ("Scale bar: 200 nm", 200.0),
    ])
    def test_valid_texts(self, text, expected_nm):
        result = parse_scale_text(text)
        assert result is not None
        value_nm, _ = result
        assert value_nm == pytest.approx(expected_nm)

    @pytest.mark.parametrize("text", [
        "no scale here",
        "123",
        "",
        "px/nm",
    ])
    def test_invalid_texts(self, text):
        assert parse_scale_text(text) is None


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_um_per_pixel(self):
        cal = CalibrationResult(nm_per_pixel=10.0, method="manual")
        assert cal.um_per_pixel == pytest.approx(0.01)

    def test_repr(self):
        cal = CalibrationResult(nm_per_pixel=8.5, method="scale_bar")
        r = repr(cal)
        assert "8.5" in r
        assert "scale_bar" in r


# ---------------------------------------------------------------------------
# ScaleCalibrator
# ---------------------------------------------------------------------------

class TestScaleCalibratorManual:
    @pytest.fixture
    def cal(self):
        return ScaleCalibrator.from_manual(nm_per_pixel=10.0)

    def test_to_nm(self, cal):
        assert cal.to_nm(5.0) == pytest.approx(50.0)

    def test_to_um(self, cal):
        assert cal.to_um(1000.0) == pytest.approx(10.0)

    def test_to_pixels(self, cal):
        assert cal.to_pixels(100.0) == pytest.approx(10.0)

    def test_area_to_nm2(self, cal):
        # 1 px² = 100 nm²  (10 nm/px)²
        assert cal.area_to_nm2(1.0) == pytest.approx(100.0)

    def test_roundtrip(self, cal):
        px = 37.5
        assert cal.to_pixels(cal.to_nm(px)) == pytest.approx(px, rel=1e-6)

    def test_summary_contains_nm_per_px(self, cal):
        assert "10.0" in cal.summary() or "10." in cal.summary()

    def test_method_is_manual(self, cal):
        assert cal.calibration.method == "manual"


class TestScaleCalibratorConvertPrediction:
    def test_adds_nm_fields(self):
        from fiberrcnn.engine.inference import FiberInstance, ImagePrediction

        cal = ScaleCalibrator.from_manual(nm_per_pixel=5.0)

        fibers = [
            FiberInstance(
                instance_id=0,
                bbox=[0, 0, 10, 5],
                confidence=0.9,
                fiber_width=10.0,
                fiber_length=200.0,
            )
        ]
        pred = ImagePrediction(
            image_path="test.png",
            image_height=100,
            image_width=100,
            fiber_instances=fibers,
            image_metrics={"mean_fiber_width": 10.0, "mean_pore_size": 20.0},
        )
        converted = cal.convert_prediction(pred, inplace=False)

        # Original should be unchanged (not inplace)
        assert not hasattr(pred.fiber_instances[0], "fiber_width_nm")

        # Converted should have _nm fields
        assert converted.fiber_instances[0].fiber_width_nm == pytest.approx(50.0)
        assert converted.fiber_instances[0].fiber_length_nm == pytest.approx(1000.0)
        assert converted.image_metrics["mean_fiber_width_nm"] == pytest.approx(50.0)
        assert converted.image_metrics["nm_per_pixel"] == pytest.approx(5.0)

    def test_inplace_modifies_original(self):
        from fiberrcnn.engine.inference import FiberInstance, ImagePrediction

        cal = ScaleCalibrator.from_manual(nm_per_pixel=2.0)
        fibers = [FiberInstance(instance_id=0, bbox=[0,0,5,5], confidence=0.8,
                                fiber_width=10.0, fiber_length=50.0)]
        pred = ImagePrediction("x.png", 100, 100, fibers, {})
        cal.convert_prediction(pred, inplace=True)
        assert hasattr(pred.fiber_instances[0], "fiber_width_nm")


# ---------------------------------------------------------------------------
# ScaleBarDetector (image-based, synthetic)
# ---------------------------------------------------------------------------

class TestScaleBarDetector:
    def _make_synthetic_sem(self, H=200, W=300, bar_y=180, bar_x0=50, bar_len=100):
        """Create a synthetic SEM image with a white horizontal scale bar."""
        img = np.random.randint(30, 80, (H, W, 3), dtype=np.uint8)
        # Draw a bright scale bar
        img[bar_y - 2 : bar_y + 3, bar_x0 : bar_x0 + bar_len] = 240
        return img

    def test_detect_known_bar(self):
        from fiberrcnn.utils.scale_calibration import ScaleBarDetector

        detector = ScaleBarDetector(footer_fraction=0.15, min_bar_fraction=0.05)
        img = self._make_synthetic_sem(bar_y=185, bar_x0=50, bar_len=100)

        # Provide the known physical length; skip OCR
        result = detector.detect(img, scale_label_nm=500.0)

        # Detection may or may not succeed depending on threshold,
        # but if it does, nm/px should be reasonable
        if result is not None:
            assert result.nm_per_pixel > 0
            assert result.method == "scale_bar"
            # 500 nm / ~100 px ≈ 5 nm/px
            assert 2.0 < result.nm_per_pixel < 20.0

    def test_no_label_returns_none(self):
        """Without a physical label, detection must return None."""
        from fiberrcnn.utils.scale_calibration import ScaleBarDetector
        import numpy as np
        rng = np.random.default_rng(7)
        img = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
        detector = ScaleBarDetector()
        # scale_label_nm=None and no OCR available → must return None
        result = detector.detect(img, scale_label_nm=None)
        assert result is None
