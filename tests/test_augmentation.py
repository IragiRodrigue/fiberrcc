"""
Unit tests for fiberrcnn.data.augmentation (pure numpy transforms).
Detectron2-dependent augmentations are skipped if not available.
"""

from __future__ import annotations

import numpy as np
import pytest

from fiberrcnn.data.augmentation import (
    add_gaussian_noise,
    add_poisson_noise,
    add_charging_artefact,
    apply_depth_blur,
    adjust_brightness_contrast,
    apply_histogram_equalisation,
    apply_motion_blur,
)


@pytest.fixture
def sample_image():
    rng = np.random.default_rng(0)
    return rng.integers(50, 200, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def grey_image():
    rng = np.random.default_rng(1)
    return rng.integers(50, 200, (128, 128), dtype=np.uint8)


# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:
    def test_output_shape_unchanged(self, sample_image):
        out = add_gaussian_noise(sample_image, std=10.0)
        assert out.shape == sample_image.shape

    def test_output_dtype_uint8(self, sample_image):
        out = add_gaussian_noise(sample_image, std=10.0)
        assert out.dtype == np.uint8

    def test_values_in_range(self, sample_image):
        out = add_gaussian_noise(sample_image, std=50.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_output_different_from_input(self, sample_image):
        out = add_gaussian_noise(sample_image, std=20.0)
        assert not np.array_equal(out, sample_image)

    def test_zero_std_equals_input(self, sample_image):
        out = add_gaussian_noise(sample_image, std=0.0)
        assert np.array_equal(out, sample_image)


# ---------------------------------------------------------------------------
# add_poisson_noise
# ---------------------------------------------------------------------------

class TestPoissonNoise:
    def test_output_shape(self, sample_image):
        out = add_poisson_noise(sample_image)
        assert out.shape == sample_image.shape

    def test_values_in_range(self, sample_image):
        out = add_poisson_noise(sample_image, scale=1.0)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_dtype_uint8(self, sample_image):
        assert add_poisson_noise(sample_image).dtype == np.uint8


# ---------------------------------------------------------------------------
# add_charging_artefact
# ---------------------------------------------------------------------------

class TestChargingArtefact:
    def test_output_shape(self, sample_image):
        out = add_charging_artefact(sample_image)
        assert out.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        assert add_charging_artefact(sample_image).dtype == np.uint8

    def test_values_in_range(self, sample_image):
        out = add_charging_artefact(sample_image, intensity=100.0, n_bands=5)
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# apply_depth_blur
# ---------------------------------------------------------------------------

class TestDepthBlur:
    def test_output_shape(self, sample_image):
        out = apply_depth_blur(sample_image)
        assert out.shape == sample_image.shape

    def test_output_dtype(self, sample_image):
        assert apply_depth_blur(sample_image).dtype == np.uint8

    def test_blurred_region_differs(self, sample_image):
        # Deterministic: fix random seed
        np.random.seed(42)
        out = apply_depth_blur(sample_image, sigma=3.0, region_fraction=0.5)
        assert not np.array_equal(out, sample_image)


# ---------------------------------------------------------------------------
# adjust_brightness_contrast
# ---------------------------------------------------------------------------

class TestBrightnessContrast:
    def test_identity(self, sample_image):
        out = adjust_brightness_contrast(sample_image, alpha=1.0, beta=0.0)
        assert np.array_equal(out, sample_image)

    def test_clipping(self, sample_image):
        out = adjust_brightness_contrast(sample_image, alpha=5.0, beta=200.0)
        assert out.max() <= 255
        assert out.min() >= 0

    def test_darkening(self, sample_image):
        out = adjust_brightness_contrast(sample_image, alpha=0.5, beta=-50.0)
        assert out.mean() < sample_image.mean()

    def test_brightening(self, sample_image):
        # Use a dark image so there's room to brighten
        dark = np.full_like(sample_image, 30)
        out = adjust_brightness_contrast(dark, alpha=1.0, beta=80.0)
        assert out.mean() > dark.mean()


# ---------------------------------------------------------------------------
# apply_histogram_equalisation (CLAHE)
# ---------------------------------------------------------------------------

class TestCLAHE:
    def test_output_shape_colour(self, sample_image):
        out = apply_histogram_equalisation(sample_image)
        assert out.shape == sample_image.shape

    def test_output_shape_grey(self, grey_image):
        out = apply_histogram_equalisation(grey_image)
        assert out.shape == grey_image.shape

    def test_output_dtype(self, sample_image):
        assert apply_histogram_equalisation(sample_image).dtype == np.uint8

    def test_output_values_in_range(self, sample_image):
        out = apply_histogram_equalisation(sample_image)
        assert out.min() >= 0
        assert out.max() <= 255


# ---------------------------------------------------------------------------
# apply_motion_blur
# ---------------------------------------------------------------------------

class TestMotionBlur:
    def test_output_shape(self, sample_image):
        out = apply_motion_blur(sample_image, kernel_size=5)
        assert out.shape == sample_image.shape

    def test_output_different_from_input(self, sample_image):
        out = apply_motion_blur(sample_image, kernel_size=7)
        assert not np.array_equal(out, sample_image)


# ---------------------------------------------------------------------------
# Detectron2 augmentation wrappers (optional)
# ---------------------------------------------------------------------------

class TestDetectron2Wrappers:
    def test_random_gaussian_noise_aug(self, sample_image):
        try:
            from fiberrcnn.data.augmentation import RandomGaussianNoise
        except ImportError:
            pytest.skip("detectron2 not available")

        aug = RandomGaussianNoise(std_range=(5.0, 10.0), p=1.0)
        transform = aug.get_transform(sample_image)
        out = transform.apply_image(sample_image)
        assert out.shape == sample_image.shape

    def test_random_brightness_contrast_aug(self, sample_image):
        try:
            from fiberrcnn.data.augmentation import RandomBrightnessContrast
        except ImportError:
            pytest.skip("detectron2 not available")

        aug = RandomBrightnessContrast(p=1.0)
        transform = aug.get_transform(sample_image)
        out = transform.apply_image(sample_image)
        assert out.shape == sample_image.shape

    def test_no_op_at_zero_probability(self, sample_image):
        try:
            from fiberrcnn.data.augmentation import RandomGaussianNoise
            from detectron2.data.transforms import NoOpTransform
        except ImportError:
            pytest.skip("detectron2 not available")

        aug = RandomGaussianNoise(p=0.0)
        transform = aug.get_transform(sample_image)
        assert isinstance(transform, NoOpTransform)
