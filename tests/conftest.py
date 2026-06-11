"""
Shared pytest configuration and fixtures.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Ensure deterministic behaviour in all tests."""
    np.random.seed(0)
    torch.manual_seed(0)
    yield


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_bgr_image():
    """100×150 synthetic BGR image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def simple_fiber_polygon():
    """A simple horizontal fiber polygon."""
    return [[10.0, 45.0], [140.0, 45.0], [140.0, 55.0], [10.0, 55.0]]
