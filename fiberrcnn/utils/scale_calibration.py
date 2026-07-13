"""
SEM Scale Calibration Utilities
================================
Tools for:

1. Manual pixel-to-nm calibration via a known scale value.
2. Automatic SEM scale-bar detection from the image footer region
   (heuristic approach; works for many standard SEM layouts).
3. Conversion of all fiber measurements to physical units.

Example
-------
>>> from fiberrcnn.utils.scale_calibration import ScaleCalibrator
>>> cal = ScaleCalibrator.from_manual(nm_per_pixel=10.5)
>>> width_nm = cal.to_nm(fiber_width_pixels)

>>> cal = ScaleCalibrator.from_image("sem_image.png")
>>> results_nm = cal.convert_prediction(prediction)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """Result of scale bar detection."""

    nm_per_pixel: float
    method: str                        # "manual" | "scale_bar" | "metadata"
    scale_bar_length_px: float = 0.0
    scale_bar_value_nm: float = 0.0
    confidence: float = 1.0
    notes: str = ""

    @property
    def um_per_pixel(self) -> float:
        return self.nm_per_pixel / 1000.0

    def __repr__(self) -> str:
        return (
            f"CalibrationResult("
            f"nm/px={self.nm_per_pixel:.4f}, "
            f"method='{self.method}', "
            f"confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Scale bar text parser
# ---------------------------------------------------------------------------

_UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1e3,
    "µm": 1e3,
    "μm": 1e3,
    "mm": 1e6,
}

_SCALE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(nm|um|µm|μm|mm)",
    re.IGNORECASE,
)


def parse_scale_text(text: str) -> tuple[float, str] | None:
    """Parse a scale bar label like '500 nm', '2 µm', '0.5 mm'.

    Returns
    -------
    (value_nm, raw_text) or None if not matched.
    """
    m = _SCALE_PATTERN.search(text)
    if m is None:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    factor = _UNIT_TO_NM.get(unit, 1.0)
    return value * factor, m.group(0)


# ---------------------------------------------------------------------------
# Scale bar detector (heuristic)
# ---------------------------------------------------------------------------

class ScaleBarDetector:
    """Detect a horizontal scale bar in the bottom region of a SEM image.

    Strategy
    --------
    1. Crop the bottom ``footer_fraction`` of the image (typically contains
       the SEM metadata strip).
    2. Threshold to find bright horizontal lines on a dark background.
    3. Find the longest connected horizontal segment — this is the scale bar.
    4. Optionally run OCR (pytesseract) on the footer to extract the label.

    Parameters
    ----------
    footer_fraction : float
        Fraction of image height to search (bottom portion).
    min_bar_fraction : float
        Minimum bar length as a fraction of image width.
    """

    def __init__(
        self,
        footer_fraction: float = 0.12,
        min_bar_fraction: float = 0.03,
    ) -> None:
        self.footer_fraction = footer_fraction
        self.min_bar_fraction = min_bar_fraction

    def detect(
        self,
        image_bgr: np.ndarray,
        scale_label_nm: float | None = None,
    ) -> CalibrationResult | None:
        """Detect the scale bar and return nm/pixel calibration.

        Parameters
        ----------
        image_bgr : (H, W, 3) uint8 BGR image
        scale_label_nm : known scale bar physical length in nm.
            If None, OCR is attempted (requires pytesseract).

        Returns
        -------
        CalibrationResult or None if detection fails.
        """
        H, W = image_bgr.shape[:2]
        footer_h = int(H * self.footer_fraction)
        footer = image_bgr[H - footer_h :, :]

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(footer, cv2.COLOR_BGR2GRAY)
        # Otsu threshold to separate bright scale bar from dark background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to connect broken bar segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find horizontal runs: sum each row, pick the one with the longest run
        bar_length_px = self._find_longest_horizontal_run(closed, W)

        if bar_length_px < W * self.min_bar_fraction:
            return None

        # Attempt OCR if no label provided
        if scale_label_nm is None:
            scale_label_nm = self._ocr_scale_label(footer)

        if scale_label_nm is None:
            return None

        nm_per_pixel = scale_label_nm / bar_length_px

        return CalibrationResult(
            nm_per_pixel=nm_per_pixel,
            method="scale_bar",
            scale_bar_length_px=float(bar_length_px),
            scale_bar_value_nm=float(scale_label_nm),
            confidence=0.85,
            notes=f"Detected bar={bar_length_px:.0f}px, label={scale_label_nm:.0f}nm",
        )

    @staticmethod
    def _find_longest_horizontal_run(binary: np.ndarray, width: int) -> float:
        """Return the pixel length of the longest bright horizontal segment."""
        best = 0
        for row in binary:
            in_run = False
            run_len = 0
            for px in row:
                if px > 128:
                    run_len += 1
                    in_run = True
                else:
                    if in_run:
                        best = max(best, run_len)
                        run_len = 0
                    in_run = False
            best = max(best, run_len)
        return float(best)

    @staticmethod
    def _ocr_scale_label(footer_bgr: np.ndarray) -> float | None:
        """Attempt OCR on the footer region to extract the scale label."""
        try:
            import pytesseract

            gray = cv2.cvtColor(footer_bgr, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--psm 6")
            result = parse_scale_text(text)
            if result:
                return result[0]
        except (ImportError, Exception):
            pass
        return None


# ---------------------------------------------------------------------------
# ScaleCalibrator — main API
# ---------------------------------------------------------------------------

class ScaleCalibrator:
    """Convert pixel measurements to physical units.

    Create via factory methods:
    * ``from_manual(nm_per_pixel)``
    * ``from_image(image_path, scale_nm)``
    * ``from_metadata(tiff_path)`` — reads embedded TIFF metadata
    """

    def __init__(self, calibration: CalibrationResult) -> None:
        self._cal = calibration

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_manual(cls, nm_per_pixel: float) -> "ScaleCalibrator":
        """Create from a known calibration value."""
        return cls(CalibrationResult(
            nm_per_pixel=nm_per_pixel,
            method="manual",
            confidence=1.0,
        ))

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        scale_label_nm: float | None = None,
        footer_fraction: float = 0.12,
    ) -> "ScaleCalibrator | None":
        """Auto-detect scale bar in a SEM image.

        Parameters
        ----------
        image_path : path to SEM image
        scale_label_nm : known scale bar physical length in nm.
            If None, OCR is attempted.
        footer_fraction : fraction of image height to search.

        Returns
        -------
        ScaleCalibrator or None if detection fails.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        detector = ScaleBarDetector(footer_fraction=footer_fraction)
        result = detector.detect(img, scale_label_nm)
        if result is None:
            return None
        return cls(result)

    @classmethod
    def from_metadata(cls, tiff_path: str | Path) -> "ScaleCalibrator | None":
        """Read pixel size from TIFF metadata (FEI/Thermo-Fisher SEM format).

        Returns None if metadata is absent or unrecognised.
        """
        try:
            from PIL import Image
            from PIL.TiffImagePlugin import IFDRational

            img = Image.open(str(tiff_path))
            meta = img.tag_v2 if hasattr(img, "tag_v2") else {}

            # FEI stores pixel size in tag 65009 (PixelWidth) in metres
            pixel_width_m = None
            for tag_id in (65009, 65010):
                if tag_id in meta:
                    val = meta[tag_id]
                    if isinstance(val, (int, float)):
                        pixel_width_m = float(val)
                    elif isinstance(val, IFDRational):
                        pixel_width_m = float(val)
                    break

            if pixel_width_m is not None and pixel_width_m > 0:
                nm_per_pixel = pixel_width_m * 1e9
                return cls(CalibrationResult(
                    nm_per_pixel=nm_per_pixel,
                    method="metadata",
                    confidence=0.99,
                    notes=f"FEI TIFF tag: {pixel_width_m:.3e} m/px",
                ))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    @property
    def nm_per_pixel(self) -> float:
        return self._cal.nm_per_pixel

    @property
    def calibration(self) -> CalibrationResult:
        return self._cal

    def to_nm(self, pixels: float) -> float:
        """Convert a pixel length to nanometres."""
        return pixels * self._cal.nm_per_pixel

    def to_um(self, pixels: float) -> float:
        """Convert a pixel length to micrometres."""
        return pixels * self._cal.nm_per_pixel / 1000.0

    def to_pixels(self, nm: float) -> float:
        """Convert nanometres to pixels."""
        return nm / self._cal.nm_per_pixel

    def area_to_nm2(self, area_px: float) -> float:
        """Convert pixel² area to nm²."""
        return area_px * (self._cal.nm_per_pixel ** 2)

    # ------------------------------------------------------------------
    # Batch conversion
    # ------------------------------------------------------------------

    def convert_prediction(
        self,
        prediction: Any,
        inplace: bool = False,
    ) -> Any:
        """Apply calibration to all length/area fields in an ImagePrediction.

        Parameters
        ----------
        prediction : ImagePrediction (from fiberrcnn.engine.inference)
        inplace : if True modify in place; otherwise work on a copy

        Returns
        -------
        ImagePrediction with physical-unit fields added (``_nm`` suffix).
        """
        import copy

        if not inplace:
            prediction = copy.deepcopy(prediction)

        for fiber in prediction.fiber_instances:
            fiber.fiber_width_nm = self.to_nm(fiber.fiber_width)
            fiber.fiber_length_nm = self.to_nm(fiber.fiber_length)

        m = prediction.image_metrics
        for key in ("mean_fiber_width", "mean_fiber_length",
                    "mean_pore_size", "median_pore_size", "max_pore_size"):
            if key in m:
                m[f"{key}_nm"] = self.to_nm(m[key])

        m["nm_per_pixel"] = self._cal.nm_per_pixel
        m["calibration_method"] = self._cal.method

        return prediction

    def summary(self) -> str:
        return (
            f"ScaleCalibrator | {self._cal.nm_per_pixel:.4f} nm/px | "
            f"method={self._cal.method} | confidence={self._cal.confidence:.2f}"
        )
