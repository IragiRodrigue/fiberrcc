"""
FiberRCNN Visualization
========================
Generates publication-quality visualizations:

* Instance overlays with coloured masks
* Centerlines and keypoints
* Width heatmaps
* Orientation maps
* Pore size maps
* Porosity maps
* Width / length / orientation histograms
* Orientation rose plots
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import distance_transform_edt

# Default colormap for fiber instances
_CMAP = plt.get_cmap("tab20")


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _instance_color(idx: int) -> tuple[int, int, int]:
    """Return an RGB colour (0-255) for instance index *idx*."""
    rgba = _CMAP(idx % 20)
    return int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)


# ---------------------------------------------------------------------------
# Instance overlay
# ---------------------------------------------------------------------------

def draw_instance_overlay(
    image: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[list[float]] | None = None,
    scores: list[float] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend instance masks over *image*.

    Parameters
    ----------
    image : (H, W, 3) uint8 BGR image
    masks : list of (H, W) bool masks
    boxes : optional list of [x1, y1, x2, y2] boxes
    scores : optional detection confidence scores
    alpha : mask blend weight

    Returns
    -------
    vis : (H, W, 3) uint8 image
    """
    vis = image.copy()
    for i, mask in enumerate(masks):
        color = _instance_color(i)
        overlay = vis.copy()
        overlay[mask] = color[::-1]  # BGR
        vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)

        if boxes is not None:
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color[::-1], 2)
            if scores is not None:
                label = f"{scores[i]:.2f}"
                cv2.putText(
                    vis, label, (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1
                )
    return vis


# ---------------------------------------------------------------------------
# Centerline + keypoint overlay
# ---------------------------------------------------------------------------

def draw_centerlines(
    image: np.ndarray,
    centerlines: list[np.ndarray],
    keypoints: list[np.ndarray] | None = None,
    line_thickness: int = 1,
    kp_radius: int = 2,
) -> np.ndarray:
    """Draw fiber centerlines (and optionally keypoints) on *image*.

    Parameters
    ----------
    image : (H, W, 3) uint8 BGR
    centerlines : list of (N, 2) arrays in (x, y)
    keypoints : optional list of (K, 2) arrays in (x, y)
    """
    vis = image.copy()
    for i, cl in enumerate(centerlines):
        color = _instance_color(i)[::-1]  # BGR
        pts = cl.astype(np.int32)

        for j in range(len(pts) - 1):
            cv2.line(vis, tuple(pts[j]), tuple(pts[j + 1]), color, line_thickness)

        if keypoints is not None:
            kps = keypoints[i].astype(np.int32)
            for kp in kps:
                cv2.circle(vis, tuple(kp[:2]), kp_radius, (255, 255, 255), -1)
                cv2.circle(vis, tuple(kp[:2]), kp_radius + 1, color, 1)

    return vis


# ---------------------------------------------------------------------------
# Width heatmap
# ---------------------------------------------------------------------------

def draw_width_map(
    masks: list[np.ndarray],
    widths: list[float],
    image_shape: tuple[int, int],
    cmap: str = "plasma",
) -> np.ndarray:
    """Generate a per-pixel width heatmap coloured by fiber width.

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    H, W = image_shape
    width_map = np.zeros((H, W), dtype=np.float32)
    for mask, w in zip(masks, widths):
        width_map[mask] = float(w)

    if width_map.max() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    norm = width_map / width_map.max()
    cm = plt.get_cmap(cmap)
    rgb = (cm(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Orientation map
# ---------------------------------------------------------------------------

def draw_orientation_map(
    masks: list[np.ndarray],
    orientations_deg: list[float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """HSV orientation map: hue encodes orientation angle.

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    H, W = image_shape
    orient_map = np.full((H, W), np.nan)
    for mask, angle in zip(masks, orientations_deg):
        orient_map[mask] = angle

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    valid = ~np.isnan(orient_map)
    if valid.any():
        # Hue in [0, 1] for [0, 180) degrees
        hue = orient_map[valid] / 180.0
        for idx, (h, r, c) in enumerate(
            zip(hue, *np.where(valid))
        ):
            r_val, g_val, b_val = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            rgb[r, c] = [int(r_val * 255), int(g_val * 255), int(b_val * 255)]
    return rgb


# ---------------------------------------------------------------------------
# Pore map
# ---------------------------------------------------------------------------

def draw_pore_map(
    combined_mask: np.ndarray,
    cmap: str = "viridis",
) -> np.ndarray:
    """Distance-transform pore map of the background.

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    bg = ~combined_mask
    edt = distance_transform_edt(bg)
    if edt.max() == 0:
        return np.zeros((*combined_mask.shape, 3), dtype=np.uint8)
    norm = edt / edt.max()
    cm = plt.get_cmap(cmap)
    rgb = (cm(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# Porosity map (binary background highlighting)
# ---------------------------------------------------------------------------

def draw_porosity_map(combined_mask: np.ndarray) -> np.ndarray:
    """Simple binary porosity visualisation.

    Returns a (H, W, 3) uint8 image: fiber=dark-green, background=light-grey.
    """
    H, W = combined_mask.shape
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 200  # light grey background
    fiber_color = np.array([30, 120, 50], dtype=np.uint8)
    rgb[combined_mask] = fiber_color
    return rgb


# ---------------------------------------------------------------------------
# Histogram report
# ---------------------------------------------------------------------------

def plot_histogram(
    values: list[float],
    title: str,
    xlabel: str,
    output_path: str | Path | None = None,
    bins: int = 30,
    color: str = "steelblue",
) -> plt.Figure:
    """Generate a clean histogram figure.

    Parameters
    ----------
    values : list of scalar values
    title, xlabel : plot labels
    output_path : if given, save the figure
    bins : number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Rose plot (orientation)
# ---------------------------------------------------------------------------

def plot_rose(
    orientations_deg: list[float],
    title: str = "Fiber Orientation",
    output_path: str | Path | None = None,
    bins: int = 18,
    color: str = "steelblue",
) -> plt.Figure:
    """Polar histogram (rose plot) for fiber orientation distribution.

    Parameters
    ----------
    orientations_deg : list of angles in [0, 180)
    """
    angles_rad = np.deg2rad(np.asarray(orientations_deg))
    # Mirror to fill full circle (orientation has π-periodicity)
    angles_full = np.concatenate([angles_rad, angles_rad + np.pi])

    bin_edges = np.linspace(0, 2 * np.pi, bins * 2 + 1)
    counts, _ = np.histogram(angles_full, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5), dpi=120)
    bars = ax.bar(
        bin_centers, counts, width=bin_width,
        color=color, alpha=0.8, edgecolor="white"
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title, pad=20, fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Full visualisation report
# ---------------------------------------------------------------------------

def save_visualisation_report(
    image: np.ndarray,
    masks: list[np.ndarray],
    centerlines: list[np.ndarray],
    keypoints_list: list[np.ndarray],
    widths: list[float],
    lengths: list[float],
    orientations: list[float],
    curvatures: list[float],
    tortuosities: list[float],
    output_dir: str | Path,
    image_name: str = "image",
) -> None:
    """Write all visualisation artefacts for one image to *output_dir*.

    Saves:
    * ``{name}_overlay.png``
    * ``{name}_centerlines.png``
    * ``{name}_width_map.png``
    * ``{name}_orientation_map.png``
    * ``{name}_pore_map.png``
    * ``{name}_hist_width.png``
    * ``{name}_hist_length.png``
    * ``{name}_hist_orientation.png``
    * ``{name}_rose_orientation.png``
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, W = image.shape[:2]
    combined = np.zeros((H, W), dtype=bool)
    for m in masks:
        combined |= m

    def _save_bgr(arr: np.ndarray, fname: str) -> None:
        bgr = arr if arr.shape[2] == 3 and arr.dtype == np.uint8 else arr
        if arr.shape[2] == 3:
            cv2.imwrite(str(out / fname), bgr)

    # Overlay
    overlay = draw_instance_overlay(image, masks)
    cv2.imwrite(str(out / f"{image_name}_overlay.png"), overlay)

    # Centerlines
    cl_img = draw_centerlines(image, centerlines, keypoints_list)
    cv2.imwrite(str(out / f"{image_name}_centerlines.png"), cl_img)

    # Width map
    wmap = draw_width_map(masks, widths, (H, W))
    bgr = cv2.cvtColor(wmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / f"{image_name}_width_map.png"), bgr)

    # Orientation map
    omap = draw_orientation_map(masks, orientations, (H, W))
    cv2.imwrite(str(out / f"{image_name}_orientation_map.png"), cv2.cvtColor(omap, cv2.COLOR_RGB2BGR))

    # Pore map
    pmap = draw_pore_map(combined)
    cv2.imwrite(str(out / f"{image_name}_pore_map.png"), cv2.cvtColor(pmap, cv2.COLOR_RGB2BGR))

    # Histograms
    if widths:
        fig = plot_histogram(widths, "Fiber Width Distribution", "Width (px)", out / f"{image_name}_hist_width.png")
        plt.close(fig)

    if lengths:
        fig = plot_histogram(lengths, "Fiber Length Distribution", "Length (px)", out / f"{image_name}_hist_length.png", color="coral")
        plt.close(fig)

    if orientations:
        fig = plot_histogram(orientations, "Orientation Distribution", "Angle (°)", out / f"{image_name}_hist_orientation.png", color="mediumseagreen")
        plt.close(fig)
        fig = plot_rose(orientations, "Fiber Orientation Rose", out / f"{image_name}_rose_orientation.png")
        plt.close(fig)
