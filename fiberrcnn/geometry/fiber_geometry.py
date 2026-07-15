"""
Fiber Geometry Pipeline
=======================
Converts polygon annotations to rich geometric descriptors:
mask → skeleton → graph → centerline → 40 keypoints →
width / length / curvature / orientation / tortuosity.

All public functions are pure (no side-effects) and accept / return
plain numpy arrays so they can be called from both the data converter
and post-processing without a Detectron2 dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.draw import polygon as sk_polygon
from skimage.morphology import skeletonize, binary_dilation, disk


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FiberGeometry:
    """All geometric descriptors for a single fiber instance."""

    mask: np.ndarray                      # H×W bool
    bbox: list[float]                     # [x, y, w, h]  COCO style
    segmentation: list[list[float]]       # [[x0,y0,x1,y1,...]]
    centerline: np.ndarray                # N×2 ordered (col, row) points
    keypoints: np.ndarray                 # 40×3  (x, y, visibility=2)
    fiber_width: float
    fiber_length: float
    fiber_curvature: float
    fiber_orientation: float              # degrees [0, 180)
    fiber_tortuosity: float
    area: int = 0
    skeleton: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


# ---------------------------------------------------------------------------
# Step 1 — Polygon → Binary Mask
# ---------------------------------------------------------------------------

def polygon_to_mask(
    points: list[list[float]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Rasterise a LabelMe polygon to a binary mask.

    Parameters
    ----------
    points:
        List of [x, y] pairs (LabelMe format).
    image_height, image_width:
        Canvas size.

    Returns
    -------
    mask : ndarray of shape (H, W), dtype bool
    """
    pts = np.asarray(points, dtype=np.float64)
    rows = pts[:, 1]
    cols = pts[:, 0]
    rr, cc = sk_polygon(rows, cols, shape=(image_height, image_width))
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[rr, cc] = True
    return mask


# ---------------------------------------------------------------------------
# Step 2 — Skeletonisation
# ---------------------------------------------------------------------------

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return medial-axis skeleton as a bool array of the same shape as *mask*."""
    return skeletonize(mask)


# ---------------------------------------------------------------------------
# Step 3 — Graph extraction from skeleton
# ---------------------------------------------------------------------------

def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert a skeleton image to an undirected NetworkX graph.

    Each skeleton pixel is a node; edges connect 8-connected neighbours.
    Node coordinates are stored as ``(row, col)``.
    """
    G: nx.Graph = nx.Graph()
    ys, xs = np.where(skeleton)
    coords = list(zip(ys.tolist(), xs.tolist()))
    G.add_nodes_from(coords)

    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in G:
                    dist = math.sqrt(dr * dr + dc * dc)
                    G.add_edge((r, c), nb, weight=dist)
    return G


# ---------------------------------------------------------------------------
# Step 4 — Longest path (diameter) in skeleton graph
# ---------------------------------------------------------------------------

def longest_path(G: nx.Graph) -> list[tuple[int, int]]:
    """Find the longest shortest-path (pseudo-diameter) in *G*.

    Falls back to a degree-based heuristic for large graphs to keep
    runtime manageable.

    Returns
    -------
    path : list of (row, col) tuples
    """
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return list(G.nodes)

    # Use endpoints (degree-1 nodes) as candidates
    endpoints = [n for n, d in G.degree() if d == 1]
    if len(endpoints) < 2:
        endpoints = list(G.nodes)

    # Limit search to avoid O(N²) on large fibres
    max_candidates = 10
    endpoints = endpoints[:max_candidates]

    best_path: list[tuple[int, int]] = []
    best_len = 0.0

    for src in endpoints:
        try:
            lengths, paths = nx.single_source_dijkstra(
                G, src, weight="weight", cutoff=None
            )
        except nx.NetworkXError:
            continue
        for tgt, path_len in lengths.items():
            if path_len > best_len:
                best_len = path_len
                best_path = paths[tgt]

    return best_path


# ---------------------------------------------------------------------------
# Step 5 & 6 — Centerline reconstruction + ordering
# ---------------------------------------------------------------------------

def extract_centerline(
    mask: np.ndarray,
    min_branch_length: int = 5,
) -> np.ndarray:
    """Return ordered centerline points as an array of shape (N, 2) in
    (col, row) / (x, y) convention.

    Pipeline:
        mask → skeleton → graph → longest-path → ordered (col, row) array
    """
    skel = skeletonize_mask(mask)
    if skel.sum() == 0:
        # Degenerate: return bounding-box centre
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.zeros((2, 2), dtype=float)
        cy = float(ys.mean())
        cx = float(xs.mean())
        return np.array([[cx, cy], [cx, cy]], dtype=float)

    G = skeleton_to_graph(skel)

    # Remove short branches
    _prune_branches(G, min_branch_length)

    path_rc = longest_path(G)
    if len(path_rc) < 2:
        ys, xs = np.where(skel)
        path_rc = list(zip(ys.tolist(), xs.tolist()))

    # Convert (row, col) → (col, row) i.e. (x, y)
    centerline_xy = np.array([[c, r] for r, c in path_rc], dtype=float)

    # Canonicalize direction so keypoint index 0 is stable across samples.
    # We choose lexicographic order in image coordinates: left-to-right,
    # then top-to-bottom when x is tied.
    if len(centerline_xy) >= 2:
        x0, y0 = centerline_xy[0]
        x1, y1 = centerline_xy[-1]
        if (x0 > x1) or (x0 == x1 and y0 > y1):
            centerline_xy = centerline_xy[::-1].copy()

    return centerline_xy


def _prune_branches(G: nx.Graph, min_length: int) -> None:
    """Remove short degree-1 branches from *G* in-place."""
    changed = True
    while changed:
        changed = False
        leaves = [n for n, d in G.degree() if d == 1]
        for leaf in leaves:
            # Walk along branch until we hit a junction or run out
            branch = [leaf]
            prev = None
            cur = leaf
            while True:
                nbrs = [n for n in G.neighbors(cur) if n != prev]
                if len(nbrs) != 1:
                    break
                prev = cur
                cur = nbrs[0]
                branch.append(cur)
                if G.degree(cur) > 2:
                    break
            if len(branch) < min_length and G.degree(branch[-1]) > 1:
                G.remove_nodes_from(branch[:-1])
                changed = True


# ---------------------------------------------------------------------------
# Step 7 — Uniform resampling
# ---------------------------------------------------------------------------

def resample_centerline(
    centerline: np.ndarray,
    n_points: int = 40,
) -> np.ndarray:
    """Resample *centerline* to exactly *n_points* uniformly spaced points.

    Parameters
    ----------
    centerline : (N, 2) array of (x, y)
    n_points : desired number of output points

    Returns
    -------
    resampled : (n_points, 2) array
    """
    if len(centerline) < 2:
        # Pad to n_points with the single point repeated
        pt = centerline[0] if len(centerline) > 0 else np.zeros(2)
        return np.tile(pt, (n_points, 1))

    # Arc-length parameterisation
    diffs = np.diff(centerline, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum_len[-1]

    if total < 1e-6:
        return np.tile(centerline[0], (n_points, 1))

    # Try spline interpolation; fall back to linear if not enough points
    t_uniform = np.linspace(0.0, total, n_points)

    if len(centerline) >= 4:
        try:
            k = min(3, len(centerline) - 1)
            tck, _ = splprep(
                [centerline[:, 0], centerline[:, 1]],
                u=cum_len / total,
                s=0,
                k=k,
            )
            x_new, y_new = splev(t_uniform / total, tck)
            return np.stack([x_new, y_new], axis=1)
        except Exception:
            pass

    # Linear fallback
    x_new = np.interp(t_uniform, cum_len, centerline[:, 0])
    y_new = np.interp(t_uniform, cum_len, centerline[:, 1])
    return np.stack([x_new, y_new], axis=1)


# ---------------------------------------------------------------------------
# Step 8 — 40 Keypoints
# ---------------------------------------------------------------------------

def generate_keypoints(resampled: np.ndarray) -> np.ndarray:
    """Convert resampled centerline points to COCO-style keypoints.

    Returns
    -------
    keypoints : (40, 3) array — columns are (x, y, visibility)
        visibility = 2 means labeled and visible.
    """
    assert resampled.shape == (40, 2), (
        f"Expected (40, 2) resampled array, got {resampled.shape}"
    )
    vis = np.full((40, 1), 2.0)
    return np.concatenate([resampled, vis], axis=1)


# ---------------------------------------------------------------------------
# Step 9 — Width estimation via distance transform
# ---------------------------------------------------------------------------

def estimate_width(
    mask: np.ndarray,
    centerline: np.ndarray,
) -> float:
    """Estimate mean fiber width using the distance transform.

    At each centerline pixel the distance transform gives the distance to
    the nearest background pixel, which equals the local half-width.
    We report 2 × mean(half-widths) as the diameter.

    Parameters
    ----------
    mask : (H, W) bool
    centerline : (N, 2) array of (x, y)

    Returns
    -------
    width : float, in pixels
    """
    dt = distance_transform_edt(mask)

    # Clamp coordinates to valid image range
    H, W = mask.shape
    xs = np.clip(np.round(centerline[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(centerline[:, 1]).astype(int), 0, H - 1)
    half_widths = dt[ys, xs]
    mean_hw = float(half_widths[half_widths > 0].mean()) if half_widths.any() else 1.0
    return 2.0 * mean_hw


# ---------------------------------------------------------------------------
# Step 10 — Fiber length
# ---------------------------------------------------------------------------

def compute_length(centerline: np.ndarray) -> float:
    """Arc-length of *centerline* in pixels."""
    if len(centerline) < 2:
        return 0.0
    diffs = np.diff(centerline, axis=0)
    return float(np.sqrt((diffs ** 2).sum(axis=1)).sum())


# ---------------------------------------------------------------------------
# Step 11 — Curvature
# ---------------------------------------------------------------------------

def compute_curvature(centerline: np.ndarray) -> float:
    """Mean absolute curvature of *centerline* (1/pixel).

    Uses the signed curvature formula for a parametric curve.
    """
    if len(centerline) < 3:
        return 0.0
    # First and second differences
    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.abs(dx * ddy - dy * ddx) / np.where(denom > 1e-10, denom, np.inf)
    return float(np.nanmean(kappa))


# ---------------------------------------------------------------------------
# Step 12 — Orientation
# ---------------------------------------------------------------------------

def compute_orientation(centerline: np.ndarray) -> float:
    """Global orientation of the fiber in degrees [0, 180).

    Fits the PCA principal axis to the centerline points.
    0° = horizontal, 90° = vertical.
    """
    if len(centerline) < 2:
        return 0.0
    pts = centerline - centerline.mean(axis=0)
    cov = pts.T @ pts
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]  # (dx, dy)
    angle = math.degrees(math.atan2(principal[1], principal[0]))
    # Map to [0, 180)
    angle = angle % 180.0
    return float(angle)


# ---------------------------------------------------------------------------
# Step 13 — Tortuosity
# ---------------------------------------------------------------------------

def compute_tortuosity(centerline: np.ndarray) -> float:
    """Tortuosity = arc-length / end-to-end straight-line distance.

    Values close to 1.0 indicate a straight fibre; higher values indicate
    a more sinuous path.
    """
    arc = compute_length(centerline)
    if arc < 1e-6:
        return 1.0
    end_to_end = float(np.linalg.norm(centerline[-1] - centerline[0]))
    if end_to_end < 1e-6:
        return 1.0
    return float(arc / end_to_end)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def compute_fiber_geometry(
    points: list[list[float]],
    image_height: int,
    image_width: int,
    n_keypoints: int = 40,
) -> FiberGeometry:
    """Full geometry pipeline for one LabelMe polygon.

    Parameters
    ----------
    points : [[x, y], ...] polygon vertices
    image_height, image_width : canvas dimensions
    n_keypoints : number of ordered keypoints (default 40)

    Returns
    -------
    FiberGeometry dataclass
    """
    # 1. Mask
    mask = polygon_to_mask(points, image_height, image_width)

    # Bounding box (COCO: x_min, y_min, width, height)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        x_min = y_min = w = h = 0.0
    else:
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        w = x_max - x_min
        h = y_max - y_min
    bbox = [x_min, y_min, w, h]

    # Segmentation (flattened polygon)
    flat_pts: list[float] = []
    for px, py in points:
        flat_pts += [float(px), float(py)]
    segmentation = [flat_pts]

    # 2–6. Centerline
    centerline = extract_centerline(mask)

    # 7. Resample
    resampled = resample_centerline(centerline, n_keypoints)

    # 8. Keypoints
    keypoints = generate_keypoints(resampled)

    # 9. Width
    fiber_width = estimate_width(mask, centerline)

    # 10. Length
    fiber_length = compute_length(centerline)

    # 11. Curvature
    fiber_curvature = compute_curvature(centerline)

    # 12. Orientation
    fiber_orientation = compute_orientation(centerline)

    # 13. Tortuosity
    fiber_tortuosity = compute_tortuosity(centerline)

    skel = skeletonize_mask(mask)
    skel_pts = np.column_stack(np.where(skel)) if skel.any() else np.zeros((0, 2))

    return FiberGeometry(
        mask=mask,
        bbox=bbox,
        segmentation=segmentation,
        centerline=centerline,
        keypoints=keypoints,
        fiber_width=fiber_width,
        fiber_length=fiber_length,
        fiber_curvature=fiber_curvature,
        fiber_orientation=fiber_orientation,
        fiber_tortuosity=fiber_tortuosity,
        area=int(mask.sum()),
        skeleton=skel_pts,
    )
