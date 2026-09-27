"""Depth reprojection and conservative mutual matching labels (index/-1/-2)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xfeat_training.data import Array, Frame


@dataclass
class Projection:
    xy: Array
    z: Array
    valid: Array


@dataclass
class MatchLabels:
    matches0: Array
    matches1: Array
    reasons0: Array
    reasons1: Array


def sample_depth(depth: Array, points: Array, *, interior: bool = False) -> tuple[Array, Array]:
    """Nearest sampling; optional valid, smooth 5x5 support, including a 2px margin."""
    if depth.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("depth must be HxW and points Nx2")
    finite = np.isfinite(points).all(axis=1)
    rounded = np.rint(np.where(np.isfinite(points), points, 0)).astype(np.int64)
    x, y = rounded.T
    h, w = depth.shape
    inside = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    values = np.zeros(len(points), dtype=np.float64)
    values[inside] = depth[y[inside], x[inside]]
    valid = inside & np.isfinite(values) & (values > 0)
    if interior:
        raw_valid = np.isfinite(depth) & (depth > 0)
        clean = np.where(raw_valid, depth, 0).astype(np.float32)
        kernel = np.ones((5, 5), np.uint8)
        full = cv2.erode(raw_valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        low = cv2.erode(clean, kernel)
        high = cv2.dilate(clean, kernel)
        good = full.astype(bool) & ((high - low) <= 0.05 * clean)
        support = np.zeros(len(points), dtype=bool)
        support[inside] = good[y[inside], x[inside]]
        valid &= support & (points[:, 0] >= 2) & (points[:, 0] <= w - 3)
        valid &= (points[:, 1] >= 2) & (points[:, 1] <= h - 3)
    return values, valid


def project_points(points: Array, source: Frame, target: Frame) -> Projection:
    """Project source pixels using T_w2c_target @ inverse(T_w2c_source)."""
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("nonfinite input keypoints")
    depth, valid = sample_depth(source.depth, points)
    rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(source.intrinsics).T
    xyz = rays * depth[:, None]
    transform = target.w2c @ np.linalg.inv(source.w2c)
    projected = xyz @ transform[:3, :3].T + transform[:3, 3]
    pixels = projected @ target.intrinsics.T
    xy = np.full((len(points), 2), np.nan, dtype=np.float64)
    np.divide(pixels[:, :2], pixels[:, 2:3], out=xy, where=pixels[:, 2:3] != 0)
    return Projection(xy, projected[:, 2], valid & np.isfinite(projected).all(axis=1))


def relative_motion(w2c0: Array, w2c1: Array) -> tuple[float, float]:
    rotation = w2c1[:3, :3] @ w2c0[:3, :3].T
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    center0 = -w2c0[:3, :3].T @ w2c0[:3, 3]
    center1 = -w2c1[:3, :3].T @ w2c1[:3, 3]
    return angle, float(np.linalg.norm(center1 - center0))


def _directional_overlap(source: Frame, target: Frame, stride: int, depth_tolerance: float) -> float:
    y, x = np.mgrid[0 : source.depth.shape[0] : stride, 0 : source.depth.shape[1] : stride]
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float64)
    projected = project_points(points, source, target)
    count = int(projected.valid.sum())
    if count == 0:
        return 0.0
    target_depth, target_valid = sample_depth(target.depth, projected.xy)
    good = projected.valid & target_valid & (projected.z > 0)
    good &= np.abs(projected.z - target_depth) <= depth_tolerance * target_depth
    return float(good.sum() / count)


def overlap(source: Frame, target: Frame, *, stride: int = 8, depth_tolerance: float = 0.03) -> float:
    """Bidirectional depth-consistent overlap; intentionally no GT edge mask."""
    if stride < 1 or depth_tolerance <= 0:
        raise ValueError("stride and depth_tolerance must be positive")
    return 0.5 * (
        _directional_overlap(source, target, stride, depth_tolerance)
        + _directional_overlap(target, source, stride, depth_tolerance)
    )


def _visibility(points: Array, source: Frame, target: Frame) -> tuple[Projection, Array, Array, Array]:
    projection = project_points(points, source, target)
    _, source_good = sample_depth(source.depth, points, interior=True)
    labels = np.full(len(points), -2, dtype=np.int64)
    reasons = np.full(len(points), "invalid_source_depth", dtype="<U32")
    h, w = target.depth.shape
    xy = projection.xy
    valid = source_good & projection.valid
    outside = valid & ((projection.z <= 0) | (xy[:, 0] < 0) | (xy[:, 0] >= w) | (xy[:, 1] < 0) | (xy[:, 1] >= h))
    labels[outside], reasons[outside] = -1, "outside"
    target_depth, target_good = sample_depth(target.depth, xy, interior=True)
    in_image = valid & ~outside
    reasons[in_image] = "target_depth_unknown"
    observed = in_image & target_good
    reasons[observed] = "depth_inconsistent"
    occluded = observed & (projection.z > target_depth * 1.03)
    labels[occluded], reasons[occluded] = -1, "occluded"
    visible = observed & (np.abs(projection.z - target_depth) <= 0.03 * target_depth)
    reasons[visible] = "ambiguous"
    return projection, visible, labels, reasons


def matching_labels(points0: Array, points1: Array, frame0: Frame, frame1: Frame) -> MatchLabels:
    """Mutual symmetric <=3px positives, visible >=5px negatives, unknown=-2."""
    points0, points1 = np.asarray(points0, dtype=np.float64), np.asarray(points1, dtype=np.float64)
    p01, visible0, labels0, reasons0 = _visibility(points0, frame0, frame1)
    p10, visible1, labels1, reasons1 = _visibility(points1, frame1, frame0)
    if len(points0) == 0 or len(points1) == 0:
        labels0[visible0], reasons0[visible0] = -1, "far"
        labels1[visible1], reasons1[visible1] = -1, "far"
        return MatchLabels(labels0, labels1, reasons0, reasons1)
    distances01 = np.linalg.norm(p01.xy[:, None] - points1[None], axis=-1)
    distances10 = np.linalg.norm(p10.xy[:, None] - points0[None], axis=-1)
    # Invalid projections stay unusable; never remove real target tokens before nearest search.
    distances01[~np.isfinite(distances01)] = np.inf
    distances10[~np.isfinite(distances10)] = np.inf
    nearest1 = distances01.argmin(axis=1)
    nearest0 = distances10.argmin(axis=1)
    idx0, idx1 = np.arange(len(points0)), np.arange(len(points1))
    far0 = visible0 & (distances01[idx0, nearest1] >= 5)
    far1 = visible1 & (distances10[idx1, nearest0] >= 5)
    labels0[far0], reasons0[far0] = -1, "far"
    labels1[far1], reasons1[far1] = -1, "far"
    symmetric = np.maximum(distances01, distances10.T)
    symmetric1, symmetric0 = symmetric.argmin(axis=1), symmetric.argmin(axis=0)
    positive = visible0 & visible1[symmetric1] & (symmetric0[symmetric1] == idx0)
    positive &= symmetric[idx0, symmetric1] <= 3
    matched0 = idx0[positive]
    matched1 = symmetric1[positive]
    labels0[matched0], labels1[matched1] = matched1, matched0
    reasons0[matched0], reasons1[matched1] = "positive", "positive"
    return MatchLabels(labels0, labels1, reasons0, reasons1)
