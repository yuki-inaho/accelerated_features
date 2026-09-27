"""Seeded photometric augmentation and known source-to-target homographies."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def photometric(image: Tensor, generator: torch.Generator | None = None) -> Tensor:
    draws = torch.rand(3, generator=generator)
    brightness = (draws[0] * 2 - 1) * 0.15
    contrast = 1 + (draws[1] * 2 - 1) * 0.15
    noise = torch.randn(image.shape, generator=generator) * (draws[2] * 0.02)
    return ((image - 0.5) * float(contrast) + 0.5 + float(brightness) + noise.to(image.device)).clamp(0, 1)


def sample_homography(height: int, width: int, generator: torch.Generator | None = None) -> Tensor:
    values = torch.rand(6, generator=generator) * 2 - 1
    angle, scale = float(values[0]) * math.pi / 12, 1 + float(values[1]) * 0.15
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    center = torch.tensor([[1.0, 0.0, -(width - 1) / 2], [0.0, 1.0, -(height - 1) / 2], [0.0, 0.0, 1.0]])
    affine = torch.tensor([[c, -s, float(values[2]) * 16], [s, c, float(values[3]) * 16], [0.0, 0.0, 1.0]])
    projective = torch.eye(3)
    projective[2, :2] = values[4:] * 1e-4
    result = torch.linalg.inv(center) @ affine @ projective @ center
    return result / result[2, 2]


def transform_points(points: Tensor, homography: Tensor) -> Tensor:
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    transformed = homogeneous @ homography.to(device=points.device, dtype=points.dtype).T
    return transformed[..., :2] / transformed[..., 2:3]


def warp_image(image: Tensor, homography: Tensor) -> tuple[Tensor, Tensor]:
    """Warp source image into target pixels and return a mask of valid source support."""
    if image.ndim != 4 or homography.shape != (3, 3) or not torch.isfinite(homography).all():
        raise ValueError("warp_image expects BCHW image and a finite 3x3 homography")
    batch, _, height, width = image.shape
    if height < 2 or width < 2:
        raise ValueError("Warp dimensions must be at least 2")
    y, x = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=torch.float32),
        torch.arange(width, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    target = torch.stack((x, y), dim=-1)
    source = transform_points(target, torch.linalg.inv(homography.to(image.device)))
    valid = torch.isfinite(source).all(-1) & (source[..., 0] >= 0) & (source[..., 0] <= width - 1)
    valid &= (source[..., 1] >= 0) & (source[..., 1] <= height - 1)
    grid = source / torch.tensor([width - 1, height - 1], device=image.device) * 2 - 1
    grid = torch.where(valid[..., None], grid, 2.0).unsqueeze(0).expand(batch, -1, -1, -1)
    warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped, valid[None, None].expand(batch, 1, -1, -1)
