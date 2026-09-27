"""Independent synthetic views with explicit support and a known relative warp."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from xfeat_training.augment import warp_image


def sample_transform(height: int, width: int, config: Mapping[str, Any], generator: torch.Generator | None) -> Tensor:
    v = torch.rand(9, generator=generator)
    extent = config["rotation_small_degrees"] if v[0] < config["rotation_small_probability"] else 180.0
    angle = float(v[1] * 2 - 1) * math.radians(extent)
    lo, hi = config["scale"]
    scale = math.exp(math.log(lo) + float(v[2]) * math.log(hi / lo))
    cosine, sine = math.cos(angle) * scale, math.sin(angle) * scale
    center = torch.tensor([[1.0, 0.0, -(width - 1) / 2], [0.0, 1.0, -(height - 1) / 2], [0.0, 0.0, 1.0]])
    affine = torch.tensor(
        [
            [cosine, -sine, (float(v[3]) * 2 - 1) * width * config["translation"] / 2],
            [sine, cosine, (float(v[4]) * 2 - 1) * height * config["translation"] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    shear = torch.eye(3)
    shear[0, 1] = (v[5] * 2 - 1) * config["shear"]
    shear[1, 0] = (v[6] * 2 - 1) * config["shear"]
    shear[2, :2] = (v[7:] * 2 - 1) * config["projective"]
    transform = torch.linalg.inv(center) @ affine @ shear @ center
    return transform / transform[2, 2]


def _photometric(image: Tensor, config: Mapping[str, Any], generator: torch.Generator | None) -> Tensor:
    v = torch.rand(4, generator=generator)
    lo, hi = config["gamma"]
    gamma = lo + float(v[0]) * (hi - lo)
    image = image.clamp(0, 1).pow(gamma) + (float(v[1]) * 2 - 1) * config["brightness"]
    if v[2] < config["blur_probability"]:
        kernel = image.new_tensor([1.0, 2.0, 1.0])
        kernel = (kernel[:, None] * kernel[None]) / 16
        channels = image.shape[1]
        image = F.conv2d(
            F.pad(image, (1,) * 4, mode="replicate"), kernel[None, None].expand(channels, 1, 3, 3), groups=channels
        )
    noise = torch.randn(image.shape, generator=generator).to(image.device)
    return (image + noise * float(v[3]) * config["noise"]).clamp(0, 1)


def synthetic_views(
    image: Tensor, config: Mapping[str, Any], generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    height, width = image.shape[-2:]
    h0 = sample_transform(height, width, config, generator).to(image.device)
    h1 = sample_transform(height, width, config, generator).to(image.device)
    image0, valid0 = warp_image(image, h0)
    image1, valid1 = warp_image(image, h1)
    relative = h1 @ torch.linalg.inv(h0)
    valid0_in1, support = warp_image(valid0.float(), relative)
    overlap = float(((valid0_in1 > 0.999) & valid1 & support).float().mean())
    return (
        _photometric(image0, config, generator),
        _photometric(image1, config, generator),
        valid0,
        valid1,
        relative,
        overlap,
    )
