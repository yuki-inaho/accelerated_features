"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

Small helpers shared by the inference modules.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested device, or CUDA when available, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_weights(
    weights: str | os.PathLike[str] | Mapping[str, Tensor],
    device: torch.device,
) -> Mapping[str, Tensor]:
    """Load a checkpoint from disk, or pass an in-memory state dict through."""
    if isinstance(weights, Mapping):
        return weights
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"XFeat weights not found: '{path}'.")
    return torch.load(path, map_location=device, weights_only=True)
