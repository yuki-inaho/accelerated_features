"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

Small helpers shared by the inference modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
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


def state_hash(state: Mapping[str, Tensor]) -> str:
    """Hash tensor contents independently of torch.save storage and file metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        if sys.byteorder != "little":
            array = array.reshape(-1, value.element_size())[:, ::-1].copy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_state_dict(
    state: Mapping[str, Tensor], expected: Mapping[str, Tensor], *, allow_missing: frozenset[str] = frozenset()
) -> None:
    """Reject missing, unexpected and incompatible tensors before loading any weights."""
    missing = expected.keys() - state.keys() - allow_missing
    unexpected = state.keys() - expected.keys()
    bad_shapes = [
        key
        for key in state.keys() & expected.keys()
        if not isinstance(state[key], Tensor) or state[key].shape != expected[key].shape
    ]
    if missing or unexpected or bad_shapes:
        raise ValueError(
            f"Invalid weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"shape_mismatch={sorted(bad_shapes)}"
        )
