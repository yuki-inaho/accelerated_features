"""Shared fixtures for the XFeat end-to-end tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from jaxtyping import UInt8
from numpy import ndarray as NDArray

from inference import load_image
from modules.xfeat import XFeat

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Root folder of the repository under test."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def weights_file(repo_root: Path) -> Path:
    """Pretrained XFeat checkpoint shipped with the repository."""
    weights = repo_root / "weights" / "xfeat.pt"
    assert weights.is_file(), f"missing pretrained weights: {weights}"
    return weights


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Device under test: XFEAT_TEST_DEVICE, else CUDA when available, else CPU."""
    requested = os.environ.get("XFEAT_TEST_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def xfeat(weights_file: Path, device: torch.device) -> XFeat:
    """XFeat initialized with the pretrained weights on the test device."""
    return XFeat(weights=weights_file, device=device)


@pytest.fixture(scope="session")
def ref_image(repo_root: Path) -> UInt8[NDArray, "H W 3"]:
    """Reference image of the bundled MegaDepth sample pair (BGR uint8)."""
    return load_image(repo_root / "assets" / "ref.png")


@pytest.fixture(scope="session")
def tgt_image(repo_root: Path) -> UInt8[NDArray, "H W 3"]:
    """Target image of the bundled MegaDepth sample pair (BGR uint8)."""
    return load_image(repo_root / "assets" / "tgt.png")
