"""Shared fixtures for the XFeat end-to-end tests."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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


@pytest.fixture
def l76_factory(tmp_path: Path) -> Callable[..., Path]:
    """Write tiny RGB-D files with the real schema and known camera geometry."""

    def make(ids: tuple[int, ...] = (2, 7, 11), chunk_size: int = 2, width: int = 80) -> Path:
        root = tmp_path / "data"
        for subset in ("colmap_rgbd_half_s0", "colmap_rgbd_half_s1"):
            base = root / subset
            scene = base / "scenes/scene_000000"
            (scene / "rgb").mkdir(parents=True)
            (scene / "depth").mkdir()
            meta = {
                "format": "colmap_rgbd_v1",
                "schema_version": 1,
                "camera": {"extrinsics": "opencv_world_to_camera", "intrinsics": "pixel_units"},
                "image": {"height": 64, "width": width, "channels": 3, "dtype": "uint8"},
                "depth": {
                    "height": 64,
                    "width": width,
                    "unit": "millimeters",
                    "dtype": "uint16",
                    "invalid_value": 0,
                    "max_depth_mm": 1300,
                },
                "scene_count": 1,
                "frame_count": len(ids),
            }
            (base / "dataset.json").write_text(json.dumps(meta))
            K = np.tile(np.array([[100, 0, 40], [0, 100, 32], [0, 0, 1]], np.float32), (len(ids), 1, 1))
            poses = np.tile(np.eye(4, dtype=np.float32)[:3], (len(ids), 1, 1))
            poses[:, 0, 3] = np.arange(len(ids)) * 0.01
            np.savez(
                scene / "cameras.npz",
                frame_ids=np.array(ids, np.int64),
                intrinsics=K,
                extrinsics_w2c=poses,
                quality_flags=np.ones(len(ids), bool),
                chunk_ids=np.arange(len(ids), dtype=np.int64) // chunk_size,
            )
            rng = np.random.default_rng(7)
            rgb = rng.integers(0, 256, (64, width, 3), dtype=np.uint8)
            rgb[0, 0] = [11, 22, 33]
            depth = np.full((64, width), 1000, np.uint16)
            depth[0, 0] = 0
            for fid in ids:
                assert cv2.imwrite(str(scene / f"rgb/frame_{fid:06d}.png"), rgb[:, :, ::-1])
                assert cv2.imwrite(str(scene / f"depth/frame_{fid:06d}.png"), depth)
        return root

    return make


@pytest.fixture
def plane_frame() -> Callable[..., Any]:
    """Known z=1 plane; camera translation is world-to-camera in metres."""
    from xfeat_training.data import Frame, FrameKey

    def make(tx: float = 0, tz: float = 0, depth: float = 1) -> Any:
        pose = np.eye(4)
        pose[0, 3], pose[2, 3] = tx, tz
        return Frame(
            key=FrameKey("s0", "scene", 0),
            rgb=torch.zeros(3, 64, 80),
            depth=np.full((64, 80), depth, np.float32),
            intrinsics=np.array([[100, 0, 40], [0, 100, 32], [0, 0, 1]], dtype=np.float64),
            w2c=pose,
            chunk_id=0,
        )

    return make


@pytest.fixture
def amuse_config() -> dict[str, Any]:
    return {
        "name": "amuse",
        "lr": 3e-4,
        "weight_decay": 0.01,
        "beta1": 0.4,
        "rho": 0.3,
        "r": 0.0,
        "weight_lr_power": 2.0,
        "weight_decay_at_y": 0.0,
        "warmup_steps": 3,
        "momentum": 0.95,
        "beta2": 0.999,
        "eps": 1e-10,
    }


@pytest.fixture
def assert_nested_equal() -> Callable[[Any, Any], None]:
    def compare(a: Any, b: Any) -> None:
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                compare(x, y)
        else:
            assert a == b

    return compare
