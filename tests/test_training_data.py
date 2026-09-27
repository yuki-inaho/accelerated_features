"""L76 schema, IDs, pixel order and metric depth contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xfeat_training.data import FrameKey, L76Dataset


def test_reader_uses_frame_ids_and_metric_depth(l76_factory) -> None:
    data = L76Dataset(l76_factory())
    assert len(data.frame_keys) == 6
    key = FrameKey("colmap_rgbd_half_s0", "scene_000000", 7)
    frame = data.load_frame(key)
    assert frame.key == key and frame.chunk_id == 0
    assert frame.rgb.shape == (3, 64, 80) and frame.rgb.dtype == torch.float32
    torch.testing.assert_close(frame.rgb[:, 0, 0], torch.tensor([11, 22, 33]) / 255)
    assert frame.depth.dtype == np.float32
    assert frame.depth[0, 0] == 0 and frame.depth[10, 10] == 1
    assert frame.w2c.shape == (4, 4) and frame.w2c[0, 3] == pytest.approx(0.01)
    assert data.load_frame(FrameKey("colmap_rgbd_half_s1", "scene_000000", 7)).key != key
    with pytest.raises(ValueError, match="99"):
        data.load_frame(FrameKey(key.subset, key.scene, 99))


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("format", "unknown")])
def test_unknown_schema_has_path(l76_factory, field, value) -> None:
    root = l76_factory()
    path = root / "colmap_rgbd_half_s0/dataset.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"dataset\.json"):
        L76Dataset(root)


@pytest.mark.parametrize("bad", ["duplicate", "length", "focal", "rotation", "nan", "missing", "quality"])
def test_invalid_camera_metadata(l76_factory, bad: str) -> None:
    root = l76_factory()
    path = root / "colmap_rgbd_half_s0/scenes/scene_000000/cameras.npz"
    with np.load(path, allow_pickle=False) as file:
        arrays = dict(file)
    if bad == "duplicate":
        arrays["frame_ids"][1] = arrays["frame_ids"][0]
    elif bad == "length":
        arrays["intrinsics"] = arrays["intrinsics"][:-1]
    elif bad == "focal":
        arrays["intrinsics"][0, 0, 0] = -1
    elif bad == "rotation":
        arrays["extrinsics_w2c"][0, 0, 0] = 2
    elif bad == "nan":
        arrays["intrinsics"][0, 0, 0] = np.nan
    elif bad == "quality":
        arrays["quality_flags"][0] = False
    else:
        del arrays["intrinsics"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=r"cameras\.npz"):
        L76Dataset(root)


def test_missing_frame_is_not_silently_skipped(l76_factory) -> None:
    root = l76_factory()
    data = L76Dataset(root)
    path = root / "colmap_rgbd_half_s0/scenes/scene_000000/depth/frame_000007.png"
    path.unlink()
    with pytest.raises(ValueError, match="000007"):
        data.load_frame(FrameKey("colmap_rgbd_half_s0", "scene_000000", 7))


def test_manifest_hash_is_portable_and_changes_with_data(l76_factory, tmp_path: Path) -> None:
    import shutil

    root = l76_factory()
    original = L76Dataset(root).fingerprint()
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    assert original == L76Dataset(copy).fingerprint()
    path = copy / "colmap_rgbd_half_s0/scenes/scene_000000/rgb/frame_000007.png"
    path.write_bytes(path.read_bytes() + b"changed")
    assert original != L76Dataset(copy).fingerprint()
