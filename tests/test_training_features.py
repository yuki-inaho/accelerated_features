"""Feature cache identity, ordinary CPU tensors and frozen BN."""

from __future__ import annotations

import pytest
import torch

from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.data import L76Dataset
from xfeat_training.features import FeatureCache


def test_cache_equals_fresh_and_preserves_bn(l76_factory, tmp_path) -> None:
    data = L76Dataset(l76_factory())
    model = XFeat(device="cpu", top_k=32)
    before = state_hash(model.net.state_dict())
    key = data.frame_keys[0]
    cache = FeatureCache(data, tmp_path / "features", model, top_k=32)
    with pytest.raises(FileNotFoundError):
        cache.get(key)
    cache.build([key])
    saved = cache.get(key)
    fresh = model.detectAndCompute(data.load_frame(key).rgb[None], top_k=32)[0]
    for name in ("keypoints", "descriptors", "scores"):
        assert torch.equal(saved[name], fresh[name])
        assert saved[name].device.type == "cpu" and not saved[name].is_inference()
    assert saved["image_size"] == (80, 64)
    assert state_hash(model.net.state_dict()) == before
    reopened = FeatureCache(data, tmp_path / "features", model, top_k=32)
    assert torch.equal(reopened.get(key)["descriptors"], saved["descriptors"])
    with pytest.raises(ValueError, match="identity"):
        FeatureCache(data, tmp_path / "features", model, top_k=16)
    with torch.no_grad():
        next(model.net.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="identity"):
        FeatureCache(data, tmp_path / "features", model, top_k=32)


def test_cache_zero_points_and_missing_frame_are_distinct(l76_factory, tmp_path) -> None:
    data = L76Dataset(l76_factory())
    model = XFeat(device="cpu", top_k=32, detection_threshold=1.0)
    cache = FeatureCache(data, tmp_path / "empty", model, top_k=32)
    key = data.frame_keys[0]
    cache.build([key])
    with torch.inference_mode():
        features = cache.get(key)
    assert features["keypoints"].shape == (0, 2)
    assert features["descriptors"].shape == (0, 64)
    assert not features["descriptors"].is_inference()
    with pytest.raises(FileNotFoundError):
        cache.get(data.frame_keys[1])
    path = next((tmp_path / "empty").rglob("*.npz"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.get(key)
