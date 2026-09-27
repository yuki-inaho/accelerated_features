"""Regressions required before RGB-D training and evaluation."""

from __future__ import annotations

import copy
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import TypeCheckError
from kornia.feature.lightglue import LightGlue

from modules.lighterglue import LighterGlue
from modules.typecheck import SparseFeaturesWithSize
from modules.xfeat import XFeat

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_lightglue_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing config-isolation test must not contaminate other tests."""
    monkeypatch.setattr(LightGlue, "default_conf", copy.deepcopy(LightGlue.default_conf))


@pytest.fixture(scope="module")
def cpu_xfeat() -> XFeat:
    return XFeat(device="cpu", top_k=64)


@pytest.mark.parametrize("method", ["match_xfeat", "match_xfeat_star"])
def test_different_image_sizes(cpu_xfeat: XFeat, method: str) -> None:
    rng = np.random.default_rng(73)
    image0 = rng.integers(0, 256, (128, 160, 3), dtype=np.uint8)
    image1 = rng.integers(0, 256, (160, 192, 3), dtype=np.uint8)
    points0, points1 = getattr(cpu_xfeat, method)(image0, image1, top_k=64)
    assert points0.shape == points1.shape
    assert points0.ndim == 2 and points0.shape[1] == 2
    assert np.isfinite(points0).all() and np.isfinite(points1).all()


def test_image_and_descriptor_types_still_checked(cpu_xfeat: XFeat) -> None:
    with pytest.raises(TypeCheckError):
        cpu_xfeat.match_xfeat(torch.ones(3, 64, 64), torch.rand(1, 3, 64, 64))
    with pytest.raises(TypeCheckError):
        cpu_xfeat.match_xfeat(torch.ones(1, 3, 64, 64, dtype=torch.int32), torch.rand(1, 3, 64, 64))
    with pytest.raises(TypeCheckError):
        cpu_xfeat.match(torch.ones(4, 63), torch.ones(4, 64))


@pytest.mark.parametrize("module", ["megadepth1500", "scannet1500"])
@pytest.mark.parametrize("legacy_numpy", [False, True])
def test_pose_auc_numpy_compat(module: str, legacy_numpy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator = importlib.import_module(f"modules.eval.{module}")
    trapezoid = np.trapezoid
    if legacy_numpy:
        monkeypatch.setattr(np, "trapz", trapezoid, raising=False)
        monkeypatch.delattr(np, "trapezoid")
    else:
        monkeypatch.delattr(np, "trapz", raising=False)
    # At each threshold the endpoint recall is the last value strictly below it.
    errors, thresholds = [0.0, 1.0, 3.0, np.inf], [1.0, 3.0]
    if module == "megadepth1500":
        result = list(evaluator.error_auc(errors, thresholds).values())
    else:
        result = evaluator.pose_auc(errors, thresholds)
    assert result == pytest.approx([0.25, 1.375 / 3])


def test_scannet_selected_factory_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from modules.eval import scannet1500

    called: list[str] = []
    sentinel = object()
    monkeypatch.setattr(scannet1500, "get_xfeat", lambda: (called.append("xfeat"), sentinel)[1])

    def forbidden() -> None:
        pytest.fail("Unselected model was initialized")

    monkeypatch.setattr(scannet1500, "get_alike", forbidden)
    monkeypatch.setattr(scannet1500, "get_xfeat_star", forbidden)
    builder = getattr(scannet1500, "build_matchers", None)
    assert callable(builder), "ScanNet needs an explicit lazy matcher factory"
    assert builder(["xfeat"]) == {"xfeat": sentinel}
    assert called == ["xfeat"]


def test_scannet_matcher_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from modules.eval import scannet1500

    monkeypatch.setattr(sys, "argv", ["scannet", "--scannet_path", "unused", "--matcher", "xfeat"])
    assert scannet1500.parse().matcher == ["xfeat"]
    monkeypatch.setattr(sys, "argv", ["scannet", "--scannet_path", "unused", "--matcher", "unknown"])
    with pytest.raises(SystemExit) as exc:
        scannet1500.parse()
    assert exc.value.code == 2


@pytest.mark.parametrize("shape", [(192, 256), (384, 512), (288, 384), (608, 800)])
def test_tps_grid_resolution(shape: tuple[int, int]) -> None:
    # Importing the legacy augmentation module seeds global RNGs. Restore them.
    state, np_state = torch.get_rng_state(), np.random.get_state()  # noqa: NPY002 -- preserve legacy global RNG
    try:
        from modules.dataset.augmentation import generateRandomTPS

        source, weights, affine = generateRandomTPS(shape, grid=(8, 6), prob=1.0)
        assert source.shape == (1, 63, 2)
        assert weights.shape == (1, 63, 2)
        assert affine.shape == (1, 3, 2)
        assert torch.equal(source[0, 0], torch.tensor([-1.0, -1.0]))
        assert torch.equal(source[0, -1], torch.tensor([1.0, 1.0]))
        assert torch.isfinite(weights).all() and torch.isfinite(affine).all()
    finally:
        torch.set_rng_state(state)
        np.random.set_state(np_state)  # noqa: NPY002 -- restore legacy global RNG


def _run_demo_script(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )


@pytest.mark.parametrize("case", ["eof", "blocking", "window", "method", "loop", "quit"])
def test_demo_releases_camera_and_threads(case: str) -> None:
    code = """
import argparse
import threading
import time
from unittest.mock import patch
import numpy as np
import cv2
import realtime_demo as demo

class Camera:
    released = 0
    calls = 0
    def read(self):
        self.calls += 1
        if CASE == 'blocking' and self.calls > 1:
            threading.Event().wait(30)
        if CASE == 'eof' and self.calls > 1:
            return False, None
        return True, np.zeros((192,256,3),dtype=np.uint8)
    def release(self): self.released += 1
    def set(self, *args): pass
    def isOpened(self): return True

cap = Camera()
if CASE in ('eof','blocking'):
    grab = demo.FrameGrabber(cap)
    grab.start()
    time.sleep(.05)
    if CASE == 'eof':
        grab.join(.5)
        assert not grab.is_alive(), 'EOF must terminate the grabber'
    grab.stop()
    grab.stop()
    assert grab.daemon, 'Blocking camera reads must not hold the process alive'
    assert cap.released == 1, cap.released
else:
    args = argparse.Namespace(width=256,height=192,cam=0,method='ORB',max_kpts=32)
    def fail(*args, **kwargs): raise RuntimeError('forced failure')
    patches = [patch.object(cv2,'VideoCapture',return_value=cap),
        patch.object(cv2,'namedWindow',side_effect=fail if CASE=='window' else None),
        patch.object(cv2,'resizeWindow'),patch.object(cv2,'setMouseCallback'),
        patch.object(cv2,'destroyAllWindows'),patch.object(cv2,'waitKey',return_value=ord('q'))]
    if CASE=='method': patches.append(patch.object(demo,'init_method',side_effect=fail))
    for p in patches: p.start()
    try:
        obj = demo.MatchingDemo(args)
        obj.process = (lambda: None) if CASE=='quit' else fail
        obj.main_loop()
    except RuntimeError as e:
        assert str(e)=='forced failure'
    else:
        assert CASE=='quit', 'Expected injected failure'
        obj.cleanup()
    assert cap.released == 1, cap.released
    assert all(t.daemon or t is threading.main_thread() for t in threading.enumerate())
print('released exactly once')
"""
    result = _run_demo_script(f"CASE = {case!r}\n" + code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "released exactly once" in result.stdout


def test_demo_canvas_follows_requested_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from realtime_demo import MatchingDemo

    demo = MatchingDemo.__new__(MatchingDemo)
    demo.width, demo.height = 256, 192
    demo.ref_frame = np.zeros((192, 256, 3), np.uint8)
    demo.current_frame = demo.ref_frame.copy()
    demo.font, demo.font_scale, demo.line_type, demo.corners = 0, 1.0, 8, []
    monkeypatch.setattr(MatchingDemo, "draw_quad", lambda *args, **kwargs: None)
    monkeypatch.setattr(MatchingDemo, "putText", lambda *args, **kwargs: None)
    assert demo.create_top_frame().shape == (192, 512, 3)


def test_lighterglue_config_isolation() -> None:
    before = copy.deepcopy(LightGlue.default_conf)
    model = LighterGlue(device="cpu")
    assert model.net.conf.descriptor_dim == 96
    assert LightGlue.default_conf == before
    # kornia accepts None for an unbound extractor; its features annotation only lists str.
    other = LightGlue(None, input_dim=32, descriptor_dim=64, n_layers=2, num_heads=4, weights=None)  # ty: ignore[invalid-argument-type]
    assert other.conf.descriptor_dim == 64 and len(other.transformers) == 2
    assert model.net.conf.descriptor_dim == 96 and len(model.net.transformers) == 6


def test_lighterglue_missing_path_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("An explicit missing checkpoint must not download official weights")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    with pytest.raises(FileNotFoundError):
        LighterGlue(weights=tmp_path / "missing.pt", device="cpu")


@pytest.mark.parametrize("bad", ["missing", "unexpected", "shape", "extractor"])
def test_lighterglue_rejects_invalid_state(bad: str) -> None:
    state = torch.load(ROOT / "weights/xfeat-lighterglue.pt", map_location="cpu", weights_only=True)
    key = next(k for k in state if k.startswith("matcher."))
    if bad == "missing":
        del state[key]
    elif bad == "unexpected":
        state["matcher.not_a_parameter"] = torch.zeros(1)
    elif bad == "extractor":
        key = next(k for k in state if k.startswith("extractor.model.net."))
        state[key] = torch.zeros(1, 2, 3, 4, 5)
    else:
        state[key] = torch.zeros(1, 2, 3, 4, 5)
    with pytest.raises((ValueError, RuntimeError)):
        LighterGlue(weights=state, device="cpu")


def test_empty_sparse_matching(cpu_xfeat: XFeat) -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    points0, points1 = cpu_xfeat.match_xfeat(image, image)
    assert points0.shape == points1.shape == (0, 2)
    features: SparseFeaturesWithSize = {
        "keypoints": torch.empty(0, 2),
        "descriptors": torch.empty(0, 64),
        "scores": torch.empty(0),
        "image_size": (128, 96),
    }
    points0, points1, indices = cpu_xfeat.match_lighterglue(features, features)
    assert points0.shape == points1.shape == indices.shape == (0, 2)


def test_custom_lg_weights_cli_option() -> None:
    result = subprocess.run([sys.executable, "inference.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0
    assert "--lg-weights" in result.stdout


def test_custom_lg_uses_its_extractor_and_checks_explicit_pair(tmp_path: Path) -> None:
    from inference import load_models
    from modules.utils import state_hash

    bundle = ROOT / "weights/xfeat-lighterglue.pt"
    model = load_models(None, bundle, "lighterglue", top_k=32, device="cpu")
    assert model.lighterglue is not None
    assert state_hash(model.net.state_dict()) == model.lighterglue.extractor_hash
    paired = tmp_path / "extractor.pt"
    torch.save(model.net.state_dict(), paired)
    restored = load_models(paired, bundle, "lighterglue", top_k=32, device="cpu")
    assert state_hash(restored.net.state_dict()) == model.lighterglue.extractor_hash
    with pytest.raises(ValueError, match="do not match"):
        load_models(ROOT / "weights/xfeat.pt", bundle, "lighterglue", device="cpu")
    with pytest.raises(ValueError, match="requires"):
        load_models(None, bundle, "xfeat", device="cpu")


def test_lighterglue_legacy_mapping_is_anchored_and_rejects_collision() -> None:
    from modules.lighterglue import split_lighterglue_state

    value = torch.ones(1)
    matcher, _ = split_lighterglue_state({"matcher.self_attn.2.Wqkv.weight": value})
    assert list(matcher) == ["transformers.2.self_attn.Wqkv.weight"]
    with pytest.raises(ValueError, match="Colliding"):
        split_lighterglue_state(
            {
                "matcher.self_attn.2.Wqkv.weight": value,
                "matcher.transformers.2.self_attn.Wqkv.weight": value,
            }
        )
    with pytest.raises(ValueError, match="Unexpected"):
        split_lighterglue_state({"not_matcher.self_attn.2.Wqkv.weight": value})


def test_state_hash_tracks_buffers_and_ignores_serialization(tmp_path: Path) -> None:
    from modules.utils import state_hash

    a = {"weight": torch.arange(4, dtype=torch.float32), "counter": torch.tensor(3)}
    path = tmp_path / "state.pt"
    torch.save(a, path)
    b = torch.load(path, weights_only=True)
    assert state_hash(a) == state_hash(dict(reversed(list(b.items()))))
    b["counter"] += 1
    assert state_hash(a) != state_hash(b)


def test_pip_inference_requirements() -> None:
    names = {re.split(r"[<>=!~]", line)[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()}
    assert {"beartype", "jaxtyping", "opencv-contrib-python"} <= names
    assert not any("opencv" in n and "headless" in n for n in names)
