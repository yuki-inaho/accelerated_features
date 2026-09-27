"""Independent pair generation, split audits and deterministic manifests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xfeat_training import mining


@pytest.fixture
def mining_config() -> dict:
    return {
        "seed": 20260927,
        "chunk_size": 2,
        "exclude_before": {},
        "frame_exclusions": [],
        "split_chunks": {"train": [0, 1], "guard": [2, 4], "val": [3], "test": [5]},
        "min_frames": {"train": 4, "val": 2, "test": 2},
        "candidate_max_gap": 60,
        "train_max_gap": 30,
        "max_rotation_deg": 15,
        "min_translation_m": 0.005,
        "max_translation_m": 1.0,
        "train_overlap": [0.2, 0.85],
        "overlap_bins": [0.2, 0.3, 0.5, 0.7, 0.85],
        "bin_weights": [0.25, 0.35, 0.25, 0.15],
        "eval_gaps": [1, 5, 10, 20, 30, 40],
        "eval_pairs_per_bin": 12,
        "smoke_per_subset": 4,
        "stride": 8,
        "depth_tolerance": 0.03,
        "leak_overlap": 0.2,
        "archive_path": None,
    }


def fake_overlap(a, b, **kwargs) -> float:
    """Controlled split overlap; geometry itself has independent plane tests."""
    split_a = "train" if a.chunk_id < 2 else a.chunk_id
    split_b = "train" if b.chunk_id < 2 else b.chunk_id
    return 0.6 if split_a == split_b else 0.0


def test_mining_repeatable_and_separate_subsets(l76_factory, mining_config, tmp_path, monkeypatch) -> None:
    root = l76_factory(ids=tuple(range(12)))
    monkeypatch.setattr(mining, "overlap", fake_overlap)
    a = mining.mine_pairs(root, tmp_path / "a", mining_config)
    b = mining.mine_pairs(root, tmp_path / "b", mining_config)
    assert a["content_hash"] == b["content_hash"]
    for split in ("train", "val", "test", "smoke"):
        arrays = mining.load_pairs(tmp_path / "a" / f"{split}.npz")
        with np.load(tmp_path / "b" / f"{split}.npz", allow_pickle=False) as other:
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], other[key])
        assert set(arrays["subset"]) == {"colmap_rgbd_half_s0", "colmap_rgbd_half_s1"}
        assert arrays["frame0"].dtype == np.int64 and arrays["overlap"].dtype == np.float32
        assert not np.isin(arrays["chunk0"], [2, 4]).any()
    assert len(mining.load_pairs(tmp_path / "a/smoke.npz")["frame0"]) == 8
    assert a["empty_train_bins"]  # fixed .6 overlap leaves other bins empty, explicitly reported
    changed = copy.deepcopy(mining_config)
    changed["seed"] += 1
    with pytest.raises(FileExistsError):
        mining.mine_pairs(root, tmp_path / "a", changed)
    path = tmp_path / "a/manifest.json"
    original = json.loads(path.read_text())
    for key in ("array_hashes", "resolved_config"):
        broken = copy.deepcopy(original)
        if key == "array_hashes":
            broken[key]["train"] = "different"
        else:
            broken[key]["bin_weights"] = [0.1, 0.1, 0.1, 0.7]
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="identity"):
            mining.pair_dataset(tmp_path / "a")


@pytest.mark.parametrize("leak_pair", [(1, 6), (1, 10), (6, 10)])
def test_all_cross_split_pairs_are_audited(l76_factory, mining_config, tmp_path, monkeypatch, leak_pair) -> None:
    root = l76_factory(ids=tuple(range(12)))

    def overlap_with_leak(a, b, **kwargs):
        if (a.key.frame_id, b.key.frame_id) == leak_pair:
            return 0.3
        return fake_overlap(a, b)

    monkeypatch.setattr(mining, "overlap", overlap_with_leak)
    with pytest.raises(ValueError, match="leak"):
        mining.mine_pairs(root, tmp_path / "bad", mining_config)


def test_no_warmup_pairs_is_explicit_error(l76_factory, mining_config, tmp_path, monkeypatch) -> None:
    root = l76_factory(ids=tuple(range(12)))
    monkeypatch.setattr(mining, "overlap", lambda a, b, **kw: 0.3 if fake_overlap(a, b) else 0.0)
    with pytest.raises(ValueError, match=r"warmup|smoke"):
        mining.mine_pairs(root, tmp_path / "no_warmup", mining_config)


def test_mining_cli_config_and_unknown_override(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "L76_DATA_ROOT": str(tmp_path)}
    command = [sys.executable, "-m", "scripts.mine_pairs"]
    show = subprocess.run([*command, "--cfg", "job", "--resolve"], cwd=root, env=env, capture_output=True, text=True)
    assert show.returncode == 0, show.stderr
    assert "train_max_gap: 30" in show.stdout
    bad = subprocess.run([*command, "typo=1"], cwd=root, env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "typo" in bad.stderr
