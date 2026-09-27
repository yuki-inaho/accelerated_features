"""Real head tasks on a tiny RGB-D fixture, including common loop and retention."""

import json
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

from modules.raco import XFeatRaCo
from modules.utils import state_hash
from xfeat_training import mining


@pytest.fixture
def raco_config(l76_factory, tmp_path, monkeypatch):
    root = l76_factory(ids=tuple(range(0, 60, 5)), width=96)
    cfg = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/pair_mining/default.yaml")))
    cfg.update(
        chunk_size=2,
        exclude_before={},
        min_frames={"train": 4, "val": 2, "test": 2},
        split_chunks={"train": [0, 1], "guard": [2, 4], "val": [3], "test": [5]},
        archive_path=None,
    )
    for key in ("defaults", "hydra", "data_root"):
        cfg.pop(key)
    monkeypatch.setattr(
        mining,
        "overlap",
        lambda a, b, **kw: 0.6 if (a.chunk_id == b.chunk_id or max(a.chunk_id, b.chunk_id) < 2) else 0.0,
    )
    mining.mine_pairs(root, tmp_path / "pairs", cfg)
    config = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/raco.yaml")))
    config.pop("defaults")
    config.pop("hydra")
    config.update(
        pairs_dir=str(tmp_path / "pairs"),
        device="cpu",
        max_steps=5,
        save_every=1,
        eval_every=1,
        val_frames=2,
        candidate_limit=64,
        detection_threshold=0.001,
        budgets=[8, 16],
        run_dir=str(tmp_path / "unused"),
    )
    return config


@pytest.mark.parametrize("task", ["raco_rank", "raco_covariance"])
def test_raco_real_tasks_resume_retention_and_frozen_base(raco_config, tmp_path, task, assert_nested_equal):
    from xfeat_training.raco_task import run_raco_training

    cfg = {**raco_config, "task": task}
    if task == "raco_covariance":
        torch.manual_seed(90)
        init = XFeatRaCo(candidate_limit=64, detection_threshold=0.001)
        init_path = tmp_path / "init.pt"
        torch.save(init.bundle(trained_heads=["rank"]), init_path)
        cfg["init_bundle"] = str(init_path)
    cfg["run_dir"] = str(tmp_path / "full")
    full = run_raco_training(cfg)
    cfg.update(run_dir=str(tmp_path / "cut"), stop_after_steps=2)
    run_raco_training(cfg)
    checkpoint = tmp_path / "cut/checkpoints/step_000002.pt"
    immutable = checkpoint.read_bytes()
    cfg.update(run_dir=str(tmp_path / "resumed"), stop_after_steps=None, resume_from=str(checkpoint))
    resumed = run_raco_training(cfg)
    assert_nested_equal(full, resumed)
    a = torch.load(tmp_path / "full/checkpoints/step_000005.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/checkpoints/step_000005.pt", weights_only=False)
    for key in ("model", "optimizer", "rng", "state", "signature"):
        assert_nested_equal(a[key], b[key])
    assert immutable == checkpoint.read_bytes()
    before = XFeatRaCo(candidate_limit=64, detection_threshold=0.001)
    for key, tensor in before.net.state_dict().items():
        torch.testing.assert_close(tensor, a["model"]["net." + key], atol=0, rtol=0)
    rows = [json.loads(x) for x in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    active = "heads.ranker." if task == "raco_rank" else "heads.covariance_head."
    assert all(all(key.startswith(active) for key in row["parameter_gradient_norms"]) for row in rows)
    assert any(any(value > 0 for value in row["parameter_gradient_norms"].values()) for row in rows)
    assert len(list((tmp_path / "full/checkpoints").glob("*.pt"))) <= 4
    ledger = json.loads((tmp_path / "full/retention.json").read_text())
    assert all(round_["status"] == "committed" for round_ in ledger["rounds"])
    assert any(round_["deletions"] for round_ in ledger["rounds"])
    assert rows[-1]["validation"]["split"] == "val"
    model = XFeatRaCo.from_bundle(tmp_path / "full/exports/last_raco.pt", covariance=task == "raco_covariance")
    assert state_hash(model.net.state_dict()) == state_hash(before.net.state_dict())
    features = model.extract(torch.rand(1, 3, 64, 96))[0]
    assert "ranker_scores" in features


def test_raco_rejects_eval_training_and_invalid_head_config(tmp_path):
    from xfeat_training.raco_task import run_raco_training

    config = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/raco.yaml")))
    config.update(run_dir=str(tmp_path / "run"), pair_split="val", device="cpu")
    with pytest.raises(ValueError, match="train/smoke"):
        run_raco_training(config)
    assert not (tmp_path / "run").exists()
