"""Checkpoint integrity, immutable preflight and exact CPU stochastic resume."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from xfeat_training.optim import build_optimizer
from xfeat_training.trainer import (
    PairSampler,
    checkpoint_signature,
    preflight_run,
    restore_checkpoint,
    save_checkpoint,
    train_loop,
    validate_precision,
)


def test_resume_cpu_restores_rng_optimizer_and_counters(tmp_path, amuse_config, assert_nested_equal) -> None:
    random.seed(3)
    np.random.seed(3)  # noqa: NPY002 -- explicitly exercise global RNG checkpointing
    torch.manual_seed(3)
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    config, identities = {"max_steps": 9, "optimizer": amuse_config}, {"data": "fixture"}
    signature = checkpoint_signature(config, identities)
    state = {
        "successful_step": 0,
        "attempted_microbatches": 0,
        "accepted_microbatches": 0,
        "microstep": 0,
        "sampler": {"cursor": 0},
        "skips": {},
    }

    def update(m, o, s):
        m(torch.rand(4, 4)).square().mean().mul(random.random() + np.random.rand()).backward()  # noqa: NPY002 -- global RNG test
        o.step()
        o.zero_grad(set_to_none=True)
        for key in ("successful_step", "attempted_microbatches", "accepted_microbatches"):
            s[key] += 1
        s["sampler"]["cursor"] += 1

    for _ in range(5):
        update(model, optimizer, state)
    path = tmp_path / "step_000005.pt"
    save_checkpoint(path, model, optimizer, state, signature, config=config, identities=identities)
    for _ in range(4):
        update(model, optimizer, state)
    resumed_model = nn.Linear(4, 4)
    resumed_opt = build_optimizer(resumed_model, "xfeat", amuse_config)
    loaded = preflight_run(tmp_path / "resumed", path, signature)
    assert loaded is not None
    assert not (tmp_path / "resumed").exists()
    resumed_state = restore_checkpoint(loaded, resumed_model, resumed_opt)
    for _ in range(4):
        update(resumed_model, resumed_opt, resumed_state)
    assert_nested_equal(model.state_dict(), resumed_model.state_dict())
    assert_nested_equal(optimizer.state_dict(), resumed_opt.state_dict())
    assert_nested_equal(state, resumed_state)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, state, signature, config=config, identities=identities)


def test_resume_rejection_does_not_create_or_modify_directories(tmp_path, amuse_config) -> None:
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    config = {
        "max_steps": 12,
        "run_dir": "old",
        "resume_from": None,
        "stop_after_steps": 5,
        "save_every": 5,
        "precision": "fp32",
        "optimizer": amuse_config,
    }
    signature = checkpoint_signature(config, {"pair_hash": "a"})
    changed_controls = {
        **config,
        "run_dir": "new",
        "resume_from": "somewhere",
        "stop_after_steps": None,
        "save_every": 1,
    }
    assert checkpoint_signature(changed_controls, {"pair_hash": "a"}) == signature
    old = tmp_path / "old"
    old.mkdir()
    path = old / "step_000000.pt"
    state = {
        "successful_step": 0,
        "attempted_microbatches": 0,
        "accepted_microbatches": 0,
        "microstep": 0,
        "sampler": {"cursor": 0},
        "skips": {},
    }
    save_checkpoint(path, model, optimizer, state, signature, config=config, identities={"pair_hash": "a"})
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    new = tmp_path / "new"
    for changed in (
        {**config, "max_steps": 13},
        {**config, "precision": "fp16"},
        {**config, "optimizer": {**amuse_config, "rho": 0.4}},
    ):
        with pytest.raises(ValueError, match="signature"):
            preflight_run(new, path, checkpoint_signature(changed, {"pair_hash": "a"}))
        assert not new.exists() and before == {p.name: p.read_bytes() for p in old.iterdir()}
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        preflight_run(new, path, signature)
    assert not new.exists()


def test_no_checkpoint_during_accumulation(tmp_path, amuse_config) -> None:
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    model(torch.ones(4, 4)).sum().backward()
    with pytest.raises(ValueError, match=r"boundary|gradient|microstep"):
        save_checkpoint(tmp_path / "bad.pt", model, optimizer, {"microstep": 1}, "signature", config={}, identities={})
    assert not list(tmp_path.iterdir())


def test_bf16_autocast_rejected_on_turing(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (7, 5))
    with pytest.raises(ValueError, match="bf16"):
        validate_precision("bf16", torch.device("cuda"))
    validate_precision("fp32", torch.device("cuda"))


class TinyTask:
    def __init__(self, directory, invalid=False, nonfinite=False):
        torch.manual_seed(123)
        self.model = nn.Linear(4, 4)
        self.directory, self.invalid, self.nonfinite = directory, invalid, nonfinite
        self.preview = torch.zeros(3, 8, 8)
        self.skip_reason = "fixture_invalid"

    def loss(self, pair_index, accepted):
        if self.invalid:
            return None
        loss = self.model(torch.rand(3, 4)).square().mean() * (random.random() + np.random.rand())  # noqa: NPY002 -- global RNG test
        if self.nonfinite:
            loss = loss * float("nan")
        return {"loss": loss, "supervised": torch.tensor(3)}

    def evaluate(self, step):
        # Evaluation may use all RNGs, but must leave training RNGs untouched.
        random.random()
        np.random.rand()  # noqa: NPY002 -- evaluation must restore global RNG
        torch.rand(3)
        return {"primary_f1": 0.5, "TP": 2, "P": 4, "G": 4, "A": 4}

    def export(self, prefix, step):
        target = self.directory / "exports"
        target.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), target / f"{prefix}.pt")


def tiny_loop(tmp_path, name, amuse_config, *, stop=None, resume=None, invalid=False, nonfinite=False, keep_best=None):
    directory = tmp_path / name
    config = {
        "run_dir": str(directory),
        "resume_from": str(resume) if resume else None,
        "stop_after_steps": stop,
        "max_steps": 12,
        "save_every": 5,
        "eval_every": 5,
        "checkpoint_keep_best": keep_best,
        "accumulation": 4,
        "grad_clip": 1.0,
        "seed": 22,
        "precision": "fp32",
        "optimizer": amuse_config,
    }
    task = TinyTask(directory, invalid, nonfinite)
    optimizer = build_optimizer(task.model, "xfeat", amuse_config)
    pairs = {
        "subset": np.array(["a", "b"]),
        "difficulty": np.array(["overlap_2", "overlap_2"]),
        "overlap": np.array([0.6, 0.6]),
    }
    sampler = PairSampler(pairs, {"bin_weights": [0.25, 0.35, 0.25, 0.15]}, 22)
    return train_loop(config, task, optimizer, sampler, {"fixture": True})


def test_retention_hook_runs_after_log_flushes(tmp_path, amuse_config, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from xfeat_training import retention

    seen = []

    def inspect_boundary(root, keep_best):
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        step = rows[-1]["step"]
        assert json.loads((root / "run.json").read_text())["completed_steps"] == step
        events = EventAccumulator(str(root / "tensorboard"), size_guidance={"scalars": 0}).Reload()
        assert events.Scalars("loss/loss")[-1].step == step
        assert (root / "checkpoints" / f"step_{step:06d}.pt.json").is_file()
        assert keep_best == 2
        seen.append(step)

    monkeypatch.setattr(retention, "prune_checkpoints", inspect_boundary)
    tiny_loop(tmp_path, "run", amuse_config, keep_best=2)
    assert seen == [5, 10, 12]
    assert json.loads((tmp_path / "run/completed.json").read_text())["successful_step"] == 12


def test_complete_loop_resume_and_tensorboard(tmp_path, amuse_config, assert_nested_equal):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    full = tiny_loop(tmp_path, "full", amuse_config)
    cut = tiny_loop(tmp_path, "cut", amuse_config, stop=5)
    resumed = tiny_loop(tmp_path, "resumed", amuse_config, resume=tmp_path / "cut/checkpoints/step_000005.pt")
    assert cut["successful_step"] == 5
    assert full["accepted_microbatches"] == full["attempted_microbatches"] == 48
    assert_nested_equal(full, resumed)
    a = torch.load(tmp_path / "full/checkpoints/step_000012.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/checkpoints/step_000012.pt", weights_only=False)
    for key in ("model", "optimizer", "rng", "state", "signature", "optimizer_metadata"):
        assert_nested_equal(a[key], b[key])
    rows = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    events = EventAccumulator(str(tmp_path / "full/tensorboard"), size_guidance={"scalars": 0, "images": 0}).Reload()
    assert [value.step for value in events.Scalars("loss/loss")] == [row["step"] for row in rows] == list(range(1, 13))
    assert [value.step for value in events.Images("pair/source")] == [1, 5, 10, 12]
    assert events.Scalars("validation/primary_f1")[-1].step == 12
    assert all(row["parameter_gradient_norms"] for row in rows)
    exported = torch.load(tmp_path / "full/exports/last.pt", weights_only=True)
    assert any(not torch.equal(exported[key], a["model"][key]) for key in exported)
    assert a["parameter_state"] == "Y"
    assert a["identities"] == b["identities"] == {"fixture": True}
    assert a["resume_origin"] is None
    assert b["resume_origin"]["identities"] == {"fixture": True}
    from xfeat_training.data import file_sha256

    assert b["resume_origin"]["file_sha256"] == file_sha256(tmp_path / "cut/checkpoints/step_000005.pt")


@pytest.mark.parametrize("invalid,nonfinite,message", [(True, False, "100 consecutive"), (False, True, "Nonfinite")])
def test_loop_failures_leave_evidence_without_false_checkpoint(tmp_path, amuse_config, invalid, nonfinite, message):
    with pytest.raises((RuntimeError, FloatingPointError), match=message):
        tiny_loop(tmp_path, "bad", amuse_config, invalid=invalid, nonfinite=nonfinite)
    failure = json.loads((tmp_path / "bad/failure.json").read_text())
    assert failure["successful_step"] == 0 and failure["recent_pair_indices"]
    assert not (tmp_path / "bad/checkpoints").exists()
    if invalid:
        assert failure["attempted_microbatches"] == 100


def test_zero_stop_rejected_without_output(tmp_path, amuse_config):
    with pytest.raises(ValueError, match="stop_after_steps"):
        tiny_loop(tmp_path, "bad", amuse_config, stop=0)
    assert not (tmp_path / "bad").exists()


def test_off_cycle_stop_does_not_change_validation_selection(tmp_path, amuse_config, assert_nested_equal):
    full = tiny_loop(tmp_path, "full", amuse_config)
    stopped = tiny_loop(tmp_path, "cut", amuse_config, stop=3)
    assert stopped["best"] is None
    assert (tmp_path / "cut/exports/last.pt").is_file()
    resumed = tiny_loop(tmp_path, "resume", amuse_config, resume=tmp_path / "cut/checkpoints/step_000003.pt")
    assert_nested_equal(full, resumed)


@pytest.mark.parametrize("split", ["val", "test"])
def test_training_never_accepts_evaluation_splits(split):
    from xfeat_training.trainer import TrainingTask

    with pytest.raises(ValueError, match="train/smoke"):
        TrainingTask({"device": "cpu", "precision": "fp32", "task": "xfeat", "pair_split": split})
