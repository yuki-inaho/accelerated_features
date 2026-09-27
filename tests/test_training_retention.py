"""Retention preserves restart/export references and leaves auditable deletion evidence."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from xfeat_training.data import file_sha256
from xfeat_training.optim import build_optimizer
from xfeat_training.retention import prune_checkpoints
from xfeat_training.trainer import atomic_json, checkpoint_signature, preflight_run, save_checkpoint


@pytest.fixture
def retention_run(tmp_path):
    root = tmp_path / "run"
    model = nn.Linear(4, 4)
    spec = {"name": "adamw", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    config, identities = (
        {"optimizer": spec, "max_steps": 100, "eval_every": 10, "save_every": 1},
        {"evaluation_hash": "fixed-eval", "pair_hash": "pairs"},
    )
    signature = checkpoint_signature(config, identities)
    optimizer = build_optimizer(model, "xfeat", spec)
    scores = [(0.8, 5), (0.9, 4), (0.9, 6), (0.9, 6), (0.7, 100), (0.6, 10), None, None]
    rows = []
    for step, score in zip([10, 20, 30, 40, 50, 60, 61, 62], scores, strict=True):
        save_checkpoint(
            root / "checkpoints" / f"step_{step:06d}.pt",
            model,
            optimizer,
            {"successful_step": step, "microstep": 0},
            signature,
            config=config,
            identities=identities,
        )
        if score is not None:
            atomic_json(
                root / "validation" / f"step_{step:06d}" / "metrics.json",
                {
                    "split": "val",
                    "evaluation_hash": "fixed-eval",
                    "pair_hash": "pairs",
                    "primary_f1": score[0],
                    "TP": score[1],
                },
            )
        metric_path = root / "validation" / f"step_{step:06d}" / "metrics.json"
        rows.append({"step": step, "validation": json.loads(metric_path.read_text()) if score is not None else None})
    atomic_json(
        root / "run.json",
        {
            "last_checkpoint": "checkpoints/step_000062.pt",
            "completed_steps": 62,
            "signature": signature,
        },
    )
    for prefix, step in (("best", 30), ("last", 60)):
        path = root / "checkpoints" / f"step_{step:06d}.pt"
        atomic_json(
            root / "exports" / f"{prefix}_manifest.json",
            {
                "checkpoint": "checkpoints/" + path.name,
                "checkpoint_status": "complete",
                "checkpoint_file_sha256": file_sha256(path),
            },
        )
    (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (root / "tensorboard").mkdir()
    (root / "tensorboard/events").write_text("all events")
    return root


def checkpoint_bytes(root):
    return {p.name: p.read_bytes() for p in (root / "checkpoints").iterdir()}


def test_ranking_protection_and_evidence(retention_run):
    root = retention_run
    other = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and "checkpoints" not in p.parts
    }
    original = checkpoint_bytes(root)
    result = prune_checkpoints(root, 2)
    assert result["status"] == "committed"
    assert result["top_k"] == ["step_000030.pt", "step_000040.pt"]
    assert set(result["kept"]) == {"step_000030.pt", "step_000040.pt", "step_000060.pt", "step_000062.pt"}
    assert result["protected_over_budget"] == 1
    assert set(result["deletions"]) == {"step_000010.pt", "step_000020.pt", "step_000050.pt", "step_000061.pt"}
    assert set(checkpoint_bytes(root)) == {name + suffix for name in result["kept"] for suffix in ("", ".json")}
    assert all(checkpoint_bytes(root)[name] == original[name] for name in checkpoint_bytes(root))
    assert all((root / name).read_bytes() == value for name, value in other.items())
    for name, record in result["deletions"].items():
        if record["score"] is not None:
            metric = root / "validation" / name.removesuffix(".pt") / "metrics.json"
            assert record["metric_sha256"] == file_sha256(metric)
    latest = root / "checkpoints/step_000062.pt"
    assert preflight_run(root.parent / "resume", latest, json.loads((root / "run.json").read_text())["signature"])
    assert prune_checkpoints(root, 2)["deletions"] == {}


def test_external_resume_reference_never_mutated(retention_run):
    root = retention_run
    external = root.parent / "input/checkpoints/step_000030.pt"
    external.parent.mkdir(parents=True)
    external.write_bytes((root / "checkpoints/step_000030.pt").read_bytes())
    atomic_json(
        root / "exports/best_manifest.json",
        {
            "checkpoint": str(external),
            "checkpoint_status": "complete",
            "checkpoint_file_sha256": file_sha256(external),
        },
    )
    before = external.read_bytes()
    prune_checkpoints(root, 1)
    assert external.read_bytes() == before


@pytest.mark.parametrize(
    "kind", ["payload", "sidecar", "score", "score_bool", "split", "eval_hash", "reference", "symlink", "boundary"]
)
def test_corruption_never_deletes_any_checkpoint(retention_run, kind):
    root = retention_run
    path = root / "checkpoints/step_000020.pt"
    if kind == "payload":
        path.write_bytes(b"damaged")
    elif kind == "sidecar":
        path.with_suffix(".pt.json").unlink()
    elif kind in {"score", "score_bool", "split", "eval_hash"}:
        metric_path = root / "validation/step_000020/metrics.json"
        metric = json.loads(metric_path.read_text())
        key, value = {
            "score": ("primary_f1", 2),
            "score_bool": ("primary_f1", True),
            "split": ("split", "test"),
            "eval_hash": ("evaluation_hash", "other"),
        }[kind]
        metric[key] = value
        atomic_json(metric_path, metric)
    elif kind == "reference":
        (root / "checkpoints/step_000060.pt").unlink()
    elif kind == "symlink":
        path.unlink()
        path.symlink_to("step_000010.pt")
    else:
        payload = torch.load(path, weights_only=False)
        payload["state"]["microstep"] = 1
        torch.save(payload, path)
        sidecar = json.loads(path.with_suffix(".pt.json").read_text())
        sidecar["file_sha256"] = file_sha256(path)
        atomic_json(path.with_suffix(".pt.json"), sidecar)
    before = checkpoint_bytes(root)
    with pytest.raises((ValueError, FileNotFoundError)):
        prune_checkpoints(root, 2)
    assert checkpoint_bytes(root) == before
    assert not (root / "retention.json").exists()


@pytest.mark.parametrize("after_unlinks", [0, 1, 2, 3, 8])
def test_interrupted_deletions_recover(retention_run, monkeypatch, after_unlinks):
    root = retention_run
    original_unlink = Path.unlink
    count = 0

    def crash(path, *args, **kwargs):
        nonlocal count
        if path.parent.name == "checkpoints":
            if count == after_unlinks:
                raise OSError("simulated crash")
            count += 1
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", crash)
    if after_unlinks < 8:
        with pytest.raises(OSError, match="simulated"):
            prune_checkpoints(root, 2)
        assert json.loads((root / "retention.json").read_text())["rounds"][-1]["status"] == "pending"
    else:
        prune_checkpoints(root, 2)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    result = prune_checkpoints(root, 2)
    assert len(result["kept"]) == 4
    assert all(item["status"] == "committed" for item in json.loads((root / "retention.json").read_text())["rounds"])


def test_pending_corruption_prevents_further_deletion(retention_run, monkeypatch):
    root = retention_run
    original = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        prune_checkpoints(root, 2)
    monkeypatch.setattr(Path, "unlink", original)
    # Corrupt a retained file as well as exercising the pending deletion path.
    (root / "checkpoints/step_000040.pt").write_bytes(b"corrupted")
    before = checkpoint_bytes(root)
    with pytest.raises(ValueError, match="checksum"):
        prune_checkpoints(root, 2)
    assert checkpoint_bytes(root) == before


def test_concurrent_pruning_lock(retention_run):
    before = checkpoint_bytes(retention_run)
    with (retention_run / "retention.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            prune_checkpoints(retention_run, 2)
    assert checkpoint_bytes(retention_run) == before


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_k_rejected_without_mutation(retention_run, value):
    before = checkpoint_bytes(retention_run)
    with pytest.raises(ValueError, match="positive integer"):
        prune_checkpoints(retention_run, value)
    assert checkpoint_bytes(retention_run) == before


def test_k_is_not_a_numeric_training_signature():
    assert checkpoint_signature({"checkpoint_keep_best": 1}, {}) == checkpoint_signature(
        {"checkpoint_keep_best": 9}, {}
    )


@pytest.mark.parametrize("new_k", [1, 3])
def test_pending_policy_change_never_deletes(retention_run, monkeypatch, new_k):
    root = retention_run
    original = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        prune_checkpoints(root, 2)
    monkeypatch.setattr(Path, "unlink", original)
    before = checkpoint_bytes(root)
    with pytest.raises(ValueError, match="original keep_best"):
        prune_checkpoints(root, new_k)
    assert checkpoint_bytes(root) == before
    assert prune_checkpoints(root, 2)["status"] == "committed"


@pytest.mark.parametrize("kind", ["body", "both", "metric", "log", "latest_pointer", "completed_steps"])
def test_missing_evidence_and_stale_pointer_never_delete(retention_run, kind):
    root = retention_run
    path = root / "checkpoints/step_000020.pt"
    if kind in {"body", "both"}:
        path.unlink()
        if kind == "both":
            path.with_suffix(".pt.json").unlink()
    elif kind == "metric":
        (root / "validation/step_000020/metrics.json").unlink()
    elif kind == "log":
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        rows[1]["validation"] = None
        (root / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        run = json.loads((root / "run.json").read_text())
        if kind == "latest_pointer":
            run["last_checkpoint"] = "checkpoints/step_000061.pt"
        else:
            run["completed_steps"] = 61
        atomic_json(root / "run.json", run)
    before = checkpoint_bytes(root)
    with pytest.raises(ValueError):
        prune_checkpoints(root, 2)
    assert checkpoint_bytes(root) == before
    assert not (root / "retention.json").exists()


def test_missing_historically_kept_pair_is_not_silent(retention_run):
    root = retention_run
    prune_checkpoints(root, 2)
    path = root / "checkpoints/step_000040.pt"
    path.unlink()
    path.with_suffix(".pt.json").unlink()
    before = checkpoint_bytes(root)
    with pytest.raises(ValueError, match="missing without retention"):
        prune_checkpoints(root, 2)
    assert checkpoint_bytes(root) == before


def test_checkpoint_directory_sync_precedes_deletion(retention_run, monkeypatch):
    from xfeat_training import retention

    order = []
    sync, unlink = retention._sync_directory, Path.unlink

    def tracked_sync(path):
        order.append(("sync", path.name))
        sync(path)

    def tracked_unlink(path, *args, **kwargs):
        order.append(("unlink", path.name))
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(retention, "_sync_directory", tracked_sync)
    monkeypatch.setattr(Path, "unlink", tracked_unlink)
    prune_checkpoints(retention_run, 2)
    first_unlink = next(i for i, (kind, _) in enumerate(order) if kind == "unlink")
    assert ("sync", "checkpoints") in order[:first_unlink]
