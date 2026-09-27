"""Prune completed checkpoint transactions while preserving exports and restart state.

Call only from the trainer's single-writer boundary after all log flushes, or after
the owning training process has exited. The lock serializes pruning operations;
it does not turn a legacy trainer into a cooperating writer.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from xfeat_training.data import file_sha256, json_hash
from xfeat_training.selection import checkpoint_candidate, selection_rank


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_ledger(root: Path, ledger: dict[str, Any]) -> None:
    from xfeat_training.trainer import atomic_json

    atomic_json(root / "retention.json", ledger)
    _sync_directory(root)


def _checkpoint_path(root: Path, name: str) -> Path:
    if not re.fullmatch(r"step_[0-9]+\.pt", name):
        raise ValueError("Invalid checkpoint name in retention ledger")
    path = root / "checkpoints" / name
    if path.is_symlink() or path.with_suffix(".pt.json").is_symlink():
        raise ValueError("Checkpoint symlinks are not supported")
    return path


def _protected(root: Path, run: dict[str, Any]) -> dict[str, list[str]]:
    protected: dict[str, list[str]] = {}
    references = [(run["last_checkpoint"], "latest", None)]
    for prefix in ("best", "last"):
        manifest_path = root / "exports" / f"{prefix}_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["checkpoint_status"] != "complete":
            raise ValueError("Cannot prune while an export transaction is incomplete")
        references.append((manifest["checkpoint"], prefix + "_export", manifest["checkpoint_file_sha256"]))
    for value, reason, digest in references:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or invalid protected checkpoint: {reason}")
        if digest is not None and file_sha256(path) != digest:
            raise ValueError(f"Protected checkpoint checksum mismatch: {reason}")
        if path.resolve().parent == (root / "checkpoints").resolve():
            _checkpoint_path(root, path.name)
            protected.setdefault(path.name, []).append(reason)
        # A resumed run can refer to its immutable input run. Never prune there.
    return protected


def _finish_pending(root: Path, ledger: dict[str, Any], protected: dict[str, list[str]]) -> None:
    if not ledger["rounds"] or ledger["rounds"][-1]["status"] != "pending":
        return
    transaction = ledger["rounds"][-1]
    deletions = transaction["deletions"]
    if set(deletions) & set(protected):
        raise ValueError("Pending deletion now references a protected checkpoint")
    # Validate every remaining file before performing any deletion in this attempt.
    for name, record in deletions.items():
        path = _checkpoint_path(root, name)
        for target, key in ((path, "file_sha256"), (path.with_suffix(".pt.json"), "sidecar_sha256")):
            if target.exists() and file_sha256(target) != record[key]:
                raise ValueError("Pending retention file checksum mismatch")
    # Persist entries for the newly retained checkpoint pair before removing old pairs.
    _sync_directory(root / "checkpoints")
    for name in deletions:
        path = _checkpoint_path(root, name)
        path.unlink(missing_ok=True)
        path.with_suffix(".pt.json").unlink(missing_ok=True)
    _sync_directory(root / "checkpoints")
    transaction["status"] = "committed"
    _write_ledger(root, ledger)


def _validation_score(
    root: Path, step: int, config: dict, evaluation_hash: str, pair_hash: str, rows: dict
) -> tuple[Any, Any]:
    metric_path = root / "validation" / f"step_{step:06d}" / "metrics.json"
    expected = step % config["eval_every"] == 0 or step == config["max_steps"]
    row = rows.get(step)
    if row is None or (row["validation"] is not None) != expected or metric_path.is_file() != expected:
        raise ValueError("Checkpoint validation schedule/log/file mismatch")
    if not expected:
        return None, None
    metric = json.loads(metric_path.read_text())
    if (
        metric != row["validation"]
        or metric["split"] != "val"
        or metric["evaluation_hash"] != evaluation_hash
        or metric["pair_hash"] != pair_hash
    ):
        raise ValueError("Invalid checkpoint ranking metric")
    return checkpoint_candidate(metric, step, config), file_sha256(metric_path)


def prune_checkpoints(run_dir: str | Path, keep_best: int) -> dict[str, Any]:
    """Keep top K by explicit val metric (default F1/TP), plus latest and exports."""
    from xfeat_training.trainer import CHECKPOINT_SCHEMA, checkpoint_signature

    if isinstance(keep_best, bool) or not isinstance(keep_best, int) or keep_best < 1:
        raise ValueError("checkpoint_keep_best must be a positive integer")
    root = Path(run_dir).resolve()
    if (root / "checkpoints").is_symlink():
        raise ValueError("Checkpoint directory symlinks are not supported")
    with (root / "retention.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = json.loads((root / "run.json").read_text())
        protected = _protected(root, run)
        ledger_path = root / "retention.json"
        ledger: dict[str, Any] = (
            json.loads(ledger_path.read_text()) if ledger_path.exists() else {"schema_version": 1, "rounds": []}
        )
        if ledger.get("schema_version") != 1 or not isinstance(ledger.get("rounds"), list):
            raise ValueError("Unsupported retention ledger")
        if any(item.get("status") != "committed" for item in ledger["rounds"][:-1]):
            raise ValueError("Invalid unfinished retention history")
        if ledger["rounds"] and ledger["rounds"][-1].get("status") not in {"pending", "committed"}:
            raise ValueError("Invalid retention status")
        pending = ledger["rounds"][-1] if ledger["rounds"] and ledger["rounds"][-1]["status"] == "pending" else None
        if pending is not None and pending["keep_best"] != keep_best:
            raise ValueError("Finish pending retention with its original keep_best before changing policy")
        pending_names = set(pending["deletions"]) if pending else set()
        bodies = {path.name for path in (root / "checkpoints").glob("*.pt")}
        sidecars = {path.name.removesuffix(".json") for path in (root / "checkpoints").glob("*.pt.json")}
        for name in bodies | sidecars:
            _checkpoint_path(root, name)
        if (bodies ^ sidecars) - pending_names:
            raise ValueError("Incomplete checkpoint/sidecar pair outside pending retention")
        rows: dict[int, Any] = {}
        for line in (root / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            step = row["step"]
            if type(step) is not int or step < 1 or step in rows:
                raise ValueError("Invalid or duplicate metrics step")
            rows[step] = row
        candidates: dict[str, Any] = {}
        configs: dict[str, Any] = {}
        evaluation_hashes: dict[str, str] = {}
        pair_hashes: dict[str, str] = {}
        for path in sorted((root / "checkpoints").glob("*.pt")):
            if path.name in pending_names:
                # A prior, validated deletion can have lost either file already.
                # _finish_pending rechecks every surviving byte before retrying.
                continue
            _checkpoint_path(root, path.name)
            sidecar_path = path.with_suffix(".pt.json")
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar["schema_version"] != CHECKPOINT_SCHEMA or sidecar["signature"] != run["signature"]:
                raise ValueError("Checkpoint retention schema/signature mismatch")
            if file_sha256(path) != sidecar["file_sha256"]:
                raise ValueError("Checkpoint retention checksum mismatch")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            step = saved["state"]["successful_step"]
            if (
                saved["schema_version"] != CHECKPOINT_SCHEMA
                or saved["signature"] != run["signature"]
                or checkpoint_signature(saved["config"], saved["identities"]) != run["signature"]
                or saved["state"]["microstep"] != 0
                or saved["parameter_state"] not in {"Y", "standard"}
                or saved["parameter_state"] != sidecar["parameter_state"]
                or saved["optimizer_metadata"]["parameter_state"] != saved["parameter_state"]
                or (
                    saved["parameter_state"] == "Y"
                    and (
                        not saved["optimizer_metadata"]["train_mode"]
                        or any(group["k"] != step for group in saved["optimizer"]["param_groups"])
                    )
                )
                or step != sidecar["successful_step"]
                or path.name != f"step_{step:06d}.pt"
                or step > run["completed_steps"]
                or not all(torch.isfinite(value).all() for value in saved["model"].values())
            ):
                raise ValueError("Checkpoint retention boundary/metadata mismatch")
            configs[path.name] = saved["config"]
            evaluation_hashes[path.name] = saved["identities"]["evaluation_hash"]
            pair_hashes[path.name] = saved["identities"]["pair_hash"]
            score, metric_digest = _validation_score(
                root, step, configs[path.name], evaluation_hashes[path.name], pair_hashes[path.name], rows
            )
            candidates[path.name] = {
                "schema_version": saved["schema_version"],
                "parameter_state": saved["parameter_state"],
                "microstep": saved["state"]["microstep"],
                "file_sha256": sidecar["file_sha256"],
                "sidecar_sha256": file_sha256(sidecar_path),
                "signature": saved["signature"],
                "step": step,
                "score": score,
                "metric_sha256": metric_digest,
                "metric_path": f"validation/step_{step:06d}/metrics.json" if score is not None else None,
            }
        completed = run["completed_steps"]
        latest = f"step_{completed:06d}.pt"
        if (
            type(completed) is not int
            or completed < 1
            or latest not in candidates
            or "latest" not in protected.get(latest, [])
            or max(record["step"] for record in candidates.values()) != completed
        ):
            raise ValueError("Latest checkpoint pointer/completed_steps mismatch")
        config, evaluation_hash = configs[latest], evaluation_hashes[latest]
        removed: dict[str, Any] = {}
        historic = set()
        for transaction in ledger["rounds"]:
            historic.update(transaction["kept"])
            for name, record in transaction["deletions"].items():
                _checkpoint_path(root, name)
                if name in removed or record["signature"] != run["signature"]:
                    raise ValueError("Invalid retention deletion history")
                if transaction["status"] == "committed" and name in (bodies | sidecars):
                    raise ValueError("Committed deletion unexpectedly reappeared")
                score, digest = _validation_score(
                    root, record["step"], config, evaluation_hash, pair_hashes[latest], rows
                )
                if record["score"] != score or record["metric_sha256"] != digest:
                    raise ValueError("Retained validation evidence no longer matches deletion history")
                removed[name] = record
        expected_names = {
            f"step_{step:06d}.pt"
            for step in rows
            if step <= completed
            and (
                step % config["save_every"] == 0
                or step % config["eval_every"] == 0
                or step in {config["max_steps"], config.get("stop_after_steps"), completed}
            )
        }
        if (expected_names | historic) - set(candidates) - set(removed):
            raise ValueError("Checkpoint missing without retention evidence")
        ranked = sorted(
            (name for name, record in candidates.items() if record["score"] is not None),
            key=lambda name: selection_rank(candidates[name]["score"]),
            reverse=True,
        )
        keep = set(ranked[:keep_best]) | set(protected)
        if not set(protected) <= set(candidates):
            raise ValueError("Protected checkpoint was not validated")
        # Validate retained candidates too before recovering an interrupted round.
        _finish_pending(root, ledger, protected)
        transaction = {
            "sequence": len(ledger["rounds"]) + 1,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "run_id": json_hash({"path": str(root), "signature": run["signature"]}),
            "run_signature": run["signature"],
            "status": "pending",
            "keep_best": keep_best,
            "completed_steps": run["completed_steps"],
            "top_k": ranked[:keep_best],
            "protected": protected,
            "kept": {name: candidates[name] for name in sorted(keep)},
            "deletions": {name: record for name, record in candidates.items() if name not in keep},
            "protected_over_budget": max(0, len(keep) - (keep_best + 1)),
        }
        ledger["rounds"].append(transaction)
        _write_ledger(root, ledger)
        _finish_pending(root, ledger, protected)
        return transaction
