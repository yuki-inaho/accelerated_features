"""Shared training loop, immutable preflight and atomic Y-state checkpoints."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from xfeat_training.data import file_sha256, json_hash
from xfeat_training.optim import (
    capture_rng,
    evaluation_parameters,
    finish_update,
    optimizer_metadata,
    restore_optimizer,
    restore_rng,
)
from xfeat_training.selection import checkpoint_candidate, selection_rank

CHECKPOINT_SCHEMA = 2
RUN_CONTROL_KEYS = {"run_dir", "resume_from", "stop_after_steps", "save_every", "checkpoint_keep_best"}


def configure_reproducibility(seed: int, *, warn_only: bool = False) -> None:
    # Called before model construction/CUDA GEMMs in the CLI path.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- seed global RNG as well as independent sampler Generator
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def checkpoint_signature(config: Mapping[str, Any], identities: Mapping[str, Any]) -> str:
    semantic = {k: v for k, v in config.items() if k not in RUN_CONTROL_KEYS}
    compatible = dict(identities)
    if "source" in compatible or "runtime_source" in compatible:
        runtime = compatible.get("runtime_source")
        if (
            not isinstance(runtime, dict)
            or runtime.get("schema_version") != 1
            or not isinstance(runtime.get("files"), dict)
            or not runtime["files"]
            or json_hash(runtime["files"]) != runtime.get("content_hash")
        ):
            raise ValueError("A valid runtime_source identity is required with source provenance")
        # Full source provenance remains in the checkpoint, inputs and exports.
        compatible.pop("source", None)
    return json_hash({"config": semantic, "identities": compatible, "schema_version": CHECKPOINT_SCHEMA})


def validate_precision(precision: str, device: torch.device) -> None:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA and GradScaler")
    if precision == "bf16" and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("bf16 autocast requires native support; unavailable on sm75/Turing")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pair_index_digest(indices: list[int]) -> str:
    """Hash ordered pair-manifest row indices without writing source frame IDs."""
    if not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("pair indices must be a list of nonnegative integers")
    return json_hash(indices)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    state: dict[str, Any],
    signature: str,
    scaler: Any = None,
    *,
    config: Mapping[str, Any],
    identities: Mapping[str, Any],
    resume_origin: Mapping[str, Any] | None = None,
) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    if state.get("microstep") != 0 or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Checkpoint requires an optimizer boundary: microstep=0 and no gradients")
    metadata = optimizer_metadata(optimizer)
    if metadata["parameter_state"] == "X":
        raise ValueError("Training checkpoint must contain Y, not evaluation X")
    if metadata["parameter_state"] == "Y" and any(g["k"] != state["successful_step"] for g in optimizer.param_groups):
        raise ValueError("Optimizer k and successful_step disagree")
    if checkpoint_signature(config, identities) != signature:
        raise ValueError("Checkpoint metadata signature mismatch")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "signature": signature,
        "config": _cpu_tree(dict(config)),
        "identities": _cpu_tree(dict(identities)),
        "resume_origin": _cpu_tree(dict(resume_origin)) if resume_origin is not None else None,
        "model": _cpu_tree(model.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "optimizer_metadata": metadata,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": _cpu_tree(state),
        "rng": capture_rng(),
        "parameter_state": metadata["parameter_state"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(
        path.with_suffix(path.suffix + ".json"),
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "file_sha256": file_sha256(path),
            "signature": signature,
            "successful_step": state["successful_step"],
            "parameter_state": metadata["parameter_state"],
        },
    )


def preflight_run(run_dir: Path, resume_from: Path | None, signature: str) -> dict[str, Any] | None:
    """Read and validate everything here without creating or mutating either run."""
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    if resume_from is None:
        return None
    sidecar = json.loads(resume_from.with_suffix(resume_from.suffix + ".json").read_text())
    if sidecar["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    if file_sha256(resume_from) != sidecar["file_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if sidecar["signature"] != signature:
        raise ValueError("Checkpoint signature mismatch")
    # Only explicitly named, checksum-verified local training checkpoints are accepted.
    checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA or checkpoint["signature"] != signature:
        raise ValueError("Checkpoint schema/signature mismatch")
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("identities"), dict):
        raise ValueError("Checkpoint is missing config/identities metadata")
    if checkpoint_signature(checkpoint["config"], checkpoint["identities"]) != signature:
        raise ValueError("Checkpoint metadata signature mismatch")
    if checkpoint["state"]["microstep"] != 0 or checkpoint["parameter_state"] not in {"Y", "standard"}:
        raise ValueError("Checkpoint is not a training optimizer boundary")
    return checkpoint


def restore_checkpoint(
    checkpoint: dict[str, Any], model: nn.Module, optimizer: Optimizer, scaler: Any = None
) -> dict[str, Any]:
    if (checkpoint["scaler"] is None) != (scaler is None):
        raise ValueError("Checkpoint scaler configuration mismatch")
    # Validate constructor/group identity before any parameter state is loaded.
    current, saved = optimizer_metadata(optimizer), checkpoint["optimizer_metadata"]
    if any(current.get(key) != saved.get(key) for key in ("spec", "groups", "constructor")):
        raise ValueError("Checkpoint optimizer signature mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    restore_optimizer(optimizer, checkpoint["optimizer"], saved)
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(checkpoint["rng"])
    return copy.deepcopy(checkpoint["state"])


class PairSampler:
    """Equal subset probability, then configured bin weight, then uniform pair."""

    def __init__(self, pairs: Mapping[str, Any], config: Mapping[str, Any], seed: int) -> None:
        self.pairs, self.config = pairs, config
        self.generator = np.random.default_rng(seed)
        self.subsets = sorted(np.unique(pairs["subset"]).tolist())
        self.cursor = 0
        self.history: list[int] = []

    def sample(self, step: int) -> int:
        subset = self.subsets[int(self.generator.integers(len(self.subsets)))]
        groups, weights = [], []
        for i, weight in enumerate(self.config["bin_weights"]):
            mask = (self.pairs["subset"] == subset) & (self.pairs["difficulty"] == f"overlap_{i}")
            if step < 100:
                mask &= self.pairs["overlap"] >= 0.5
            indices = np.flatnonzero(mask)
            if len(indices):
                groups.append(indices)
                weights.append(weight)
        if not groups:
            raise ValueError(f"No sampling/warmup candidates for {subset}")
        probability = np.asarray(weights, dtype=float)
        group = groups[int(self.generator.choice(len(groups), p=probability / probability.sum()))]
        index = int(group[int(self.generator.integers(len(group)))])
        self.history.append(index)
        self.cursor += 1
        return index

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": copy.deepcopy(self.generator.bit_generator.state),
            "cursor": self.cursor,
            "history": self.history.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.generator.bit_generator.state = copy.deepcopy(state["rng"])
        self.cursor, self.history = int(state["cursor"]), list(state["history"])


def train_loop(
    config: Mapping[str, Any], task: Any, optimizer: Optimizer, sampler: PairSampler, identities: Mapping[str, Any]
) -> dict[str, Any]:
    """One optimizer boundary owns evaluation, checkpointing and one log step."""
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter

    from xfeat_training.mining import repo_path

    cfg = dict(config)
    device = next(task.model.parameters()).device
    validate_precision(cfg["precision"], device)
    if cfg["accumulation"] < 1 or cfg["max_steps"] < 1 or cfg["save_every"] < 1 or cfg["eval_every"] < 1:
        raise ValueError("steps, accumulation and save/eval intervals must be positive")
    keep_best = cfg.get("checkpoint_keep_best")
    if keep_best is not None and (type(keep_best) is not int or keep_best < 1):
        raise ValueError("checkpoint_keep_best must be null or a positive integer")
    stop = cfg["max_steps"] if cfg.get("stop_after_steps") is None else cfg["stop_after_steps"]
    if not 0 < stop <= cfg["max_steps"]:
        raise ValueError("stop_after_steps must lie within max_steps")
    run_dir = repo_path(cfg["run_dir"])
    resume = repo_path(cfg["resume_from"]) if cfg.get("resume_from") else None
    signature = checkpoint_signature(cfg, identities)
    checkpoint = preflight_run(run_dir, resume, signature)
    resume_origin = (
        {
            "file_sha256": file_sha256(resume),
            "signature": checkpoint["signature"],
            "identities": checkpoint["identities"],
        }
        if checkpoint is not None and resume is not None
        else None
    )
    scaler = torch.amp.GradScaler("cuda") if cfg["precision"] == "fp16" else None
    configure_reproducibility(cfg["seed"], warn_only=cfg.get("deterministic_warn_only", False))
    state: dict[str, Any] = {
        "successful_step": 0,
        "attempted_microbatches": 0,
        "accepted_microbatches": 0,
        "microstep": 0,
        "skips": {},
        "sampler": sampler.state_dict(),
        "best": None,
        "best_model": None,
    }
    if checkpoint is not None:
        state = restore_checkpoint(checkpoint, task.model, optimizer, scaler)
        sampler.load_state_dict(state["sampler"])
    if state["successful_step"] >= stop:
        raise ValueError("Resume already reached requested stop/max_steps")
    # No filesystem output is created before all input/resume checks above.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "resolved.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True))
    atomic_json(
        run_dir / "inputs.json",
        {
            "signature": signature,
            "identities": dict(identities),
            "optimizer": optimizer_metadata(optimizer),
            "resume_origin": resume_origin,
        },
    )
    atomic_json(
        run_dir / "environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "precision": cfg["precision"],
            "attention_flash": cfg.get("matcher", {}).get("flash", False),
            "amuse_internal_ns": "bfloat16" if cfg["optimizer"]["name"] == "amuse" else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
    )
    if state["best_model"] is not None:
        assert resume is not None and state["best"] is not None
        with evaluation_parameters(task.model, optimizer):
            task.model.load_state_dict(state["best_model"], strict=True)
            task.export_origin = {
                "checkpoint": str(resume),
                "checkpoint_file_sha256": file_sha256(resume),
                "checkpoint_model_key": "state.best_model",
                "checkpoint_status": "complete",
                "checkpoint_parameter_state": "X" if cfg["optimizer"]["name"] == "amuse" else "standard",
            }
            try:
                task.export("best", state["best"]["step"])
            finally:
                del task.export_origin
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    invalid_streak, overflow_streak = 0, 0
    termination_reason = None
    try:
        with (run_dir / "metrics.jsonl").open("x") as stream:
            if cfg.get("evaluate_initial", False) and checkpoint is None:
                with evaluation_parameters(task.model, optimizer):
                    initial_metrics = task.evaluate(0)
                atomic_json(run_dir / "initial_validation.json", initial_metrics)
                for name, value in initial_metrics.items():
                    if type(value) in (int, float):
                        writer.add_scalar(f"validation/{name}", value, 0)
                writer.flush()
            while state["successful_step"] < stop:
                started = time.perf_counter()
                totals: dict[str, float] = {}
                sampled = []
                accepted = []
                if hasattr(task, "before_update"):
                    task.before_update(state["successful_step"], optimizer)
                task.model.train()
                from xfeat_training.objectives import freeze_batchnorm

                freeze_batchnorm(task.model)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                while state["microstep"] < cfg["accumulation"]:
                    pair_index = sampler.sample(state["successful_step"])
                    sampled.append(pair_index)
                    state["attempted_microbatches"] += 1
                    dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=cfg["precision"] != "fp32"):
                        losses = task.loss(pair_index, state["accepted_microbatches"])
                    if losses is None:
                        reason = getattr(task, "skip_reason", "no_supervision")
                        state["skips"][reason] = state["skips"].get(reason, 0) + 1
                        invalid_streak += 1
                        if invalid_streak >= 100:
                            raise RuntimeError(f"100 consecutive invalid microbatches: {reason}")
                        continue
                    invalid_streak = 0
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError("Nonfinite training loss")
                    accepted.append(pair_index)
                    scaled_loss = losses["loss"] / cfg["accumulation"]
                    if scaler is None:
                        scaled_loss.backward()
                    else:
                        scaler.scale(scaled_loss).backward()
                    for name, value in losses.items():
                        totals[name] = totals.get(name, 0.0) + float(value.detach()) / cfg["accumulation"]
                    state["accepted_microbatches"] += 1
                    state["microstep"] += 1
                gradient_names, gradient_values = [], []
                for name, parameter in task.model.named_parameters():
                    if parameter.grad is not None:
                        gradient_names.append(name)
                        gradient_values.append(parameter.grad.detach().float().norm())
                gradient_norms = (
                    dict(zip(gradient_names, torch.stack(gradient_values).cpu().tolist(), strict=True))
                    if gradient_values
                    else {}
                )
                update = finish_update(optimizer, max_norm=cfg["grad_clip"], scaler=scaler)
                state["microstep"] = 0
                if not update["performed"]:
                    overflow_streak += 1
                    state["skips"]["amp_overflow"] = state["skips"].get("amp_overflow", 0) + 1
                    if overflow_streak >= 100:
                        raise RuntimeError("100 consecutive AMP overflow updates")
                    continue
                overflow_streak = 0
                state["successful_step"] += 1
                step = state["successful_step"]
                final = step == stop
                metrics = None
                if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                    with evaluation_parameters(task.model, optimizer):
                        metrics = task.evaluate(step)
                        candidate = checkpoint_candidate(metrics, step, cfg)
                        if state["best"] is None or selection_rank(candidate) > selection_rank(state["best"]):
                            state["best"] = candidate
                            state["best_model"] = _cpu_tree(task.model.state_dict())
                            task.export("best", step)
                        task.export("last", step)
                elif final:
                    # A process stop must not add a best candidate absent from a continuous run.
                    with evaluation_parameters(task.model, optimizer):
                        task.export("last", step)
                state["sampler"] = sampler.state_dict()
                if cfg.get("auto_stop", {}).get("enabled", False):
                    history = state.setdefault("continuation_history", {"validation": [], "losses": []})
                    history["losses"] = (history["losses"] + [totals["loss"]])[-200:]
                    if metrics is not None:
                        history["validation"].append({key: value for key, value in metrics.items() if key != "rows"})
                    task.continuation_history = history
                checkpoint_written = step % cfg["save_every"] == 0 or metrics is not None or final
                if checkpoint_written:
                    path = run_dir / "checkpoints" / f"step_{step:06d}.pt"
                    save_checkpoint(
                        path,
                        task.model,
                        optimizer,
                        state,
                        signature,
                        scaler,
                        config=cfg,
                        identities=identities,
                        resume_origin=resume_origin,
                    )
                    for prefix in ("best", "last"):
                        manifest_path = run_dir / "exports" / f"{prefix}_manifest.json"
                        if manifest_path.is_file():
                            manifest = json.loads(manifest_path.read_text())
                            reference = Path(manifest["checkpoint"])
                            if not reference.is_absolute():
                                reference = run_dir / reference
                            if reference.resolve() == path.resolve():
                                manifest.update(checkpoint_file_sha256=file_sha256(path), checkpoint_status="complete")
                                atomic_json(manifest_path, manifest)
                    atomic_json(
                        run_dir / "run.json",
                        {
                            "last_checkpoint": str(path),
                            "best": state["best"],
                            "signature": signature,
                            "completed_steps": step,
                        },
                    )
                record = {
                    "step": step,
                    "time": time.time(),
                    "losses": totals,
                    "pair_indices": sampled,
                    "attempted_pair_digest": pair_index_digest(sampled),
                    "accepted_pair_digest": pair_index_digest(accepted),
                    "microbatch_skip_rate": (len(sampled) - len(accepted)) / len(sampled),
                    "cumulative_microbatch_skip_rate": (
                        state["attempted_microbatches"] - state["accepted_microbatches"]
                    )
                    / state["attempted_microbatches"],
                    "grad_norm": update["grad_norm"],
                    "parameter_gradient_norms": gradient_norms,
                    "attempted_microbatches": state["attempted_microbatches"],
                    "accepted_microbatches": state["accepted_microbatches"],
                    "skips": dict(state["skips"]),
                    "step_ms": (time.perf_counter() - started) * 1000,
                    "validation": metrics,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                for name, value in totals.items():
                    writer.add_scalar(f"loss/{name}", value, step)
                for name in (
                    "grad_norm",
                    "step_ms",
                    "peak_memory_bytes",
                    "attempted_microbatches",
                    "accepted_microbatches",
                ):
                    writer.add_scalar(name, record[name], step)
                for i, group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"optimizer/{i}/lr", group["lr"], step)
                    writer.add_scalar(
                        f"optimizer/{i}/beta1", getattr(optimizer, "beta1", group.get("betas", (0.0,))[0]), step
                    )
                for name, value in state["skips"].items():
                    writer.add_scalar(f"skips/{name}", value, step)
                if metrics is not None:
                    for name, value in metrics.items():
                        if type(value) in (int, float):
                            writer.add_scalar(f"validation/{name}", value, step)
                if (step == 1 or metrics is not None) and getattr(task, "preview", None) is not None:
                    writer.add_image("pair/source", task.preview.detach().cpu(), step)
                writer.flush()
                if checkpoint_written and keep_best is not None:
                    from xfeat_training.retention import prune_checkpoints

                    prune_checkpoints(run_dir, keep_best)
                if step == 1 or step % 10 == 0 or final:
                    print(
                        f"step {step}/{cfg['max_steps']} loss={totals['loss']:.5f} skipped={sum(state['skips'].values())}",
                        flush=True,
                    )
                if checkpoint_written and hasattr(task, "stop_reason"):
                    termination_reason = task.stop_reason(step)
                    if termination_reason is not None:
                        if not isinstance(termination_reason, str):
                            raise TypeError("stop_reason must return a string or None")
                        print(f"Stopped at optimizer boundary {step}: {termination_reason}", flush=True)
                        break
    except Exception as error:
        atomic_json(
            run_dir / "failure.json",
            {
                "error": str(error),
                "type": type(error).__name__,
                "successful_step": state["successful_step"],
                "microstep": state["microstep"],
                "attempted_microbatches": state["attempted_microbatches"],
                "recent_pair_indices": sampler.history[-cfg["accumulation"] :],
            },
        )
        raise
    finally:
        writer.close()
    completed = {"successful_step": state["successful_step"], "signature": signature}
    if hasattr(task, "stop_reason"):
        completed["stop_reason"] = termination_reason or (
            "max_steps" if stop == cfg["max_steps"] else "stop_after_steps"
        )
    atomic_json(run_dir / "completed.json", completed)
    return state


class TrainingTask:
    """The two explicit tasks share only run state and reporting."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from functools import lru_cache

        from modules.lighterglue import LighterGlue
        from modules.utils import state_hash
        from modules.xfeat import XFeat
        from xfeat_training.features import FeatureCache
        from xfeat_training.mining import (
            load_pairs,
            pair_dataset,
            pair_keys,
            repo_path,
            runtime_source_identity,
            source_identity,
        )

        self.config = dict(config)
        self.device = torch.device(config["device"])
        validate_precision(config["precision"], self.device)
        if config["task"] not in {"lighterglue", "xfeat"}:
            raise ValueError("task must be lighterglue or xfeat")
        if config["pair_split"] not in {"train", "smoke"}:
            raise ValueError("Training pair_split must be train/smoke; val/test are evaluation-only")
        if config["batch_size"] != 1 or config["num_workers"] != 0:
            raise ValueError("Variable-size features and exact resume require batch_size=1, num_workers=0")
        if config["top_k"] != 1024:
            raise ValueError("This run contract requires top_k=1024; a new baseline contract is required to change it")
        if config["matcher"]["width_confidence"] != -1 or config["matcher"]["depth_confidence"] != -1:
            raise ValueError("Training/evaluation require width/depth_confidence=-1")
        self.dataset, self.pair_manifest = pair_dataset(config["pairs_dir"], config.get("data_root"))
        self.pairs = load_pairs(repo_path(config["pairs_dir"]) / f"{config['pair_split']}.npz")
        if len(self.pairs["subset"]) == 0:
            raise ValueError("Training pair split is empty")
        self.frame = lru_cache(maxsize=64)(self.dataset.load_frame)
        self.extractor = XFeat(weights=repo_path(config["xfeat_weights"]), device=self.device, top_k=config["top_k"])
        self.matcher, self.teacher, self.cache = None, None, None
        extractor_hash = state_hash(self.extractor.net.state_dict())
        self.identities = {
            "pair_hash": self.pair_manifest["content_hash"],
            "source_weights": {"extractor": extractor_hash},
            "source": source_identity(),
            "runtime_source": runtime_source_identity(),
        }
        if config["task"] == "lighterglue":
            self.matcher = LighterGlue(weights=repo_path(config["lg_weights"]), device=self.device, **config["matcher"])
            if not config["init_matcher_only"] and self.matcher.extractor_hash != extractor_hash:
                raise ValueError(
                    "Initial bundle extractor differs; explicitly select init_matcher_only for fresh initialization"
                )
            self.extractor.net.requires_grad_(False).eval()
            self.model = self.matcher.net
            self.cache = FeatureCache(self.dataset, config["cache_dir"], self.extractor, top_k=config["top_k"])
            keys = {key for i in range(len(self.pairs["subset"])) for key in pair_keys(self.pairs, i)}
            if not self.cache.root.exists() or any(key.name() not in self.cache.frames for key in keys):
                raise ValueError("Feature cache is incomplete for the training split")
            self.identities["cache_hash"] = self.cache.identity_hash
            self.identities["cache_manifest_sha256"] = file_sha256(self.cache.root / "manifest.json")
            self.identities["source_weights"]["matcher"] = state_hash(self.model.state_dict())
        else:
            self.model = self.extractor.net
            self.teacher = XFeat(weights=repo_path(config["teacher_weights"]), device=self.device).net
            self.teacher.requires_grad_(False).eval()
            self.identities["source_weights"]["teacher"] = state_hash(self.teacher.state_dict())
        eval_manifest = json.loads((repo_path(config["eval_cache_dir"]) / "manifest.json").read_text())
        if json_hash(eval_manifest["identity"]) != eval_manifest["content_hash"]:
            raise ValueError("Evaluation anchor manifest content hash mismatch")
        if eval_manifest["identity"]["pair_hash"] != self.pair_manifest["content_hash"]:
            raise ValueError("Evaluation anchor cache belongs to different pairs")
        if eval_manifest["identity"]["top_k"] != config["top_k"]:
            raise ValueError("Evaluation anchor top_k mismatch")
        self.identities["evaluation_hash"] = eval_manifest["content_hash"]
        self.preview = None
        self.skip_reason = "no_supervision"

    def loss(self, pair_index: int, accepted_microbatches: int) -> dict[str, torch.Tensor] | None:
        from xfeat_training.augment import photometric, sample_homography, warp_image
        from xfeat_training.geometry import matching_labels
        from xfeat_training.lightglue import forward_layers, lightglue_loss
        from xfeat_training.mining import pair_keys
        from xfeat_training.objectives import rgbd_correspondences, synthetic_correspondences, xfeat_objective

        key0, key1 = pair_keys(self.pairs, pair_index)
        frame0, frame1 = self.frame(key0), self.frame(key1)
        self.preview = frame0.rgb
        if self.config["task"] == "lighterglue":
            assert self.cache is not None
            features0, features1 = self.cache.get(key0), self.cache.get(key1)
            if not len(features0["keypoints"]) or not len(features1["keypoints"]):
                self.skip_reason = "empty_keypoints"
                return None
            labels = matching_labels(features0["keypoints"].numpy(), features1["keypoints"].numpy(), frame0, frame1)
            if not ((labels.matches0 != -2).any() or (labels.matches1 != -2).any()):
                self.skip_reason = "all_gt_ignored"
                return None
            data = {}
            for name, features in (("image0", features0), ("image1", features1)):
                data[name] = {key: features[key].to(self.device)[None] for key in ("keypoints", "descriptors")}
                data[name]["image_size"] = torch.tensor(features["image_size"], device=self.device)[None]
            return lightglue_loss(
                self.model,
                forward_layers(self.model, data),
                torch.from_numpy(labels.matches0).to(self.device),
                torch.from_numpy(labels.matches1).to(self.device),
            )
        assert self.teacher is not None
        image0 = frame0.rgb[None].to(self.device)
        synthetic = accepted_microbatches % 2 == 1
        valid1 = None
        if synthetic:
            height, width = frame0.depth.shape
            homography = sample_homography(height, width)
            image1, valid1 = warp_image(image0, homography)
            correspondence = synthetic_correspondences(homography, height, width)
        else:
            image1 = frame1.rgb[None].to(self.device)
            correspondence = rgbd_correspondences(frame0, frame1)
        self.skip_reason = "fewer_than_two_correspondences"
        result = xfeat_objective(
            self.extractor.net,
            self.teacher,
            image0,
            image1,
            photometric(image0),
            photometric(image1),
            correspondence,
            synthetic=synthetic,
            valid1=valid1,
        )
        if result is not None:
            result["synthetic"] = torch.tensor(float(synthetic), device=self.device)
        return result

    def evaluate(self, step: int) -> dict[str, Any]:
        from xfeat_training.evaluate import evaluate_task

        return evaluate_task(self, step)

    def export(self, prefix: str, step: int) -> None:
        from xfeat_training.evaluate import export_task

        export_task(self, prefix, step)


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    from xfeat_training.optim import build_optimizer

    configure_reproducibility(config["seed"])
    task = TrainingTask(config)
    optimizer = build_optimizer(task.model, config["task"], config["optimizer"])
    sampler = PairSampler(task.pairs, task.pair_manifest["resolved_config"], config["seed"])
    return train_loop(config, task, optimizer, sampler, task.identities)
