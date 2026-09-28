"""Independent head tasks on synthetic views of the existing L76 training split."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from modules.raco import XFeatRaCo, resize_input, sample_map
from modules.utils import state_hash
from xfeat_training.convergence import plateau_decision
from xfeat_training.data import FrameKey, file_sha256, json_hash
from xfeat_training.mining import (
    load_pairs,
    pair_dataset,
    pair_keys,
    repo_path,
    runtime_source_identity,
    source_identity,
)
from xfeat_training.optim import build_optimizer
from xfeat_training.raco_augment import synthetic_views
from xfeat_training.raco_objectives import (
    covariance_nll,
    geometric_matches,
    hard_utility,
    project_with_jacobian,
    ranking_loss,
)
from xfeat_training.trainer import PairSampler, atomic_json, configure_reproducibility, train_loop


def validate_config(config: Mapping[str, Any]) -> None:
    if config["pair_split"] not in {"train", "smoke"}:
        raise ValueError("RaCo training requires train/smoke split")
    if config["task"] not in {"raco_rank", "raco_covariance"}:
        raise ValueError("Unknown RaCo task")
    if config["precision"] != "fp32" or config["batch_size"] != 1 or config["num_workers"] != 0:
        raise ValueError("RaCo contract requires fp32, batch_size=1, num_workers=0")
    if config["optimizer"]["name"] != "adamw":
        raise ValueError("RaCo requires AdamW")
    for key in ("max_steps", "save_every", "eval_every", "val_frames", "accumulation", "candidate_limit"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("temperature_start", "temperature_end", "rank_threshold", "covariance_threshold", "final_lr"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not config["budgets"] or any(type(k) is not int or k <= 0 for k in config["budgets"]):
        raise ValueError("budgets must be positive integers")
    if len(set(config["budgets"])) != len(config["budgets"]):
        raise ValueError("Duplicate budgets are not supported")
    if not config["covariance_diagnostics"] or any(
        not math.isfinite(t) or t <= 0 for t in config["covariance_diagnostics"]
    ):
        raise ValueError("Covariance diagnostics thresholds must be finite and positive")
    size = config["image_size"]
    if size is not None and (len(size) != 2 or any(type(s) is not int or s < 32 or s % 32 for s in size)):
        raise ValueError("image_size must be [height,width] multiples of 32")
    aug = config["augmentation"]
    for key in ("scale", "gamma"):
        if len(aug[key]) != 2 or not 0 < aug[key][0] <= aug[key][1] or not all(math.isfinite(x) for x in aug[key]):
            raise ValueError(f"Invalid augmentation {key}")
    for key in ("rotation_small_probability", "blur_probability", "minimum_overlap"):
        if not math.isfinite(aug[key]) or not 0 <= aug[key] <= 1:
            raise ValueError(f"Invalid augmentation {key}")
    for key in ("rotation_small_degrees", "translation", "shear", "projective", "brightness", "noise"):
        if not math.isfinite(aug[key]) or aug[key] < 0:
            raise ValueError(f"Invalid augmentation {key}")
    stopping = config["auto_stop"]
    if type(stopping["enabled"]) is not bool:
        raise ValueError("auto_stop.enabled must be boolean")
    if stopping["enabled"]:
        if type(stopping["interval"]) is not int or type(stopping["window"]) is not int or stopping["window"] < 1:
            raise ValueError("Invalid auto_stop interval/window")
        if (
            stopping["interval"] < 2 * stopping["window"] * config["eval_every"]
            or stopping["interval"] % config["eval_every"]
        ):
            raise ValueError("auto_stop interval must align with eval and include two windows")
        for key in ("rank_minimum_gain", "covariance_minimum_gain"):
            if not math.isfinite(stopping[key]) or stopping[key] < 0:
                raise ValueError("auto_stop gains must be finite and nonnegative")


def _subset_frames(pairs: Mapping[str, Any], count: int) -> list[FrameKey]:
    all_keys = sorted({key for i in range(len(pairs["subset"])) for key in pair_keys(pairs, i)})
    subsets = sorted({key.subset for key in all_keys})
    if not subsets:
        raise ValueError("Empty validation split")
    if count < len(subsets) or count % len(subsets):
        raise ValueError("val_frames must be a positive multiple of subset count")
    result = []
    for subset in subsets:
        keys = [key for key in all_keys if key.subset == subset]
        n = count // len(subsets)
        if len(keys) < n:
            raise ValueError("Not enough distinct validation frames")
        result.extend(keys[i] for i in np.linspace(0, len(keys) - 1, n).astype(int))
    return result


class RacoTask:
    def __init__(self, config: Mapping[str, Any]) -> None:
        validate_config(config)
        self.config = dict(config)
        self.device = torch.device(config["device"])
        self.active = "rank" if config["task"] == "raco_rank" else "covariance"
        self.dataset, self.pair_manifest = pair_dataset(config["pairs_dir"], config.get("data_root"))
        self.pairs = load_pairs(repo_path(config["pairs_dir"]) / f"{config['pair_split']}.npz")
        if not len(self.pairs["subset"]):
            raise ValueError("Empty training split")
        validation = load_pairs(repo_path(config["pairs_dir"]) / "val.npz")
        self.val_keys = _subset_frames(validation, config["val_frames"])
        train_keys = {key for i in range(len(self.pairs["subset"])) for key in pair_keys(self.pairs, i)}
        if train_keys & set(self.val_keys):
            raise ValueError("Training and validation source frames overlap")
        self.frame = lru_cache(maxsize=32)(self.dataset.load_frame)
        settings = {key: config[key] for key in ("candidate_limit", "detection_threshold", "border", "sigma_min")}
        self.model = XFeatRaCo(weights=repo_path(config["xfeat_weights"]), device=self.device, **settings)
        self.base_hash = state_hash(self.model.net.state_dict())
        warm_start_report = self.model.warm_start_report
        if config.get("init_bundle"):
            loaded = XFeatRaCo.from_bundle(
                repo_path(config["init_bundle"]), device=self.device, ranking=False, covariance=False
            )
            if loaded.bundle(trained_heads=[])["config"] != settings:
                raise ValueError("Initial head bundle extraction configuration mismatch")
            if state_hash(loaded.net.state_dict()) != self.base_hash:
                raise ValueError("Initial head bundle frozen extractor mismatch")
            self.model = loaded
        self.model.heads.requires_grad_(False)
        active = self.model.heads.ranker if self.active == "rank" else self.model.heads.covariance_head
        active.requires_grad_(True)
        self.inactive = self.model.heads.covariance_head if self.active == "rank" else self.model.heads.ranker
        self.inactive_hash = state_hash(self.inactive.state_dict())
        self.trained_heads = self.model.trained_heads | {self.active}
        validation_identity = {
            "frames": [key.name() for key in self.val_keys],
            "seed": config["val_seed"],
            "augmentation": config["augmentation"],
            "settings": settings,
            "image_size": config["image_size"],
            "budgets": config["budgets"],
            "rank_threshold": config["rank_threshold"],
            "covariance_threshold": config["covariance_threshold"],
            "covariance_diagnostics": config["covariance_diagnostics"],
            "base_hash": self.base_hash,
            "coordinate_convention": "integer pixel centers; half-pixel resize; original pixel^2",
        }
        self.identities: dict[str, Any] = {
            "pair_hash": self.pair_manifest["content_hash"],
            "evaluation_hash": json_hash(validation_identity),
            "evaluation": validation_identity,
            "source_weights": {"extractor": self.base_hash},
            "warm_start": warm_start_report,
            "init_bundle_sha256": file_sha256(repo_path(config["init_bundle"])) if config.get("init_bundle") else None,
            "source": source_identity(),
            "runtime_source": runtime_source_identity(),
        }
        self.successful_step = 0
        self.temperature = config["temperature_start"]
        self.preview = None
        self.skip_reason = "no_supervision"
        self.continuation_history: dict[str, Any] = {"validation": [], "losses": []}

    def before_update(self, step: int, optimizer: Optimizer) -> None:
        self.successful_step = step
        fraction = step / max(1, self.config["max_steps"] - 1)
        initial, final = self.config["optimizer"]["lr"], self.config["final_lr"]
        lr = final + 0.5 * (initial - final) * (1 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = lr
        self.temperature = self.config["temperature_start"] + fraction * (
            self.config["temperature_end"] - self.config["temperature_start"]
        )

    def stop_reason(self, step: int) -> str | None:
        stopping = self.config["auto_stop"]
        if not stopping["enabled"] or step % stopping["interval"]:
            return None
        root = repo_path(self.config["run_dir"])
        metric = self.config["selection_metric"]
        validation = self.continuation_history["validation"]
        if not validation or validation[-1]["step"] != step:
            raise ValueError("Missing validation at continuation boundary")
        for value in validation:
            if value["evaluation_hash"] != self.identities["evaluation_hash"] or value["skipped_frames"]:
                raise ValueError("Continuation requires the same complete fixed validation set")
        threshold = stopping["rank_minimum_gain"] if self.active == "rank" else stopping["covariance_minimum_gain"]
        decision = plateau_decision(
            [v[metric["name"]] for v in validation],
            mode=metric["mode"],
            window=stopping["window"],
            minimum_gain=threshold,
        )
        losses = self.continuation_history["losses"]
        if len(losses) < 200:
            raise ValueError("Continuation diagnostics require at least 200 logged updates")
        decision.update(
            step=step,
            metric=metric["name"],
            evaluation_hash=self.identities["evaluation_hash"],
            validation_steps=[v["step"] for v in validation[-2 * stopping["window"] :]],
            previous_loss_mean=sum(losses[-200:-100]) / 100,
            recent_loss_mean=sum(losses[-100:]) / 100,
            interpretation="fixed 4-frame validation plateau heuristic, not global convergence",
        )
        atomic_json(root / "convergence" / f"step_{step:06d}.json", decision)
        print(
            f"Continuation check {step}: {decision['reason']} mean_gain={decision['mean_gain']:.6g} peak_gain={decision['peak_gain']:.6g}",
            flush=True,
        )
        return decision["reason"] if decision["stop"] else None

    def _pair(self, key: FrameKey, generator: torch.Generator | None = None):
        image = self.frame(key).rgb[None].to(self.device)
        if self.config["image_size"] is not None:
            image = F.interpolate(image, size=self.config["image_size"], mode="bilinear", align_corners=False)
        image, _ = resize_input(image)
        a, b, mask0, mask1, h, overlap = synthetic_views(image, self.config["augmentation"], generator)
        self.preview = a[0].detach()
        if overlap < self.config["augmentation"]["minimum_overlap"]:
            self.skip_reason = "insufficient_shared_support"
            return None
        c0, c1 = self._make_candidates(a, mask0), self._make_candidates(b, mask1)
        if not len(c0["keypoints"]) or not len(c1["keypoints"]):
            self.skip_reason = "empty_candidates"
            return None
        return c0, c1, h, overlap

    def _make_candidates(self, image: Tensor, support: Tensor) -> dict[str, Tensor]:
        return self.model.candidates(image, support)[0]

    def _evaluation_extras(self, a: dict[str, Tensor], b: dict[str, Tensor], h: Tensor) -> dict[str, Any]:
        return {}

    def _finalize_evaluation_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Allow task-specific aggregates before publishing validation metrics."""
        return metrics

    def _matches(self, a: dict[str, Tensor], b: dict[str, Tensor], h: Tensor, threshold: float) -> Tensor:
        x, y = a["keypoints"].float(), b["keypoints"].float()
        matches = geometric_matches(x, y, h, threshold)
        if not len(matches):
            return matches
        forward, _ = project_with_jacobian(x[matches[:, 0]], h)
        reverse, _ = project_with_jacobian(y[matches[:, 1]], torch.linalg.inv(h))
        height, width = a["support"].shape[-2:]
        valid1 = sample_map(b["support"].float(), forward, height, width).squeeze(-1) > 0.999
        valid0 = sample_map(a["support"].float(), reverse, height, width).squeeze(-1) > 0.999
        return matches[valid0 & valid1]

    def _rank(self, candidate: dict[str, Tensor]) -> Tensor:
        return self.model.heads.rank(candidate["z"], candidate["keypoints"], candidate["scores"])

    def _cov(self, candidate: dict[str, Tensor]) -> Tensor:
        return self.model.heads.covariance(candidate["z"], candidate["keypoints"])

    def loss(self, pair_index: int, accepted_microbatches: int) -> dict[str, Tensor] | None:
        del accepted_microbatches
        pair = self._pair(pair_keys(self.pairs, pair_index)[0])
        if pair is None:
            return None
        a, b, h, overlap = pair
        threshold = self.config["rank_threshold"] if self.active == "rank" else self.config["covariance_threshold"]
        matches = self._matches(a, b, h, threshold)
        if not len(matches):
            self.skip_reason = "no_geometric_pseudo_matches"
            return None
        if self.active == "rank":
            # Entire valid candidate pools compete, including points outside covisibility.
            loss = ranking_loss(self._rank(a), self._rank(b), matches, self.config["budgets"], self.temperature)
        else:
            loss, _ = covariance_nll(
                a["keypoints"].float(), b["keypoints"].float(), self._cov(a), self._cov(b), matches, h
            )
        return {
            "loss": loss,
            "pseudo_matches": loss.new_tensor(len(matches)),
            "overlap": loss.new_tensor(overlap),
            "temperature": loss.new_tensor(self.temperature),
        }

    def _check_frozen(self) -> None:
        if (
            state_hash(self.model.net.state_dict()) != self.base_hash
            or state_hash(self.inactive.state_dict()) != self.inactive_hash
        ):
            raise RuntimeError("Frozen XFeat/BN or inactive head changed")

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, Any]:
        self._check_frozen()
        generator = torch.Generator().manual_seed(self.config["val_seed"])
        rows, arrays = [], {}
        output = repo_path(self.config["run_dir"]) / "validation" / f"step_{step:06d}"
        for index, key in enumerate(self.val_keys):
            pair = self._pair(key, generator)
            if pair is None:
                rows.append({"frame": key.name(), "skip": self.skip_reason})
                continue
            a, b, h, overlap = pair
            scores0, scores1 = self._rank(a), self._rank(b)
            cov0, cov1 = self._cov(a), self._cov(b)
            matches = self._matches(a, b, h, self.config["rank_threshold"])
            row: dict[str, Any] = {
                "frame": key.name(),
                "overlap": overlap,
                "candidate_counts": [len(scores0), len(scores1)],
                "candidate_matches": len(matches),
                "rank_utility": hard_utility(scores0, scores1, matches, self.config["budgets"]),
                "baseline_utility": hard_utility(a["scores"], b["scores"], matches, self.config["budgets"]),
                "budgets": {},
                "covariance": {},
            }
            for k in self.config["budgets"]:
                order0 = torch.argsort(scores0, descending=True, stable=True)[:k]
                order1 = torch.argsort(scores1, descending=True, stable=True)[:k]
                similarities = a["descriptors"][order0] @ b["descriptors"][order1].T
                nn0, nn1 = similarities.argmax(1), similarities.argmax(0)
                ids = torch.arange(len(order0), device=self.device)
                mutual = nn1[nn0] == ids
                coords0, coords1 = a["keypoints"][order0[mutual]].float(), b["keypoints"][order1[nn0[mutual]]].float()
                projected, _ = project_with_jacobian(coords0, h)
                reverse, _ = project_with_jacobian(coords1, torch.linalg.inv(h))
                correct = ((projected - coords1).norm(dim=-1) <= self.config["rank_threshold"]) & (
                    (reverse - coords0).norm(dim=-1) <= self.config["rank_threshold"]
                )
                row["budgets"][str(k)] = {
                    "utility": hard_utility(scores0, scores1, matches, [k]),
                    "baseline_utility": hard_utility(a["scores"], b["scores"], matches, [k]),
                    "descriptor_matches": int(mutual.sum()),
                    "descriptor_TP": int(correct.sum()),
                }
            thresholds = sorted(set(self.config["covariance_diagnostics"] + [self.config["covariance_threshold"]]))
            for threshold in thresholds:
                cm = self._matches(a, b, h, threshold)
                if not len(cm):
                    row["covariance"][str(threshold)] = {"matches": 0}
                    continue
                nll, stats = covariance_nll(a["keypoints"].float(), b["keypoints"].float(), cov0, cov1, cm, h)
                row["covariance"][str(threshold)] = {"matches": len(cm), "nll": float(nll), **stats}
                if threshold == self.config["covariance_threshold"]:
                    row["covariance_nll"] = float(nll)
                    projected, _ = project_with_jacobian(a["keypoints"][cm[:, 0]].float(), h)
                    error = b["keypoints"][cm[:, 1]].float() - projected
                    phase = a["keypoints"][cm[:, 0]] % 8
                    row["pixel_phase"] = {
                        str(p): {
                            "count": int(mask.sum()),
                            "mean_squared_residual": float(error[mask].square().sum(-1).mean()),
                        }
                        for p in range(64)
                        if (mask := (phase[:, 1] * 8 + phase[:, 0] == p)).any()
                    }
            for side, c, scores, cov in ((0, a, scores0, cov0), (1, b, scores1, cov1)):
                for name, value in {
                    "keypoints": c["keypoints"],
                    "scores": c["scores"],
                    "descriptors": c["descriptors"],
                    "ranker_scores": scores,
                    "covariances": cov,
                }.items():
                    arrays[f"{index}_{side}_{name}"] = value.cpu().numpy()
            arrays[f"{index}_homography"] = h.cpu().numpy()
            row.update(self._evaluation_extras(a, b, h))
            rows.append(row)
        usable = [row for row in rows if "skip" not in row]
        cov_rows = [row for row in usable if "covariance_nll" in row]
        if not usable or not cov_rows:
            raise ValueError("Fixed validation has no usable rank/covariance supervision")
        metrics = {
            "split": "val",
            "step": step,
            "evaluation_hash": self.identities["evaluation_hash"],
            "pair_hash": self.identities["pair_hash"],
            "rank_utility": sum(r["rank_utility"] for r in usable) / len(usable),
            "baseline_utility": sum(r["baseline_utility"] for r in usable) / len(usable),
            "covariance_nll": sum(r["covariance_nll"] for r in cov_rows) / len(cov_rows),
            "frames": len(rows),
            "skipped_frames": len(rows) - len(usable),
            "covariance_frames": len(cov_rows),
            "rows": rows,
            "interpretation": "conditional on fixed geometric pseudo-correspondence selection; not calibrated uncertainty",
        }
        metrics = self._finalize_evaluation_metrics(metrics)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "metrics.json", metrics)
        with (output / "predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        self._check_frozen()
        return metrics

    def export(self, prefix: str, step: int) -> None:
        self._check_frozen()
        output = repo_path(self.config["run_dir"]) / "exports"
        output.mkdir(exist_ok=True)
        path = output / f"{prefix}_raco.pt"
        temporary = path.with_suffix(".pt.tmp")
        with temporary.open("wb") as stream:
            torch.save(self.model.bundle(trained_heads=sorted(self.trained_heads)), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        atomic_json(
            output / f"{prefix}_manifest.json",
            {
                "schema_version": 1,
                "step": step,
                "parameter_state": "standard",
                "trained_heads": sorted(self.trained_heads),
                "files": {path.name: file_sha256(path)},
                "extractor_state_hash": self.base_hash,
                "model_state_hash": state_hash(self.model.state_dict()),
                "identities": self.identities,
                **getattr(
                    self,
                    "export_origin",
                    {
                        "checkpoint": f"checkpoints/step_{step:06d}.pt",
                        "checkpoint_model_key": "model",
                        "checkpoint_parameter_state": "standard",
                        "checkpoint_file_sha256": None,
                        "checkpoint_status": "pending",
                    },
                ),
            },
        )


def run_raco_training(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    validate_config(cfg)
    if repo_path(cfg["run_dir"]).exists():
        raise FileExistsError(f"Run directory already exists: {cfg['run_dir']}")
    selection = (
        {"name": "rank_utility", "mode": "max"}
        if cfg["task"] == "raco_rank"
        else {"name": "covariance_nll", "mode": "min"}
    )
    if cfg.get("selection_metric") is not None and cfg["selection_metric"] != selection:
        raise ValueError("Selection metric does not match independent head task")
    cfg["selection_metric"] = selection
    configure_reproducibility(cfg["seed"])
    task = RacoTask(cfg)
    optimizer = build_optimizer(task.model, cfg["task"], cfg["optimizer"])
    sampler = PairSampler(task.pairs, task.pair_manifest["resolved_config"], cfg["seed"])
    return train_loop(cfg, task, optimizer, sampler, task.identities)
