"""Separate rank KD, covariance KD, and student-error adaptation stages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import mean
from typing import Any

import torch
from torch import Tensor

from xfeat_training.convergence import plateau_decision
from xfeat_training.data import json_hash
from xfeat_training.mining import pair_keys, repo_path
from xfeat_training.optim import build_optimizer
from xfeat_training.raco_distillation import (
    covariance_distance,
    gaussian_kl,
    rank_list_kl,
    rank_pair_kl,
    teacher_weighted_mean,
)
from xfeat_training.raco_objectives import covariance_nll, hard_utility, project_with_jacobian
from xfeat_training.raco_task import RacoTask, validate_config
from xfeat_training.raco_teacher import OfficialRacoTeacher
from xfeat_training.trainer import PairSampler, atomic_json, configure_reproducibility, train_loop


def validate_distillation(config: Mapping[str, Any]) -> None:
    validate_config(config)
    kd = config["distillation"]
    if not config.get("init_bundle"):
        raise ValueError("Distillation requires an explicit existing student init_bundle")
    if kd["direction"] not in {"teacher_to_student", "student_to_teacher", "symmetric", "mse"}:
        raise ValueError("Unknown covariance KD direction")
    if kd["transform"] not in {"identity", "log1p", "bounded_log1p"}:
        raise ValueError("Unknown covariance KD transformation")
    if kd["direction"] == "mse" and kd["transform"] != "identity":
        raise ValueError("MSE control requires the identity transform")
    for key in ("comparisons", "rank_budget", "boundary_radius"):
        if type(kd[key]) is not int or kd[key] < 1:
            raise ValueError(f"distillation.{key} must be a positive integer")
    if kd["comparisons"] % 2:
        raise ValueError("Rank comparison count must be even")
    for key in ("minimum_absolute_gain", "minimum_relative_gain"):
        if not math.isfinite(kd[key]) or kd[key] < 0:
            raise ValueError(f"Invalid {key}")
    geometry = kd["geometry"]
    if type(geometry["enabled"]) is not bool:
        raise ValueError("geometry.enabled must be a boolean")
    if geometry["enabled"] and config["task"] != "raco_covariance":
        raise ValueError("Geometry adaptation is a covariance-only stage")
    if type(geometry["ramp_steps"]) is not int or geometry["ramp_steps"] < 1:
        raise ValueError("Geometry ramp_steps must be a positive integer")
    if not math.isfinite(geometry["final_kd_weight"]) or not 0 <= geometry["final_kd_weight"] <= 1:
        raise ValueError("Geometry final_kd_weight must lie in [0,1]")


def _coverage(a: Tensor, b: Tensor, cov0: Tensor, cov1: Tensor, matches: Tensor, h: Tensor) -> dict[str, float]:
    """Bidirectional empirical coverage on the fixed, truncated match population."""
    i, j = matches.unbind(1)

    def mahalanobis(x: Tensor, y: Tensor, c0: Tensor, c1: Tensor, transform: Tensor) -> Tensor:
        projected, jacobian = project_with_jacobian(x, transform)
        covariance = c1 + jacobian @ c0 @ jacobian.mT
        factor = torch.linalg.cholesky(covariance)
        whitened = torch.linalg.solve_triangular(factor, (y - projected)[..., None], upper=False)
        return whitened.square().sum((-2, -1))

    q = torch.cat(
        (
            mahalanobis(a[i], b[j], cov0[i], cov1[j], h),
            mahalanobis(b[j], a[i], cov1[j], cov0[i], torch.linalg.inv(h)),
        )
    )
    return {f"coverage{int(100 * p)}": float((q <= -2 * math.log1p(-p)).float().mean()) for p in (0.5, 0.9, 0.95)}


class DistillationTask(RacoTask):
    def __init__(self, config: Mapping[str, Any]) -> None:
        validate_distillation(config)
        super().__init__(config)
        self.kd = config["distillation"]
        self.teacher = OfficialRacoTeacher(
            repo_path(self.kd["teacher_source"]), repo_path(self.kd["teacher_weights"]), self.device
        )
        self.identities["teacher"] = self.teacher.identity
        self.identities["evaluation"]["distillation"] = {
            key: value for key, value in self.kd.items() if key not in {"teacher_source", "teacher_weights"}
        }
        self.identities["evaluation"]["teacher"] = self.teacher.identity
        self.identities["evaluation_hash"] = json_hash(self.identities["evaluation"])
        self.evaluation_index = 0

    def _check_frozen(self) -> None:
        super()._check_frozen()
        self.teacher.check_frozen()

    def _make_candidates(self, image: Tensor, support: Tensor) -> dict[str, Tensor]:
        candidates = super()._make_candidates(image, support)
        if len(candidates["keypoints"]):
            targets = self.teacher.at(image, candidates["keypoints"][None].float())
            candidates.update({f"teacher_{name}": value[0] for name, value in targets.items()})
            floor = self.config["sigma_min"] ** 2 * torch.eye(2, device=image.device, dtype=image.dtype)
            candidates["teacher_covariance"] = candidates["teacher_covariance"] + floor
        return candidates

    def _rank_kd(self, candidate: dict[str, Tensor], generator: torch.Generator | None = None) -> Tensor:
        return rank_pair_kl(
            self._rank(candidate),
            candidate["teacher_rank"],
            comparisons=self.kd["comparisons"],
            budget=self.kd["rank_budget"],
            boundary_radius=self.kd["boundary_radius"],
            generator=generator,
        )

    def _cov_kd(self, candidate: dict[str, Tensor]) -> Tensor:
        per_point = covariance_distance(
            self._cov(candidate),
            candidate["teacher_covariance"],
            direction=self.kd["direction"],
            transform=self.kd["transform"],
        )
        return teacher_weighted_mean(per_point, candidate["teacher_detector_logits"])

    def loss(self, pair_index: int, accepted_microbatches: int) -> dict[str, Tensor] | None:
        del accepted_microbatches
        pair = self._pair(pair_keys(self.pairs, pair_index)[0])
        if pair is None:
            return None
        a, b, h, overlap = pair
        try:
            kd = (
                0.5 * (self._rank_kd(a) + self._rank_kd(b))
                if self.active == "rank"
                else 0.5 * (self._cov_kd(a) + self._cov_kd(b))
            )
            geometry = kd.new_zeros(())
            geometry_weight, kd_weight = 0.0, 1.0
            if self.kd["geometry"]["enabled"]:
                matches = self._matches(a, b, h, self.config["covariance_threshold"])
                if not len(matches):
                    self.skip_reason = "no_geometric_pseudo_matches"
                    return None
                geometry, _ = covariance_nll(
                    a["keypoints"].float(), b["keypoints"].float(), self._cov(a), self._cov(b), matches, h
                )
                geometry_weight = min(self.successful_step / self.kd["geometry"]["ramp_steps"], 1.0)
                kd_weight = 1 - (1 - self.kd["geometry"]["final_kd_weight"]) * geometry_weight
            return {
                "loss": kd_weight * kd + geometry_weight * geometry,
                "kd_loss": kd,
                "geometry_nll": geometry,
                "geometry_weight": kd.new_tensor(geometry_weight),
                "kd_weight": kd.new_tensor(kd_weight),
                "overlap": kd.new_tensor(overlap),
            }
        except Exception:
            torch.save(
                {"a": a, "b": b, "homography": h, "pair_index": pair_index},
                repo_path(self.config["run_dir"]) / "failed_kd_input.pt",
            )
            raise

    @torch.no_grad()
    def _evaluation_extras(self, a: dict[str, Tensor], b: dict[str, Tensor], h: Tensor) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(self.config["val_seed"] + self.evaluation_index)
        self.evaluation_index += 1
        rows = []
        for c in (a, b):
            rank, cov = self._rank(c), self._cov(c)
            per_point = gaussian_kl(c["teacher_covariance"], cov)
            student_logdet, teacher_logdet = (
                torch.linalg.slogdet(cov).logabsdet,
                torch.linalg.slogdet(c["teacher_covariance"]).logabsdet,
            )
            k = min(self.kd["rank_budget"], len(rank))
            si, ti = (
                rank.argsort(descending=True, stable=True)[:k],
                c["teacher_rank"].argsort(descending=True, stable=True)[:k],
            )
            rows.append(
                {
                    "rank_kd": float(self._rank_kd(c, generator)),
                    "rank_list_kl": float(rank_list_kl(rank, c["teacher_rank"])),
                    "topk_teacher_overlap": float(torch.isin(si, ti).float().mean()),
                    "covariance_kd": float(self._cov_kd(c)),
                    "covariance_forward_kl": float(teacher_weighted_mean(per_point, c["teacher_detector_logits"])),
                    "covariance_kl_uniform": float(per_point.mean()),
                    "covariance_kl_median": float(per_point.median()),
                    "covariance_kl_p95": float(torch.quantile(per_point, 0.95)),
                    "logdet_absolute_error": float((student_logdet - teacher_logdet).abs().mean()),
                }
            )
        metrics: dict[str, Any] = {name: mean(row[name] for row in rows) for name in rows[0]}
        rank_matches = self._matches(a, b, h, self.config["rank_threshold"])
        metrics["teacher_rank_utility"] = hard_utility(
            a["teacher_rank"], b["teacher_rank"], rank_matches, self.config["budgets"]
        )
        matches = self._matches(a, b, h, self.config["covariance_threshold"])
        if len(matches):
            x, y = a["keypoints"].float(), b["keypoints"].float()
            student0, student1 = self._cov(a), self._cov(b)
            teacher0, teacher1 = a["teacher_covariance"], b["teacher_covariance"]
            metrics.update({f"student_{k}": v for k, v in _coverage(x, y, student0, student1, matches, h).items()})
            metrics.update({f"teacher_{k}": v for k, v in _coverage(x, y, teacher0, teacher1, matches, h).items()})
            nll, _ = covariance_nll(x, y, teacher0, teacher1, matches, h)
            metrics["teacher_covariance_nll"] = float(nll)
        return {"distillation": metrics}

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, Any]:
        self.evaluation_index = 0
        return super().evaluate(step)

    def _finalize_evaluation_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        rows = [r["distillation"] for r in metrics["rows"] if "distillation" in r]
        for name in sorted({name for row in rows for name in row}):
            values = [r[name] for r in rows if name in r]
            metrics[name] = mean(values)
        return metrics

    def stop_reason(self, step: int) -> str | None:
        stopping = self.config["auto_stop"]
        if not stopping["enabled"] or step % stopping["interval"]:
            return None
        name = self.config["selection_metric"]["name"]
        values = [v[name] for v in self.continuation_history["validation"]]
        window = stopping["window"]
        threshold = max(
            self.kd["minimum_absolute_gain"],
            abs(mean(values[-2 * window : -window])) * self.kd["minimum_relative_gain"],
        )
        decision = plateau_decision(values, mode="min", window=window, minimum_gain=threshold)
        decision.update(
            step=step, metric=name, interpretation="fixed 4-frame validation heuristic, not global convergence"
        )
        atomic_json(repo_path(self.config["run_dir"]) / "convergence" / f"step_{step:06d}.json", decision)
        print(
            f"KD continuation {step}: {decision['reason']}; gain={decision['mean_gain']:.6g}; threshold={threshold:.6g}",
            flush=True,
        )
        return decision["reason"] if decision["stop"] else None


def run_distillation(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    validate_distillation(cfg)
    metric = (
        "rank_kd"
        if cfg["task"] == "raco_rank"
        else "covariance_nll"
        if cfg["distillation"]["geometry"]["enabled"]
        else "covariance_kd"
    )
    selection = {"name": metric, "mode": "min"}
    if cfg.get("selection_metric") is not None and cfg["selection_metric"] != selection:
        raise ValueError("Selection metric does not match the distillation stage")
    cfg["selection_metric"] = selection
    configure_reproducibility(cfg["seed"])
    task = DistillationTask(cfg)
    optimizer = build_optimizer(task.model, cfg["task"], cfg["optimizer"])
    sampler = PairSampler(task.pairs, task.pair_manifest["resolved_config"], cfg["seed"])
    return train_loop(cfg, task, optimizer, sampler, task.identities)
