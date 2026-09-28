"""Training-only wrapper for RaCo multiscale student and auxiliary heads."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from statistics import median
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.raco import XFeatRaCo, resize_input, sample_map
from modules.utils import state_hash
from xfeat_training.data import json_hash
from xfeat_training.mining import pair_keys, repo_path
from xfeat_training.optim import build_optimizer
from xfeat_training.raco_distillation import rank_pair_kl
from xfeat_training.raco_multiscale_losses import (
    LocalContrastProjection,
    TwoScaleFeatureAlignment,
    build_mutual_nearest_pairs,
    symmetric_local_info_nce,
    two_scale_feature_loss,
)
from xfeat_training.raco_objectives import covariance_nll, ranking_loss
from xfeat_training.raco_task import RacoTask
from xfeat_training.raco_teacher import OfficialRacoTeacher
from xfeat_training.trainer import PairSampler, configure_reproducibility, train_loop


def calibrate_gradient_weights(
    loss_provider: Callable[[int], Mapping[str, Tensor]],
    shared_parameters: Mapping[str, nn.Parameter],
    *,
    batch_count: int = 8,
    covariance_bounds: tuple[float, float] = (0.1, 10.0),
    shared_aux_fraction: float = 0.25,
    variant_aux_fraction: float = 0.25,
    variant_aux_key: str | None = None,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Calibrate fixed loss weights from median gradients on shared student parameters."""

    if type(batch_count) is not int or batch_count < 1:
        raise ValueError("batch_count must be a positive integer")
    lower, upper = covariance_bounds
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        raise ValueError("covariance_bounds must be finite positive ordered values")
    if not (math.isfinite(shared_aux_fraction) and 0 < shared_aux_fraction <= 1):
        raise ValueError("shared_aux_fraction must lie in (0, 1]")
    if not (math.isfinite(variant_aux_fraction) and 0 < variant_aux_fraction <= 1):
        raise ValueError("variant_aux_fraction must lie in (0, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    names, parameters = tuple(shared_parameters), tuple(shared_parameters.values())
    if not names or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("shared_parameters must be a nonempty mapping of trainable parameters")

    keys = ["rank_geometry", "covariance_geometry", "descriptor", "rank_kd"]
    if variant_aux_key is not None:
        keys.append(variant_aux_key)
    common = [True] * len(parameters)

    def losses_for(index: int) -> Mapping[str, Tensor]:
        losses = loss_provider(index)
        missing = set(keys) - set(losses)
        if missing:
            raise ValueError(f"gradient calibration is missing losses: {sorted(missing)}")
        for name in keys:
            value = losses[name]
            if not isinstance(value, Tensor) or value.ndim != 0 or not torch.isfinite(value):
                raise ValueError(f"gradient calibration loss {name} must be a finite scalar tensor")
            if not value.requires_grad:
                raise ValueError(f"gradient calibration loss {name} has no gradient graph")
        return losses

    # First pass finds the same parameter intersection for every loss and batch.
    for index in range(batch_count):
        losses = losses_for(index)
        for name in keys:
            gradients = torch.autograd.grad(losses[name], parameters, retain_graph=True, allow_unused=True)
            common = [
                was_common and gradient is not None for was_common, gradient in zip(common, gradients, strict=True)
            ]
    common_indices = [index for index, valid in enumerate(common) if valid]
    if not common_indices:
        raise ValueError("gradient calibration has no common student parameters")
    common_parameters = tuple(parameters[index] for index in common_indices)
    common_names = tuple(names[index] for index in common_indices)

    def norm(loss: Tensor) -> float:
        gradients = torch.autograd.grad(loss, common_parameters, retain_graph=True, allow_unused=False)
        squared = torch.stack([gradient.detach().float().square().sum() for gradient in gradients]).sum()
        value = float(squared.sqrt().cpu())
        if not math.isfinite(value) or value <= 0:
            raise ValueError("gradient calibration encountered zero or nonfinite common gradient norm")
        return value

    # Second pass recomputes one batch at a time, keeping peak graph memory bounded.
    rank_norms, covariance_norms = [], []
    for index in range(batch_count):
        losses = losses_for(index)
        rank_norms.append(norm(losses["rank_geometry"]))
        covariance_norms.append(norm(losses["covariance_geometry"]))
    rank_median, covariance_median = median(rank_norms), median(covariance_norms)
    covariance_weight = min(upper, max(lower, rank_median / (covariance_median + epsilon)))

    task_norms, shared_aux_norms, variant_aux_norms = [], [], []
    for index in range(batch_count):
        losses = losses_for(index)
        task_norms.append(norm(losses["rank_geometry"] + covariance_weight * losses["covariance_geometry"]))
        shared_aux_norms.append(norm(losses["descriptor"] + losses["rank_kd"]))
        if variant_aux_key is not None:
            variant_aux_norms.append(norm(losses[variant_aux_key]))
    task_median, shared_aux_median = median(task_norms), median(shared_aux_norms)
    shared_aux_weight = min(1.0, shared_aux_fraction * task_median / (shared_aux_median + epsilon))
    variant_aux_weight = None
    if variant_aux_norms:
        variant_aux_weight = min(1.0, variant_aux_fraction * task_median / (median(variant_aux_norms) + epsilon))
    return {
        "batches": batch_count,
        "common_parameter_names": list(common_names),
        "rank_gradient_norms": rank_norms,
        "covariance_gradient_norms": covariance_norms,
        "task_gradient_norms": task_norms,
        "shared_aux_gradient_norms": shared_aux_norms,
        "variant_aux_gradient_norms": variant_aux_norms,
        "lambda_cov": covariance_weight,
        "lambda_shared_aux": shared_aux_weight,
        "lambda_variant_aux": variant_aux_weight,
        "variant_aux_key": variant_aux_key,
    }


def calibrate_multiscale_weights(
    loss_provider: Callable[[int], Mapping[str, Tensor]],
    shared_parameters: Mapping[str, nn.Parameter],
    variant_auxiliaries: Mapping[str, str],
    *,
    batch_count: int = 8,
    covariance_bounds: tuple[float, float] = (0.1, 10.0),
    shared_aux_fraction: float = 0.25,
    variant_aux_fraction: float = 0.25,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fix common coefficients once, then calibrate only enabled variant losses."""

    weight_names = tuple(variant_auxiliaries.values())
    if len(set(variant_auxiliaries)) != len(variant_auxiliaries) or len(set(weight_names)) != len(weight_names):
        raise ValueError("Variant auxiliary loss keys and weight names must be unique")
    common = calibrate_gradient_weights(
        loss_provider,
        shared_parameters,
        batch_count=batch_count,
        covariance_bounds=covariance_bounds,
        shared_aux_fraction=shared_aux_fraction,
        variant_aux_fraction=variant_aux_fraction,
    )
    weights = {
        "lambda_cov": common["lambda_cov"],
        "lambda_shared_aux": common["lambda_shared_aux"],
    }
    records: dict[str, Any] = {"shared": common}
    for loss_key, weight_name in variant_auxiliaries.items():
        result = calibrate_gradient_weights(
            loss_provider,
            shared_parameters,
            batch_count=batch_count,
            covariance_bounds=covariance_bounds,
            shared_aux_fraction=shared_aux_fraction,
            variant_aux_fraction=variant_aux_fraction,
            variant_aux_key=loss_key,
        )
        variant_weight = result["lambda_variant_aux"]
        if variant_weight is None:
            raise RuntimeError(f"Variant auxiliary loss {loss_key} did not produce a coefficient")
        weights[weight_name] = variant_weight
        result["lambda_cov"] = common["lambda_cov"]
        result["lambda_shared_aux"] = common["lambda_shared_aux"]
        result["shared_coefficient_source"] = "shared"
        records[loss_key] = result
    return weights, records


class RacoMultiscaleTrainingModel(nn.Module):
    """Checkpoint student and train-only heads together; export the student alone."""

    def __init__(
        self,
        student: XFeatRaCo,
        auxiliary_heads: Mapping[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.student = student
        self.auxiliary_heads = nn.ModuleDict(dict(auxiliary_heads or {}))
        self.to(next(self.student.parameters()).device)

    @property
    def net(self) -> nn.Module:
        return self.student.net

    @property
    def heads(self) -> nn.Module:
        return self.student.heads

    @property
    def trained_heads(self) -> set[str]:
        return self.student.trained_heads

    def configure_multiscale_training(self) -> tuple[str, ...]:
        student_names = self.student.configure_multiscale_training()
        batch_norm_prefixes = tuple(
            f"{module_name}."
            for module_name, module in self.auxiliary_heads.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        auxiliary_names = []
        for name, parameter in self.auxiliary_heads.named_parameters():
            enabled = not name.startswith(batch_norm_prefixes)
            parameter.requires_grad_(enabled)
            if enabled:
                auxiliary_names.append(f"auxiliary_heads.{name}")
        self.train(True)
        return tuple(sorted([*(f"student.{name}" for name in student_names), *auxiliary_names]))

    def candidates(self, image: Tensor, valid_mask: Tensor | None = None) -> list[dict[str, Tensor]]:
        return self.student.candidates(image, valid_mask)

    def training_candidates(
        self,
        image: Tensor,
        valid_mask: Tensor | None = None,
        *,
        include_feature_maps: bool = False,
    ) -> list[dict[str, Tensor]]:
        return self.student.training_candidates(image, valid_mask, include_feature_maps=include_feature_maps)

    def predict(self, candidates: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return self.student.predict(candidates, **kwargs)

    def extract(self, image: Tensor, **kwargs) -> list[dict[str, Tensor]]:
        return self.student.extract(image, **kwargs)

    def bundle(self, *, trained_heads: list[str]) -> dict:
        """Keep auxiliary modules out of the existing five-key inference bundle."""
        return self.student.bundle(trained_heads=trained_heads)


class RacoMultiscaleTask(RacoTask):
    """Joint rank/covariance geometry task with optional train-only KD branches."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.multiscale = self.config["multiscale"]
        student = cast(XFeatRaCo, self.model)
        self.reference = (
            XFeatRaCo.from_bundle(
                repo_path(self.config["init_bundle"]), device=self.device, ranking=False, covariance=False
            )
            .requires_grad_(False)
            .eval()
        )
        self.reference_hash = state_hash(self.reference.state_dict())
        self.teacher = OfficialRacoTeacher(
            repo_path(self.config["teacher_source"]), repo_path(self.config["teacher_weights"]), self.device
        )
        auxiliary: dict[str, nn.Module] = {}
        if self.multiscale["feature_alignment"]["enabled"]:
            auxiliary["feature_alignment"] = TwoScaleFeatureAlignment()
        if self.multiscale["local_contrast"]["enabled"]:
            auxiliary["local_contrast"] = LocalContrastProjection()
        self.model = RacoMultiscaleTrainingModel(student, auxiliary)
        self.trainable_names = self.model.configure_multiscale_training()
        self.trained_heads = student.trained_heads | {"rank", "covariance"}
        self.loss_weights: dict[str, float] = {}
        self.calibration_pair_indices: list[int] = []
        self.temperature = self.config["temperature_start"]
        self.preview = None
        self._frozen_parameters = {
            name: parameter.detach().clone()
            for name, parameter in student.named_parameters()
            if not parameter.requires_grad
        }
        self._buffers = {name: value.detach().clone() for name, value in student.named_buffers()}
        self.identities["teacher"] = self.teacher.identity
        self.identities["reference_student"] = {
            "init_bundle_sha256": self.identities["init_bundle_sha256"],
            "state_hash": self.reference_hash,
        }
        self.identities["trainable_parameters"] = list(self.trainable_names)
        self.identities["evaluation"]["multiscale"] = self.multiscale
        self.identities["evaluation"]["teacher"] = self.teacher.identity
        self.identities["evaluation_hash"] = json_hash(self.identities["evaluation"])

    @property
    def training_model(self) -> RacoMultiscaleTrainingModel:
        return cast(RacoMultiscaleTrainingModel, self.model)

    def set_calibration_pairs(self, indices: list[int]) -> None:
        if len(indices) != self.multiscale["calibration"]["batches"]:
            raise ValueError("Calibration pair list must match the configured batch count")
        self.calibration_pair_indices = list(indices)
        self.identities["calibration_pair_indices"] = list(indices)
        self.identities["calibration_pair_hash"] = json_hash(
            [tuple(key.name() for key in pair_keys(self.pairs, index)) for index in indices]
        )

    def set_calibration(self, weights: Mapping[str, float], records: Mapping[str, Any]) -> None:
        self.loss_weights = {name: float(value) for name, value in weights.items() if value is not None}
        self.multiscale["loss_weights"] = dict(self.loss_weights)
        self.multiscale["calibration_result"] = dict(records)
        self.identities["calibration"] = dict(records)
        self.identities["evaluation"]["multiscale"] = self.multiscale
        self.identities["evaluation_hash"] = json_hash(self.identities["evaluation"])

    def _check_frozen(self) -> None:
        student = self.training_model.student
        current_parameters = dict(student.named_parameters())
        current_buffers = dict(student.named_buffers())
        if any(
            not torch.equal(current_parameters[name].detach(), value) for name, value in self._frozen_parameters.items()
        ):
            raise RuntimeError("A frozen XFeat/RaCo parameter changed during multiscale training")
        if any(not torch.equal(current_buffers[name].detach(), value) for name, value in self._buffers.items()):
            raise RuntimeError("A frozen BatchNorm or model buffer changed during multiscale training")
        if state_hash(self.reference.state_dict()) != self.reference_hash:
            raise RuntimeError("The fixed XFeat descriptor reference changed")
        self.teacher.check_frozen()

    def _pair(self, key, generator: torch.Generator | None = None):
        result = super()._pair(key, generator)
        self.preview = None
        return result

    def _make_candidates(self, image: Tensor, support: Tensor) -> dict[str, Tensor]:
        if not torch.is_grad_enabled():
            return self.training_model.candidates(image, support)[0]
        wants_maps = self.multiscale["feature_alignment"]["enabled"] or self.multiscale["local_contrast"]["enabled"]
        candidate = self.training_model.training_candidates(image, support, include_feature_maps=wants_maps)[0]
        points = candidate["keypoints"]
        reference_input, _ = resize_input(image)
        with torch.no_grad():
            reference_features = self.reference.net.forward_with_features(reference_input)[0]
            reference_map = F.normalize(reference_features, dim=1)
            if len(points):
                candidate["reference_descriptors"] = F.normalize(
                    sample_map(
                        reference_map,
                        points,
                        height=reference_input.shape[-2],
                        width=reference_input.shape[-1],
                        mode="bicubic",
                    ),
                    dim=-1,
                )
            else:
                candidate["reference_descriptors"] = reference_features.new_empty((0, reference_features.shape[1]))
            teacher_maps = self.teacher.dense(image, include_features=self.multiscale["feature_alignment"]["enabled"])
            if len(points):
                teacher_values = self.teacher.sample(teacher_maps, points[None])
                candidate["teacher_rank"] = teacher_values["rank"][0].detach()
            else:
                candidate["teacher_rank"] = points.new_empty((0,), dtype=torch.float32)
            if self.multiscale["feature_alignment"]["enabled"]:
                candidate["teacher_block3_map"] = teacher_maps["block3"].detach()
                candidate["teacher_block4_map"] = teacher_maps["block4"].detach()
        return candidate

    def _raw_loss_terms(
        self,
        pair: tuple[dict[str, Tensor], dict[str, Tensor], Tensor, float],
        *,
        rank_generator: torch.Generator | None = None,
        pair_seed: int,
    ) -> dict[str, Tensor] | None:
        a, b, homography, _overlap = pair
        rank_matches = self._matches(a, b, homography, self.config["rank_threshold"])
        covariance_matches = self._matches(a, b, homography, self.config["covariance_threshold"])
        if not len(rank_matches) or not len(covariance_matches):
            self.skip_reason = "missing_rank_or_covariance_geometric_matches"
            return None
        rank_geometry = ranking_loss(
            self._rank(a), self._rank(b), rank_matches, self.config["budgets"], self.temperature
        )
        covariance_geometry, _ = covariance_nll(
            a["keypoints"].float(), b["keypoints"].float(), self._cov(a), self._cov(b), covariance_matches, homography
        )
        descriptor = 0.5 * (
            (1 - (a["descriptors"] * a["reference_descriptors"]).sum(-1)).mean()
            + (1 - (b["descriptors"] * b["reference_descriptors"]).sum(-1)).mean()
        )
        kd = self.multiscale["rank_kd"]
        rank_distillation = 0.5 * (
            rank_pair_kl(
                self._rank(a),
                a["teacher_rank"],
                comparisons=kd["comparisons"],
                budget=kd["budget"],
                boundary_radius=kd["boundary_radius"],
                generator=rank_generator,
            )
            + rank_pair_kl(
                self._rank(b),
                b["teacher_rank"],
                comparisons=kd["comparisons"],
                budget=kd["budget"],
                boundary_radius=kd["boundary_radius"],
                generator=rank_generator,
            )
        )
        zero = rank_geometry.new_zeros(())
        feature_alignment, local_contrast = zero, zero
        local_pairs = local_valid = local_invalid = local_active = zero
        if self.multiscale["feature_alignment"]["enabled"]:
            adapter = cast(TwoScaleFeatureAlignment, self.training_model.auxiliary_heads["feature_alignment"])
            per_view = [
                two_scale_feature_loss(
                    c["block3_map"],
                    c["block5_map"],
                    c["teacher_block3_map"],
                    c["teacher_block4_map"],
                    c["support"],
                    adapter,
                )
                for c in (a, b)
            ]
            feature_alignment = 0.5 * (per_view[0]["loss"] + per_view[1]["loss"])
        if self.multiscale["local_contrast"]["enabled"]:
            height, width = a["support"].shape[-2:]
            local = self.multiscale["local_contrast"]
            matches = build_mutual_nearest_pairs(
                a["keypoints"].float(),
                b["keypoints"].float(),
                homography,
                (height, width),
                max_distance_px=local["match_distance_px"],
                max_pairs=local["max_pairs"],
                seed=pair_seed,
                ambiguity_margin_px=local["ambiguity_margin_px"],
            )
            points_a = a["keypoints"][matches["source_indices"]].float()
            points_b = b["keypoints"][matches["target_indices"]].float()
            projector = cast(LocalContrastProjection, self.training_model.auxiliary_heads["local_contrast"])
            embedding_a = projector(a["block3_map"], points_a, image_size=(height, width))
            embedding_b = projector(b["block3_map"], points_b, image_size=(height, width))
            contrast = symmetric_local_info_nce(
                embedding_a,
                embedding_b,
                points_a,
                points_b,
                temperature=local["temperature"],
                exclusion_px=local["exclusion_px"],
                min_pairs=local["min_pairs"],
            )
            local_contrast = contrast["loss"]
            local_pairs = zero.new_tensor(contrast["pair_count"])
            local_valid = zero.new_tensor(contrast["valid_queries"])
            local_invalid = zero.new_tensor(contrast["invalid_queries"])
            local_active = zero.new_tensor(float(contrast["active"]))
        return {
            "rank_geometry": rank_geometry,
            "covariance_geometry": covariance_geometry,
            "descriptor": descriptor,
            "rank_kd": rank_distillation,
            "feature": feature_alignment,
            "local": local_contrast,
            "local_pairs": local_pairs,
            "local_valid_queries": local_valid,
            "local_invalid_queries": local_invalid,
            "local_active": local_active,
        }

    def calibration_loss_terms(self, calibration_index: int) -> dict[str, Tensor]:
        if not 0 <= calibration_index < len(self.calibration_pair_indices):
            raise IndexError("Calibration pair index is out of range")
        pair_index = self.calibration_pair_indices[calibration_index]
        seed = self.config["seed"] + calibration_index * 1009
        generator = torch.Generator().manual_seed(seed)
        pair = self._pair(pair_keys(self.pairs, pair_index)[0], generator)
        if pair is None:
            raise ValueError(f"Calibration microbatch {calibration_index} has no usable pair: {self.skip_reason}")
        rank_generator = torch.Generator().manual_seed(seed + 313)
        terms = self._raw_loss_terms(pair, rank_generator=rank_generator, pair_seed=seed + 617)
        if terms is None:
            raise ValueError(f"Calibration microbatch {calibration_index} has no geometric matches")
        return terms

    def loss(self, pair_index: int, accepted_microbatches: int) -> dict[str, Tensor] | None:
        del accepted_microbatches
        pair = self._pair(pair_keys(self.pairs, pair_index)[0])
        if pair is None:
            return None
        seed = self.config["seed"] + self.successful_step * 1009 + pair_index
        rank_generator = torch.Generator().manual_seed(seed + 313)
        terms = self._raw_loss_terms(pair, rank_generator=rank_generator, pair_seed=seed + 617)
        if terms is None:
            return None
        total = terms["rank_geometry"] + self.loss_weights["lambda_cov"] * terms["covariance_geometry"]
        total = total + self.loss_weights["lambda_shared_aux"] * (terms["descriptor"] + terms["rank_kd"])
        if "lambda_feature" in self.loss_weights:
            total = total + self.loss_weights["lambda_feature"] * terms["feature"]
        if "lambda_local" in self.loss_weights:
            total = total + self.loss_weights["lambda_local"] * terms["local"]
        return {"loss": total, **terms, "overlap": total.new_tensor(pair[3])}

    def _finalize_evaluation_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        from xfeat_training.raco_multiscale_train import multiscale_selection_score

        metrics["multiscale_q"] = multiscale_selection_score(metrics, self.multiscale["baseline"])
        metrics["variant"] = self.multiscale["variant"]
        return metrics


def run_multiscale_task(config: Mapping[str, Any]) -> dict[str, Any]:
    """Calibrate fixed weights, then use the shared strict-resume training loop."""

    from xfeat_training.raco_multiscale_train import validate_multiscale_config

    cfg = dict(config)
    validate_multiscale_config(cfg)
    if cfg["multiscale"]["variant"] == "SSKD":
        raise ValueError("SSKD is gated until the teacher auxiliary head is trained and validated")
    if repo_path(cfg["run_dir"]).exists():
        raise FileExistsError(f"Run directory already exists: {cfg['run_dir']}")
    configure_reproducibility(cfg["seed"], warn_only=cfg["deterministic_warn_only"])
    task = RacoMultiscaleTask(cfg)
    calibration_sampler = PairSampler(task.pairs, task.pair_manifest["resolved_config"], cfg["seed"])
    calibration_indices = [
        calibration_sampler.sample(step) for step in range(cfg["multiscale"]["calibration"]["batches"])
    ]
    task.set_calibration_pairs(calibration_indices)
    shared = {
        name: parameter
        for name, parameter in task.model.named_parameters()
        if name.startswith(("student.net.block3.", "student.net.block5.", "student.net.block_fusion."))
    }
    variant_auxiliaries = {}
    if cfg["multiscale"]["feature_alignment"]["enabled"]:
        variant_auxiliaries["feature"] = "lambda_feature"
    if cfg["multiscale"]["local_contrast"]["enabled"]:
        variant_auxiliaries["local"] = "lambda_local"
    weights, calibration_results = calibrate_multiscale_weights(
        task.calibration_loss_terms,
        shared,
        variant_auxiliaries,
        batch_count=cfg["multiscale"]["calibration"]["batches"],
        covariance_bounds=tuple(cfg["multiscale"]["calibration"]["covariance_weight_bounds"]),
        shared_aux_fraction=cfg["multiscale"]["calibration"]["shared_aux_fraction"],
        variant_aux_fraction=cfg["multiscale"]["calibration"]["variant_aux_fraction"],
    )
    task.set_calibration(weights, {"pair_indices": calibration_indices, "results": calibration_results})
    cfg["multiscale"]["loss_weights"] = weights
    optimizer = build_optimizer(task.model, "raco_multiscale", cfg["optimizer"])
    sampler = PairSampler(task.pairs, task.pair_manifest["resolved_config"], cfg["seed"])
    return train_loop(cfg, task, optimizer, sampler, task.identities)
