"""Hydra entry point and fair-run configuration helpers for RaCo multiscale KD."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import reject_multirun
from xfeat_training.raco_task import validate_config

VARIANTS = {"A", "B", "C", "BC", "SSKD"}


def build_variant_config(base: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """Clone shared settings and change only the explicitly selected auxiliary path."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown multiscale variant: {variant}")
    config = copy.deepcopy(dict(base))
    multiscale = config.get("multiscale")
    if not isinstance(multiscale, dict):
        raise ValueError("multiscale configuration template is required")
    for key in ("feature_alignment", "local_contrast", "relation_kd", "rank_kd", "covariance_kd"):
        if not isinstance(multiscale.get(key), dict) or type(multiscale[key].get("enabled")) is not bool:
            raise ValueError(f"multiscale.{key}.enabled must be configured as a boolean")
    multiscale["variant"] = variant
    multiscale["feature_alignment"]["enabled"] = variant in {"B", "BC", "SSKD"}
    multiscale["local_contrast"]["enabled"] = variant in {"C", "BC", "SSKD"}
    multiscale["relation_kd"]["enabled"] = variant == "SSKD"
    multiscale["rank_kd"]["enabled"] = True
    multiscale["covariance_kd"]["enabled"] = False
    config["selection_metric"] = {"name": "multiscale_q", "mode": "max"}
    config["checkpoint_keep_best"] = 3
    return config


def validate_multiscale_config(config: Mapping[str, Any]) -> None:
    """Validate the shared base training contract and variant-specific loss gates."""

    validate_config(config)
    if config["task"] != "raco_rank":
        raise ValueError("Multiscale training uses the rank task contract as its common base")
    if config.get("val_frames") != 16:
        raise ValueError("Multiscale runs require a 16-frame fixed validation cohort")
    if config.get("image_size") != [480, 640]:
        raise ValueError("Multiscale runs require the baseline 480x640 image size")
    if config.get("deterministic_warn_only") is not True:
        raise ValueError("Multiscale CUDA grid sampling requires deterministic warn-only mode")
    if not config.get("init_bundle"):
        raise ValueError("Multiscale training requires an explicit init_bundle")
    stop = config.get("stop_after_steps")
    if type(stop) is not int or not 0 < stop <= config["max_steps"]:
        raise ValueError("stop_after_steps must be a positive integer no greater than max_steps")
    if config.get("checkpoint_keep_best") != 3:
        raise ValueError("Multiscale runs retain validation top 3 plus latest")
    if config.get("selection_metric") != {"name": "multiscale_q", "mode": "max"}:
        raise ValueError("Multiscale runs must select checkpoints by multiscale_q")

    multiscale = config.get("multiscale")
    if not isinstance(multiscale, Mapping):
        raise ValueError("Missing multiscale settings")
    variant = multiscale.get("variant")
    if variant not in VARIANTS:
        raise ValueError("Unknown multiscale variant")
    expected = {
        "feature_alignment": variant in {"B", "BC", "SSKD"},
        "local_contrast": variant in {"C", "BC", "SSKD"},
        "relation_kd": variant == "SSKD",
        "rank_kd": True,
        "covariance_kd": False,
    }
    for name, enabled in expected.items():
        section = multiscale.get(name)
        if not isinstance(section, Mapping) or section.get("enabled") is not enabled:
            raise ValueError(f"Variant {variant} has inconsistent {name} switch")
    baseline = multiscale.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ValueError("Multiscale fixed-val baseline is required")
    for key in ("rank_utility", "covariance_nll"):
        value = baseline.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"multiscale.baseline.{key} must be finite and positive")
    calibration = multiscale.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("batches") != 8:
        raise ValueError("Multiscale gradient calibration requires eight fixed microbatches")


def multiscale_selection_score(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    """Return Q=min(relative rank-utility gain, relative covariance-NLL gain)."""

    values = {
        "rank_utility": metrics.get("rank_utility"),
        "covariance_nll": metrics.get("covariance_nll"),
        "baseline rank_utility": baseline.get("rank_utility"),
        "baseline covariance_nll": baseline.get("covariance_nll"),
    }
    numeric: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(value, (int, float)) or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric")
        numeric[name] = float(value)
    utility, nll = numeric["rank_utility"], numeric["covariance_nll"]
    base_utility = numeric["baseline rank_utility"]
    base_nll = numeric["baseline covariance_nll"]
    if min(utility, nll, base_utility, base_nll) <= 0:
        raise ValueError("Rank utility and covariance NLL must be positive")
    return min((utility - base_utility) / base_utility, (base_nll - nll) / base_nll)


def run_multiscale_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create the task and hand the immutable run contract to the shared trainer."""

    cfg = build_variant_config(config, config["multiscale"]["variant"])
    validate_multiscale_config(cfg)
    from xfeat_training.raco_multiscale_task import run_multiscale_task

    return run_multiscale_task(cfg)


@hydra.main(version_base="1.3", config_path="../configs", config_name="raco_multiscale")
def main(config: DictConfig) -> None:
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    run_multiscale_training(resolved)


if __name__ == "__main__":
    reject_multirun()
    main()
