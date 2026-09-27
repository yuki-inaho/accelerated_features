"""AMUSE parameter groups and exact evaluation/resume state management."""

from __future__ import annotations

import copy
import fnmatch
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from third_party.amuse.AMUSE import AMUSE

XFEAT_AUX = {
    "skip1.1.weight",
    "block1.0.layer.0.weight",
    "block_fusion.2.weight",
    "heatmap_head.2.weight",
    "keypoint_head.3.weight",
    "fine_matcher.12.weight",
}
LG_AUX = {
    "input_proj.weight",
    "posenc.Wr.weight",
    "log_assignment.*.matchability.weight",
    "token_confidence.*.token.0.weight",
}


def parameter_groups(model: nn.Module, task: str, *, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    if task not in {"xfeat", "lighterglue"}:
        raise ValueError(f"Unknown optimizer task: {task}")
    names_and_params = [(n, p) for n, p in model.named_parameters(remove_duplicate=False) if p.requires_grad]
    storage = [(str(p.device), p.untyped_storage().data_ptr()) for _, p in names_and_params]
    if len(set(storage)) != len(storage):
        raise ValueError("Trainable parameters share storage or aliases")
    patterns = XFEAT_AUX if task == "xfeat" else LG_AUX
    partitions: dict[str, list[Any]] = {"muon": [], "aux_decay": [], "aux_no_decay": []}
    for name, parameter in names_and_params:
        if parameter.ndim not in (1, 2, 4):
            raise ValueError(f"Unsupported trainable parameter shape: {name} {parameter.shape}")
        auxiliary = parameter.ndim == 1 or any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        group = "aux_no_decay" if parameter.ndim == 1 else "aux_decay" if auxiliary else "muon"
        partitions[group].append((name, parameter))
    groups = []
    for kind, entries in partitions.items():
        if not entries:
            continue
        entries.sort(key=lambda item: (-item[1].numel(), item[0]))
        groups.append(
            {
                "params": [p for _, p in entries],
                "param_names": [n for n, _ in entries],
                "group_name": kind,
                "use_muon": kind == "muon",
                "lr": lr,
                "weight_decay": 0.0 if kind == "aux_no_decay" else weight_decay,
            }
        )
    if not groups:
        raise ValueError("No trainable parameters")
    return groups


def build_optimizer(model: nn.Module, task: str, config: Mapping[str, Any], *, scheduler: Any = None) -> Optimizer:
    if scheduler is not None or config.get("scheduler") is not None:
        raise ValueError("External scheduler is not supported")
    spec = copy.deepcopy(dict(config))
    groups = parameter_groups(model, task, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    if config["name"] == "amuse":
        for group in groups:
            if group["use_muon"]:
                group.update(momentum=config["momentum"], aux_update_type="adamw")
            else:
                group.update(update_type="adamw", beta2=config["beta2"], eps=config["eps"])
        optimizer = AMUSE(
            groups,
            **{k: config[k] for k in ("beta1", "rho", "r", "weight_lr_power", "weight_decay_at_y", "warmup_steps")},
        )
        # Upstream sorts Muon tensors by shape. Re-establish the documented order;
        # its state is keyed by Parameter, so this does not change any state binding.
        lookup = {id(p): n for n, p in model.named_parameters()}
        for group in optimizer.param_groups:
            group["params"].sort(key=lambda p: (-p.numel(), lookup[id(p)]))
            group["param_names"] = [lookup[id(p)] for p in group["params"]]
        optimizer.train()
    elif config["name"] == "adamw":
        for group in groups:
            group.pop("use_muon")
        optimizer = torch.optim.AdamW(
            groups, lr=float(config["lr"]), betas=tuple(config["betas"]), eps=float(config["eps"])
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['name']}")
    vars(optimizer)["_training_spec"] = {"task": task, "config": spec}
    return optimizer


def optimizer_metadata(optimizer: Optimizer) -> dict[str, Any]:
    amuse = isinstance(optimizer, AMUSE)
    groups = [
        {"names": list(g["param_names"]), "shapes": [list(p.shape) for p in g["params"]], "group_name": g["group_name"]}
        for g in optimizer.param_groups
    ]
    metadata = {
        "spec": copy.deepcopy(vars(optimizer)["_training_spec"]),
        "groups": groups,
        "parameter_state": "Y" if amuse and optimizer.train_mode else "X" if amuse else "standard",
        "train_mode": optimizer.train_mode if amuse else None,
    }
    if amuse:
        metadata["constructor"] = {
            k: getattr(optimizer, k)
            for k in ("weight_decay_at_y", "beta1_init", "weight_lr_power", "warmup_steps", "rho", "r")
        }
        metadata["current_beta1"] = getattr(optimizer, "beta1", None)
    return metadata


def restore_optimizer(optimizer: Optimizer, state: dict[str, Any], metadata: dict[str, Any]) -> None:
    current = optimizer_metadata(optimizer)
    for key in ("spec", "groups", "constructor"):
        if current.get(key) != metadata.get(key):
            raise ValueError(f"Optimizer metadata mismatch: {key}")
    if isinstance(optimizer, AMUSE) and (metadata["parameter_state"] != "Y" or metadata["train_mode"] is not True):
        raise ValueError("AMUSE training resume requires a Y checkpoint")
    optimizer.load_state_dict(state)
    if isinstance(optimizer, AMUSE):
        # Model tensors already contain Y. Calling train() would interpolate twice.
        optimizer.train_mode = True
        if metadata.get("current_beta1") is not None:
            optimizer.beta1 = metadata["current_beta1"]


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 -- global RNG is part of checkpoint contract
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002 -- restore global RNG from checkpoint
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def finish_update(optimizer: Optimizer, *, max_norm: float = 1.0, scaler: Any = None) -> dict[str, Any]:
    """Unscale, reject overflow, clip and step; skipped updates never advance AMUSE k."""
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not parameters:
        return {"performed": False, "overflow": False, "grad_norm": 0.0}
    if scaler is not None:
        scaler.unscale_(optimizer)
    finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            raise FloatingPointError("nonfinite fp32 gradient")
        scaler.update()
        return {"performed": False, "overflow": True, "grad_norm": None}
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {"performed": True, "overflow": False, "grad_norm": float(norm)}


@contextmanager
def evaluation_parameters(model: nn.Module, optimizer: Optimizer) -> Iterator[None]:
    """Use X temporarily, restoring Y/buffers/RNG/modes bit-for-bit even on failure."""
    snapshot = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    train_mode = optimizer.train_mode if isinstance(optimizer, AMUSE) else None
    try:
        if isinstance(optimizer, AMUSE):
            optimizer.eval()
        model.eval()
        yield
    finally:
        model.load_state_dict(snapshot, strict=True)
        for module, training in modes:
            module.training = training
        if isinstance(optimizer, AMUSE):
            assert train_mode is not None
            optimizer.train_mode = train_mode
        restore_rng(rng)
