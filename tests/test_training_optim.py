"""AMUSE partitioning, explicit Y/X state and exact CPU resume."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from modules.lighterglue import LighterGlue
from modules.model import XFeatModel
from modules.utils import state_hash
from third_party.amuse.AMUSE import AMUSE
from xfeat_training.optim import (
    build_optimizer,
    evaluation_parameters,
    finish_update,
    optimizer_metadata,
    parameter_groups,
    restore_optimizer,
)


@pytest.mark.parametrize("task", ["xfeat", "lighterglue"])
def test_all_parameters_have_one_stable_group(task: str) -> None:
    model = XFeatModel() if task == "xfeat" else LighterGlue(device="cpu", flash=False).net
    groups = parameter_groups(model, task, lr=3e-4, weight_decay=0.01)
    names = [name for group in groups for name in group["param_names"]]
    assert sorted(names) == sorted(name for name, p in model.named_parameters() if p.requires_grad)
    assert len(names) == len(set(names))
    lookup = dict(model.named_parameters())
    for group in groups:
        assert group["param_names"] == sorted(group["param_names"], key=lambda n: (-lookup[n].numel(), n))
        assert [id(p) for p in group["params"]] == [id(lookup[n]) for n in group["param_names"]]
        if group["use_muon"]:
            assert all(p.ndim in (2, 4) for p in group["params"])
        if all(p.ndim == 1 for p in group["params"]):
            assert group["weight_decay"] == 0
    aux = {n for g in groups if not g["use_muon"] for n in g["param_names"]}
    expected = (
        {
            "skip1.1.weight",
            "block1.0.layer.0.weight",
            "block_fusion.2.weight",
            "heatmap_head.2.weight",
            "keypoint_head.3.weight",
            "fine_matcher.12.weight",
        }
        if task == "xfeat"
        else {
            "input_proj.weight",
            "posenc.Wr.weight",
            "log_assignment.0.matchability.weight",
            "token_confidence.0.token.0.weight",
        }
    )
    assert expected <= aux


def _update(model, optimizer) -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
    model(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_amuse_cpu_continuous_and_resumed_are_equal(amuse_config, assert_nested_equal) -> None:
    torch.manual_seed(7)
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    for _ in range(5):
        _update(model, optimizer)
    saved_model = copy.deepcopy(model.state_dict())
    saved_optim, metadata = copy.deepcopy(optimizer.state_dict()), optimizer_metadata(optimizer)
    assert metadata["parameter_state"] == "Y" and metadata["train_mode"] is True
    for _ in range(4):
        _update(model, optimizer)
    resumed = nn.Linear(4, 4)
    resumed.load_state_dict(saved_model)
    resumed_opt = build_optimizer(resumed, "xfeat", amuse_config)
    restore_optimizer(resumed_opt, saved_optim, metadata)
    for _ in range(4):
        _update(resumed, resumed_opt)
    assert_nested_equal(model.state_dict(), resumed.state_dict())
    assert_nested_equal(optimizer.state_dict(), resumed_opt.state_dict())


def test_eval_exception_restores_y_buffers_modes_and_rng(amuse_config, assert_nested_equal) -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    model[1].eval()
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    for _ in range(3):
        _update(model, optimizer)
    before = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    assert isinstance(optimizer, AMUSE)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())  # noqa: NPY002 -- global RNG snapshot
    with pytest.raises(RuntimeError, match="evaluation failed"), evaluation_parameters(model, optimizer):
        assert not optimizer.train_mode
        assert state_hash(model.state_dict()) != state_hash(before)
        batchnorm = model[1]
        assert isinstance(batchnorm, nn.BatchNorm1d) and batchnorm.running_mean is not None
        batchnorm.running_mean.add_(10)
        random.random()
        np.random.rand()  # noqa: NPY002 -- exercise restoration after evaluation
        torch.rand(1)
        raise RuntimeError("evaluation failed")
    assert optimizer.train_mode
    assert_nested_equal(before, model.state_dict())
    assert [m.training for m in model.modules()] == modes
    assert_nested_equal(rng, (random.getstate(), np.random.get_state(), torch.get_rng_state()))  # noqa: NPY002 -- exact global state


def test_scheduler_rejected_and_adamw_has_no_amuse_transition(amuse_config) -> None:
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="scheduler"):
        build_optimizer(model, "xfeat", amuse_config, scheduler="cosine")
    optimizer = build_optimizer(
        model, "xfeat", {"name": "adamw", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    )
    _update(model, optimizer)
    before = state_hash(model.state_dict())
    with evaluation_parameters(model, optimizer):
        assert state_hash(model.state_dict()) == before
    assert optimizer_metadata(optimizer)["parameter_state"] == "standard"


def test_shared_storage_is_rejected() -> None:
    model = nn.Module()
    shared = nn.Parameter(torch.ones(4, 4))
    model.register_parameter("a", shared)
    model.register_parameter("b", nn.Parameter(shared.data.view(4, 4)))
    with pytest.raises(ValueError, match="storage"):
        parameter_groups(model, "xfeat", lr=3e-4, weight_decay=0.01)


def test_skip_and_overflow_never_advance_amuse(amuse_config) -> None:
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    before = state_hash(model.state_dict())
    assert not finish_update(optimizer)["performed"]
    assert all(g["k"] == 0 for g in optimizer.param_groups)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        finish_update(optimizer)
    assert all(g["k"] == 0 for g in optimizer.param_groups)
    assert state_hash(model.state_dict()) == before

    class Scaler:
        def __init__(self):
            self.calls = []

        def unscale_(self, opt):
            self.calls.append("unscale")

        def update(self):
            self.calls.append("update")

        def step(self, opt):
            self.calls.append("step")
            opt.step()

    scaler = Scaler()
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    result = finish_update(optimizer, scaler=scaler)
    assert result["overflow"] and not result["performed"]
    assert scaler.calls == ["unscale", "update"]
    assert all(g["k"] == 0 for g in optimizer.param_groups)
    model(torch.ones(4, 4)).sum().backward()
    assert finish_update(optimizer, scaler=scaler)["performed"]
    assert scaler.calls[-3:] == ["unscale", "step", "update"]
    assert all(g["k"] == 1 for g in optimizer.param_groups)
