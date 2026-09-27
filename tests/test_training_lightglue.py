"""Layer supervision and confidence targets, including the dustbin."""

from __future__ import annotations

import pytest
import torch

from modules.lighterglue import LighterGlue
from xfeat_training.lightglue import assignment_nll, confidence_targets, forward_layers, lightglue_loss


def inputs(n=5, m=7):
    return {
        f"image{i}": {
            "keypoints": torch.rand(1, count, 2) * 64,
            "descriptors": torch.randn(1, count, 64, requires_grad=True),
            "image_size": torch.tensor([[80.0, 64.0]]),
        }
        for i, count in enumerate((n, m))
    }


def test_training_final_assignment_matches_existing_forward() -> None:
    torch.manual_seed(13)
    net = LighterGlue(device="cpu", flash=False, width_confidence=-1, depth_confidence=-1).net.eval()
    data = inputs()
    outputs = forward_layers(net, data)
    assert outputs is not None
    with torch.no_grad():
        reference = net(data)
    torch.testing.assert_close(outputs.assignments[-1], reference["log_assignment"], atol=1e-6, rtol=1e-5)
    assert len(outputs.assignments) == 6


def test_every_assignment_and_confidence_head_gets_gradient() -> None:
    torch.manual_seed(14)
    net = LighterGlue(device="cpu", flash=False, width_confidence=-1, depth_confidence=-1).net
    data = inputs()
    output = forward_layers(net, data)
    losses = lightglue_loss(net, output, torch.tensor([0, 1, -1, -2, -1]), torch.tensor([0, 1, -1, -1, -1, -2, -1]))
    assert losses is not None and torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for head in list(net.log_assignment) + list(net.token_confidence):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert data["image0"]["descriptors"].grad is not None
    position_gradient = net.posenc.Wr.weight.grad
    assert position_gradient is not None and position_gradient.abs().sum() > 0


def test_nll_ignores_unknown_and_handles_zero_positive() -> None:
    log = torch.tensor([[[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, 0.0]]], requires_grad=True)
    # one positive log[0,0], source-negative log[1,dustbin], target-negative log[dustbin,1]
    loss = assignment_nll(log, torch.tensor([0, -1]), torch.tensor([0, -1]))
    assert loss.item() == pytest.approx(0.5 * 1 + 0.5 * (6 + 8) / 2)
    unknown = assignment_nll(log, torch.tensor([-2, -2]), torch.tensor([-2, -2]))
    assert unknown.item() == 0
    negative = assignment_nll(log, torch.tensor([-1, -2]), torch.tensor([-2, -1]))
    assert negative.item() == pytest.approx(0.5 * (3 + 8) / 2)


def test_confidence_targets_include_dustbin_and_are_detached() -> None:
    first = torch.tensor([[[0.0, 0.0, 5.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]], requires_grad=True)
    final = first.detach().clone()
    final[0, 0] = torch.tensor([6.0, 0.0, 5.0])
    targets = confidence_targets([first, final])
    assert targets[0][0].tolist() == [[0.0, 1.0]]
    assert not targets[0][0].requires_grad


def test_empty_and_all_ignored_supervision_skip() -> None:
    net = LighterGlue(device="cpu", flash=False, width_confidence=-1, depth_confidence=-1).net
    output = forward_layers(net, inputs())
    assert lightglue_loss(net, output, torch.full((5,), -2), torch.full((7,), -2)) is None
    assert forward_layers(net, inputs(0, 7)) is None


def test_confidence_loss_detaches_backbone_descriptors() -> None:
    net = LighterGlue(device="cpu", flash=False, width_confidence=-1, depth_confidence=-1).net
    data = inputs()
    output = forward_layers(net, data)
    losses = lightglue_loss(net, output, torch.tensor([0, 1, -1, -2, -1]), torch.tensor([0, 1, -1, -1, -1, -2, -1]))
    assert losses is not None
    losses["confidence"].backward()
    assert data["image0"]["descriptors"].grad is None
    assert all(p.grad is None for p in net.transformers.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.token_confidence.parameters())
