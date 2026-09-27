"""Fine-bin geometry, independent keypoint consistency and XFeat gradients."""

from __future__ import annotations

import copy

import torch

from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.objectives import (
    descriptor_objective,
    fine_labels,
    freeze_batchnorm,
    keypoint_consistency,
    rgbd_correspondences,
    synthetic_correspondences,
    xfeat_objective,
)


def test_all_64_fine_bins_and_outside_offsets() -> None:
    y, x = torch.meshgrid(torch.arange(-4, 4), torch.arange(-4, 4), indexing="ij")
    offsets = torch.stack((x.flatten(), y.flatten()), dim=-1).float()
    anchors = torch.full_like(offsets, 16)
    labels, valid = fine_labels(anchors + offsets, anchors)
    assert valid.all() and labels.tolist() == list(range(64))
    decoded = torch.stack((labels % 8 - 4, labels // 8 - 4), dim=-1)
    assert torch.equal(decoded.float(), offsets)
    _, valid = fine_labels(torch.tensor([[20.0, 16], [11.0, 16]]), torch.tensor([[16.0, 16], [16.0, 16]]))
    assert not valid.any()


def test_synthetic_sampling_covers_image_and_reproduces() -> None:
    H = torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, -0.6], [0.0, 0.0, 1.0]])
    a = synthetic_correspondences(H, 96, 128, max_points=16, generator=torch.Generator().manual_seed(9))
    b = synthetic_correspondences(H, 96, 128, max_points=16, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a.source, b.source) and torch.equal(a.target, b.target)
    assert len(a.source) == 16
    assert a.target[:, 1].max() > 48 and a.target[:, 0].max() > 64
    assert len(torch.unique(a.source, dim=0)) == len(a.source)
    assert len(torch.unique(a.target, dim=0)) == len(a.target)


def test_identity_kl_zero_but_photometric_keypoint_gradient_nonzero() -> None:
    torch.manual_seed(6)
    student = XFeat(device="cpu").net
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    image = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        target = teacher(image)[1]
    logits = student(image)[1]
    assert abs(keypoint_consistency(logits, target).item()) < 1e-6
    perturbed = (image + 0.02 * torch.randn_like(image)).clamp(0, 1)
    loss = keypoint_consistency(student(perturbed)[1], target)
    loss.backward()
    assert loss > 0
    head = student.keypoint_head[-1]
    assert isinstance(head, torch.nn.Conv2d)
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_descriptor_loss_uses_raw_dot_products() -> None:
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    b = torch.tensor([[3.0, 0.0], [0.0, 4.0]], requires_grad=True)
    loss, confidence = descriptor_objective(a, b)
    logits = a @ b.T * 0.2
    expected = -(logits.log_softmax(0).diagonal() + logits.log_softmax(1).diagonal()).mean()
    torch.testing.assert_close(loss, expected)
    assert not confidence.requires_grad


def test_xfeat_objectives_reach_all_heads_without_changing_teacher_or_bn() -> None:
    torch.manual_seed(22)
    student = XFeat(device="cpu").net
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student.train()
    freeze_batchnorm(student)
    teacher_hash = state_hash(teacher.state_dict())
    buffers = {n: b.clone() for n, b in student.named_buffers()}
    image0, image1 = torch.rand(1, 3, 64, 96), torch.rand(1, 3, 64, 96)
    H = torch.tensor([[1.0, 0.0, 0.4], [0.0, 1.0, -0.25], [0.0, 0.0, 1.0]])
    corr = synthetic_correspondences(H, 64, 96, max_points=32, generator=torch.Generator().manual_seed(11))
    for synthetic in (False, True):
        result = xfeat_objective(
            student,
            teacher,
            image0,
            image1,
            (image0 + 0.01 * torch.randn_like(image0)).clamp(0, 1),
            (image1 + 0.01 * torch.randn_like(image1)).clamp(0, 1),
            corr,
            synthetic=synthetic,
        )
        assert result is not None and torch.isfinite(result["loss"])
        result["loss"].backward()
        if not synthetic:
            assert all(p.grad is None for p in student.fine_matcher.parameters())
    for name in ("block_fusion.2.weight", "heatmap_head.2.weight", "keypoint_head.3.weight", "fine_matcher.12.weight"):
        p = dict(student.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert state_hash(teacher.state_dict()) == teacher_hash
    assert all(torch.equal(buffers[n], b) for n, b in student.named_buffers())


def test_rgbd_coarse_geometry_has_no_fine_labels(plane_frame) -> None:
    source, target = plane_frame(), plane_frame(tx=0.08)
    corr = rgbd_correspondences(source, target, max_points=32, generator=torch.Generator().manual_seed(9))
    assert len(corr.source) == 32
    torch.testing.assert_close(corr.target - corr.source, torch.tensor([[8.0, 0.0]]).expand(32, 2))
    assert corr.fine_class is None and corr.reverse is None
    source.depth[:] = 0
    empty = rgbd_correspondences(source, target)
    assert len(empty.source) == 0


def test_keypoint_padding_cells_are_excluded() -> None:
    teacher = torch.randn(1, 65, 2, 2)
    student = teacher.clone()
    student[0, 0, 0, 0] += 10
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    valid[:, :, :8, :8] = False
    assert keypoint_consistency(student, teacher) > 0
    assert abs(keypoint_consistency(student, teacher, valid).item()) < 1e-6
    assert keypoint_consistency(student, teacher, torch.zeros_like(valid)).item() == 0
