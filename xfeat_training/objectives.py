"""Geometric coarse descriptors, frozen-teacher keypoints and synthetic fine bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.model import XFeatModel
from xfeat_training.augment import transform_points
from xfeat_training.data import Frame
from xfeat_training.geometry import project_points, sample_depth


@dataclass
class Correspondences:
    source: Tensor
    target: Tensor
    fine_class: Tensor | None = None
    reverse: Correspondences | None = None


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def fine_labels(points: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
    bins = torch.round(points - anchors).long() + 4
    valid = ((bins >= 0) & (bins <= 7)).all(-1)
    return bins[:, 0] + 8 * bins[:, 1], valid


def _grid(height: int, width: int) -> Tensor:
    y, x = torch.meshgrid(torch.arange(0, height, 8), torch.arange(0, width, 8), indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), -1).float()


def _select(
    source: Tensor,
    target: Tensor,
    valid: Tensor,
    max_points: int,
    generator: torch.Generator | None,
    labels: Tensor | None = None,
) -> Correspondences:
    if max_points < 2:
        raise ValueError("max_points must be at least two")
    indices = []
    used_source, used_target = set(), set()
    for i in torch.where(valid)[0].tolist():
        a, b = tuple(source[i].tolist()), tuple(target[i].tolist())
        if a not in used_source and b not in used_target:
            indices.append(i)
            used_source.add(a)
            used_target.add(b)
    choices = torch.tensor(indices, dtype=torch.long)
    if len(choices) > max_points:
        choices = choices[torch.randperm(len(choices), generator=generator)[:max_points]]
    return Correspondences(source[choices], target[choices], labels[choices] if labels is not None else None)


def _synthetic_direction(
    homography: Tensor, height: int, width: int, max_points: int, generator: torch.Generator | None
) -> Correspondences:
    target = _grid(height, width)
    points = transform_points(target, torch.linalg.inv(homography.detach().cpu()))
    source = 8 * torch.floor(points / 8 + 0.5)
    labels, valid = fine_labels(points, source)
    valid &= torch.isfinite(points).all(-1)
    for coordinates in (points, source):
        valid &= (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
        valid &= (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height)
    return _select(source, target, valid, max_points, generator, labels)


def synthetic_correspondences(
    homography: Tensor, height: int, width: int, *, max_points: int = 256, generator: torch.Generator | None = None
) -> Correspondences:
    result = _synthetic_direction(homography, height, width, max_points, generator)
    result.reverse = _synthetic_direction(torch.linalg.inv(homography), height, width, max_points, generator)
    return result


def rgbd_correspondences(
    source: Frame, target: Frame, *, max_points: int = 256, generator: torch.Generator | None = None
) -> Correspondences:
    """Round reverse-projected target grid points to unique valid source anchors."""
    target_xy = _grid(*target.depth.shape)
    q = target_xy.numpy().astype(np.float64)
    projection = project_points(q, target, source)
    p = projection.xy
    # Keep invalid geometry masked; zero only provides an index placeholder.
    anchors = 8 * np.floor(np.where(np.isfinite(p), p, 0) / 8 + 0.5)
    _, target_valid = sample_depth(target.depth, q, interior=True)
    source_depth, source_valid = sample_depth(source.depth, p, interior=True)
    _, anchor_valid = sample_depth(source.depth, anchors, interior=True)
    valid = projection.valid & target_valid & source_valid & anchor_valid & (projection.z > 0)
    valid &= np.abs(projection.z - source_depth) <= 0.03 * source_depth
    safe_p = np.where(np.isfinite(p), p, 0)
    returned = project_points(safe_p, source, target)
    valid &= returned.valid & (np.linalg.norm(returned.xy - q, axis=1) <= 3)
    anchor_projection = project_points(anchors, source, target)
    check_depth, check_valid = sample_depth(target.depth, anchor_projection.xy, interior=True)
    valid &= anchor_projection.valid & check_valid & (anchor_projection.z > 0)
    valid &= np.abs(anchor_projection.z - check_depth) <= 0.03 * check_depth
    return _select(torch.from_numpy(anchors).float(), target_xy, torch.from_numpy(valid), max_points, generator)


def descriptor_objective(desc0: Tensor, desc1: Tensor) -> tuple[Tensor, Tensor]:
    if len(desc0) < 2 or len(desc0) != len(desc1):
        raise ValueError("Descriptor objective requires at least two paired anchors")
    with torch.autocast(device_type=desc0.device.type, enabled=False):
        logits = desc0.float() @ desc1.float().T * 0.2
        row, column = logits.log_softmax(1), logits.log_softmax(0)
        loss = -(row.diagonal() + column.diagonal()).mean()
        confidence = (row.exp().max(1).values * column.exp().max(0).values).detach()
    return loss, confidence


def keypoint_consistency(student: Tensor, teacher: Tensor, valid: Tensor | None = None) -> Tensor:
    kl = F.kl_div(student.float().log_softmax(1), teacher.detach().float().softmax(1), reduction="none").sum(1)
    if valid is None:
        return kl.mean()
    mask = valid.float()
    if mask.shape[-2:] != kl.shape[-2:]:
        mask = F.avg_pool2d(mask, 8, 8)
    mask = (mask[:, 0] == 1).to(kl.device)
    return kl[mask].mean() if mask.any() else kl.sum() * 0


def _at(features: Tensor, anchors: Tensor) -> Tensor:
    cells = (anchors.to(features.device) / 8).long()
    return features[0, :, cells[:, 1], cells[:, 0]].T


def _fine_loss(model: XFeatModel, map0: Tensor, map1: Tensor, correspondence: Correspondences) -> Tensor | None:
    if correspondence.fine_class is None or len(correspondence.source) < 2:
        return None
    desc0, desc1 = _at(map0, correspondence.source), _at(map1, correspondence.target)
    with torch.no_grad():
        _, confidence = descriptor_objective(desc0, desc1)
    if confidence.sum() <= 0:
        return None
    features = torch.cat((F.normalize(desc0, dim=-1), F.normalize(desc1, dim=-1)), -1)
    logits = model.fine_matcher(features)
    ce = F.cross_entropy(logits.float(), correspondence.fine_class.to(logits.device), reduction="none")
    return 2 * (confidence * ce).sum() / confidence.sum()


def xfeat_objective(
    model: XFeatModel,
    teacher: XFeatModel,
    image0: Tensor,
    image1: Tensor,
    student0: Tensor,
    student1: Tensor,
    correspondence: Correspondences,
    *,
    synthetic: bool,
    valid0: Tensor | None = None,
    valid1: Tensor | None = None,
) -> dict[str, Tensor] | None:
    if len(correspondence.source) < 2:
        return None
    freeze_batchnorm(model)
    teacher.eval()
    feat0, key0, reliability0 = model(student0)
    feat1, key1, reliability1 = model(student1)
    with torch.no_grad():
        target0, target1 = teacher(image0)[1], teacher(image1)[1]
    desc0, desc1 = _at(feat0, correspondence.source), _at(feat1, correspondence.target)
    descriptor, confidence = descriptor_objective(desc0, desc1)
    kl = (keypoint_consistency(key0, target0, valid0) + keypoint_consistency(key1, target1, valid1)) / 2
    reliability = descriptor * 0
    fine = descriptor * 0
    if synthetic:
        losses = [_fine_loss(model, feat0, feat1, correspondence)]
        if correspondence.reverse is not None:
            losses.append(_fine_loss(model, feat1, feat0, correspondence.reverse))
        available = [value for value in losses if value is not None]
        if not available:
            return None
        fine = torch.stack(available).mean()
    else:
        confidence0, confidence1 = (
            _at(reliability0, correspondence.source).flatten(),
            _at(reliability1, correspondence.target).flatten(),
        )
        reliability = 3 * (F.l1_loss(confidence0, confidence) + F.l1_loss(confidence1, confidence)) / 2
    loss = descriptor + reliability + kl + fine
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite XFeat objective")
    return {
        "loss": loss,
        "descriptor": descriptor,
        "reliability": reliability,
        "keypoint": kl,
        "fine": fine,
        "supervised": torch.tensor(len(correspondence.source), device=loss.device),
    }
