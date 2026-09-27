"""Differentiable LighterGlue forward with supervision at every assignment layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from kornia.feature.lightglue import normalize_keypoints
from torch import Tensor


@dataclass
class LayerOutputs:
    assignments: list[Tensor]
    descriptors0: list[Tensor]
    descriptors1: list[Tensor]


def forward_layers(net: Any, data: dict[str, dict[str, Tensor]]) -> LayerOutputs | None:
    if net.conf.width_confidence != -1 or net.conf.depth_confidence != -1:
        raise ValueError("Training requires width/depth_confidence=-1")
    a, b = data["image0"], data["image1"]
    if a["keypoints"].shape[0] != 1 or b["keypoints"].shape[0] != 1:
        raise ValueError("Training supports batch=1 with variable real point counts")
    if a["keypoints"].shape[1] == 0 or b["keypoints"].shape[1] == 0:
        return None
    for features in (a, b):
        if not all(torch.isfinite(value).all() for value in features.values()):
            raise ValueError("Nonfinite matcher input")
    keypoints0 = normalize_keypoints(a["keypoints"], a["image_size"])
    keypoints1 = normalize_keypoints(b["keypoints"], b["image_size"])
    # Intentionally retain gradients at input_proj and through input descriptors.
    desc0, desc1 = net.input_proj(a["descriptors"].contiguous()), net.input_proj(b["descriptors"].contiguous())
    encoding0, encoding1 = net.posenc(keypoints0), net.posenc(keypoints1)
    result = LayerOutputs([], [], [])
    for transformer, assignment in zip(net.transformers, net.log_assignment, strict=True):
        desc0, desc1 = transformer(desc0, desc1, encoding0, encoding1)
        log_assignment, _ = assignment(desc0, desc1)
        result.assignments.append(log_assignment)
        result.descriptors0.append(desc0)
        result.descriptors1.append(desc1)
    return result


def assignment_nll(log_assignment: Tensor, gt0: Tensor, gt1: Tensor) -> Tensor:
    gt0, gt1 = gt0.reshape(-1), gt1.reshape(-1)
    if log_assignment.shape != (1, len(gt0) + 1, len(gt1) + 1):
        raise ValueError("Assignment and GT dimensions disagree")
    if (gt0 >= len(gt1)).any() or (gt1 >= len(gt0)).any() or (gt0 < -2).any() or (gt1 < -2).any():
        raise ValueError("Invalid GT index")
    positive = gt0 >= 0
    negative0, negative1 = gt0 == -1, gt1 == -1
    zero = log_assignment.sum() * 0
    pos_loss = -log_assignment[0, torch.where(positive)[0], gt0[positive]].mean() if positive.any() else zero
    negative = torch.cat((log_assignment[0, :-1, -1][negative0], log_assignment[0, -1, :-1][negative1]))
    neg_loss = -negative.mean() if negative.numel() else zero
    return 0.5 * pos_loss + 0.5 * neg_loss


def confidence_targets(assignments: list[Tensor]) -> list[tuple[Tensor, Tensor]]:
    final = assignments[-1].detach()
    final0, final1 = final[:, :-1, :].argmax(-1), final[:, :, :-1].argmax(-2)
    return [
        (
            (scores.detach()[:, :-1, :].argmax(-1) == final0).float(),
            (scores.detach()[:, :, :-1].argmax(-2) == final1).float(),
        )
        for scores in assignments[:-1]
    ]


def lightglue_loss(net: Any, outputs: LayerOutputs | None, gt0: Tensor, gt1: Tensor) -> dict[str, Tensor] | None:
    if outputs is None or not ((gt0 != -2).any() or (gt1 != -2).any()):
        return None
    nll = torch.stack([assignment_nll(scores, gt0, gt1) for scores in outputs.assignments]).mean()
    targets = confidence_targets(outputs.assignments)
    confidence_losses = []
    for i, (target0, target1) in enumerate(targets):
        # token[0] is Linear; applying BCEWithLogits to token's Sigmoid would be wrong.
        head = net.token_confidence[i].token[0]
        loss0 = F.binary_cross_entropy_with_logits(head(outputs.descriptors0[i].detach()).squeeze(-1), target0)
        loss1 = F.binary_cross_entropy_with_logits(head(outputs.descriptors1[i].detach()).squeeze(-1), target1)
        confidence_losses.append((loss0 + loss1) / 2)
    confidence = torch.stack(confidence_losses).mean()
    loss = nll + confidence
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite LightGlue assignment/confidence loss")
    return {
        "loss": loss,
        "nll": nll,
        "confidence": confidence,
        "supervised": (gt0 != -2).sum() + (gt1 != -2).sum(),
        "ignored": (gt0 == -2).sum() + (gt1 == -2).sum(),
    }
