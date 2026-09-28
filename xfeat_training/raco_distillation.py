"""Tensor losses on shared image coordinates (not on rendered RaCo colors).

Matrices have shape ``[..., 2, 2]``; leading dimensions broadcast. The Gaussian
means coincide. KL direction, nonlinear transformation and weighting are explicit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _cholesky(covariance: Tensor) -> Tensor:
    if covariance.shape[-2:] != (2, 2):
        raise ValueError("Expected [..., 2, 2] covariance matrices")
    if covariance.dtype not in (torch.float32, torch.float64):
        covariance = covariance.float()
    factor, info = torch.linalg.cholesky_ex(covariance)
    if (info != 0).any() or not torch.isfinite(factor).all():
        raise ValueError("Nonfinite or non-positive-definite covariance in distillation")
    return factor


def gaussian_kl(source: Tensor, destination: Tensor) -> Tensor:
    """KL(N(0, source) || N(0, destination)), with standard 1/2 coefficient.

    For teacher-to-student KD, call ``gaussian_kl(teacher.detach(), student)``.
    No matrix inverse, determinant ratio, angle regression, or per-point loop.
    """
    source_l, destination_l = _cholesky(source), _cholesky(destination)
    relative = torch.linalg.solve_triangular(destination_l, source_l, upper=False)
    log_ratio = 2 * (
        destination_l.diagonal(dim1=-2, dim2=-1).log().sum(-1) - source_l.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    result = 0.5 * (relative.square().sum((-2, -1)) - 2 + log_ratio)
    if not torch.isfinite(result).all() or (result < -1e-5).any():
        raise ValueError("Gaussian KL is nonfinite or below the roundoff tolerance")
    return result.clamp_min(0)


def covariance_distance(student: Tensor, teacher: Tensor, *, direction: str, transform: str) -> Tensor:
    """Per-point distance; ``mse`` is explicitly pixel^4, not Gaussian KL."""
    target = teacher.detach()
    if direction == "teacher_to_student":
        distance = gaussian_kl(target, student)
    elif direction == "student_to_teacher":
        distance = gaussian_kl(student, target)
    elif direction == "symmetric":
        distance = 0.5 * (gaussian_kl(target, student) + gaussian_kl(student, target))
    elif direction == "mse":
        if transform != "identity":
            raise ValueError("The component-MSE control uses the identity transform")
        return (student - target).square().mean((-2, -1))
    else:
        raise ValueError(f"Unknown covariance distance direction: {direction}")
    if transform == "identity":
        return distance
    if transform == "log1p":
        return distance.log1p()
    if transform == "bounded_log1p":
        return 1 - (1 + distance.log1p()).reciprocal()
    raise ValueError(f"Unknown covariance distance transform: {transform}")


def teacher_weighted_mean(values: Tensor, teacher_logits: Tensor) -> Tensor:
    """Normalize teacher detector confidence over the same candidate set only."""
    return (values * teacher_logits.detach().softmax(-1)).sum(-1).mean()


def standardized(scores: Tensor) -> Tensor:
    return (scores - scores.mean(-1, keepdim=True)) / scores.var(-1, unbiased=False, keepdim=True).clamp_min(
        1e-6
    ).sqrt()


def rank_list_kl(student: Tensor, teacher: Tensor) -> Tensor:
    target = standardized(teacher.detach()).log_softmax(-1)
    prediction = standardized(student).log_softmax(-1)
    return (target.exp() * (target - prediction)).sum(-1).mean()


def rank_pair_kl(
    student: Tensor,
    teacher: Tensor,
    *,
    comparisons: int = 4096,
    budget: int = 512,
    boundary_radius: int = 128,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Soft RankNet on final student scores, O(N log N + comparisons).

    Half the pairs span all candidates; half sample around the teacher's budget
    boundary. Teacher ties/self-pairs get zero weight. Return Bernoulli KL rather
    than BCE so perfect agreement has zero loss despite soft target entropy.
    """
    if student.ndim != 1 or teacher.shape != student.shape or student.numel() == 0:
        raise ValueError("Rank distillation expects matching nonempty score vectors")
    if comparisons < 2 or comparisons % 2 or budget < 1 or boundary_radius < 1:
        raise ValueError("Invalid rank pair sampling configuration")
    n = student.numel()
    target = standardized(teacher.detach())
    order = teacher.argsort(descending=True, stable=True)
    boundary = budget if n > budget else n // 2
    low, high = max(0, boundary - boundary_radius), min(n, boundary + boundary_radius)
    device = student.device if generator is None else generator.device
    all_pairs = torch.randint(n, (comparisons // 2, 2), device=device, generator=generator).to(student.device)
    boundary_ids = torch.randint(low, high, (comparisons // 2, 2), device=device, generator=generator).to(
        student.device
    )
    pairs = torch.cat((all_pairs, order[boundary_ids]))
    i, j = pairs.unbind(-1)
    logits = student[i] - student[j]
    target_logits = target[i] - target[j]
    q = target_logits.sigmoid()
    weight = 2 * (q - 0.5).abs()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, q, reduction="none")
    entropy = F.binary_cross_entropy_with_logits(target_logits, q, reduction="none")
    return ((cross_entropy - entropy) * weight).sum() / weight.sum().clamp_min(1e-8)
