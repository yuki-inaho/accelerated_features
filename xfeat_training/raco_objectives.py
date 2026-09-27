"""Geometric pseudo-correspondences and symmetric positional-error likelihood."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from modules.raco import finite


class _BudgetSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, scores: Tensor, budget: int, temperature: float) -> Tensor:
        # Float64 root finding prevents mass drift at large N and small temperatures.
        value = scores.double()
        margin = temperature * (math.log(len(value)) + 40)
        low, high = value.min() - margin, value.max() + margin
        for _ in range(64):
            middle = (low + high) / 2
            mass = ((value - middle) / temperature).sigmoid().sum()
            low, high = torch.where(mass > budget, middle, low), torch.where(mass > budget, high, middle)
        alpha = ((value - (low + high) / 2) / temperature).sigmoid()
        ctx.save_for_backward(alpha)
        ctx.temperature = temperature
        return alpha.to(scores.dtype)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Tensor, None, None]:
        (gradient,) = grad_outputs
        (alpha,) = ctx.saved_tensors
        weight = alpha * (1 - alpha) / ctx.temperature
        total = weight.sum()
        # An exactly saturated selection has a locally zero derivative.
        denominator = total.clamp_min(torch.finfo(weight.dtype).tiny)
        centered = gradient.double() - (weight * gradient.double()).sum() / denominator
        return (weight * centered).to(gradient.dtype), None, None


def soft_topk(scores: Tensor, budget: int, temperature: float) -> Tensor:
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("Scores must be a floating vector")
    finite(scores, "scores")
    if type(budget) is not int or budget < 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid budget or temperature")
    if budget == 0 or not len(scores):
        return scores * 0
    if budget >= len(scores):
        return scores * 0 + 1
    return _BudgetSelection.apply(scores, budget, temperature)


def _validate_matches(matches: Tensor, n0: int, n1: int, budgets: list[int]) -> None:
    if not budgets or any(type(k) is not int or k <= 0 for k in budgets):
        raise ValueError("Budgets must be positive integers")
    if matches.ndim != 2 or matches.shape[-1] != 2 or matches.dtype != torch.int64:
        raise ValueError("Matches must be int64 Nx2")
    if (matches < 0).any() or (matches >= matches.new_tensor([n0, n1])).any():
        raise ValueError("Match index outside candidates")
    if any(len(matches[:, i].unique()) != len(matches) for i in (0, 1)):
        raise ValueError("Matches must be one-to-one")


def ranking_loss(scores0: Tensor, scores1: Tensor, matches: Tensor, budgets: list[int], temperature: float) -> Tensor:
    _validate_matches(matches, len(scores0), len(scores1), budgets)
    rewards = []
    for budget in budgets:
        a, b = soft_topk(scores0, budget, temperature), soft_topk(scores1, budget, temperature)
        rewards.append((a[matches[:, 0]] * b[matches[:, 1]]).sum() / budget)
    return 1 - torch.stack(rewards).mean()


@torch.no_grad()
def hard_utility(scores0: Tensor, scores1: Tensor, matches: Tensor, budgets: list[int]) -> float:
    _validate_matches(matches, len(scores0), len(scores1), budgets)
    finite(scores0, "rank scores")
    finite(scores1, "rank scores")
    rewards = []
    for k in budgets:
        a, b = torch.zeros_like(scores0, dtype=torch.bool), torch.zeros_like(scores1, dtype=torch.bool)
        a[torch.argsort(scores0, descending=True, stable=True)[:k]] = True
        b[torch.argsort(scores1, descending=True, stable=True)[:k]] = True
        rewards.append(float((a[matches[:, 0]] & b[matches[:, 1]]).sum()) / k)
    return sum(rewards) / len(rewards)


def project_with_jacobian(points: Tensor, homography: Tensor) -> tuple[Tensor, Tensor]:
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("Expected Nx2 points")
    finite(points, "points")
    h = homography.to(points)
    if h.shape != (3, 3) or not torch.isfinite(h).all() or abs(float(torch.linalg.det(h))) < 1e-10:
        raise ValueError("Invalid or singular homography")
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), -1) @ h.T
    denom = homogeneous[:, 2:3]
    if (denom.abs() <= 1e-8).any():
        raise ValueError("Homography maps points to infinity")
    projected = homogeneous[:, :2] / denom
    jacobian = (h[:2, :2][None] - projected[:, :, None] * h[2, :2][None, None]) / denom[:, :, None]
    finite(projected, "projected points")
    finite(jacobian, "homography Jacobian")
    return projected, jacobian


def _nearest(source: Tensor, target: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    indices, distances = [], []
    for part in source.split(block_size):
        # Direct distances avoid cancellation in GEMM-based cdist at subpixel scales.
        distance = torch.cdist(part, target, compute_mode="donot_use_mm_for_euclid_dist")
        value, index = distance.min(-1)
        indices.append(index)
        distances.append(value)
    return torch.cat(indices), torch.cat(distances)


@torch.no_grad()
def geometric_matches(
    points0: Tensor, points1: Tensor, homography: Tensor, threshold: float, block_size: int = 256
) -> Tensor:
    if not math.isfinite(threshold) or threshold <= 0 or block_size < 1:
        raise ValueError("threshold and block_size must be positive")
    projected0, _ = project_with_jacobian(points0, homography)
    projected1, _ = project_with_jacobian(points1, torch.linalg.inv(homography))
    if not len(points0) or not len(points1):
        return torch.empty((0, 2), dtype=torch.long, device=points0.device)
    forward, distance0 = _nearest(projected0, points1, block_size)
    backward, distance1 = _nearest(projected1, points0, block_size)
    index = torch.arange(len(points0), device=points0.device)
    valid = (backward[forward] == index) & (distance0 <= threshold) & (distance1[forward] <= threshold)
    return torch.stack((index[valid], forward[valid]), -1)


def covariance_nll(
    points0: Tensor, points1: Tensor, cov0: Tensor, cov1: Tensor, matches: Tensor, homography: Tensor
) -> tuple[Tensor, dict[str, float]]:
    """Mean of forward/backward Gaussian NLL; directions are not independent samples."""
    if matches.ndim != 2 or matches.shape[-1] != 2 or not len(matches):
        raise ValueError("covariance NLL requires nonempty Nx2 matches")
    if cov0.shape != (len(points0), 2, 2) or cov1.shape != (len(points1), 2, 2):
        raise ValueError("Covariances must align with candidate points")
    finite(cov0, "source covariance")
    finite(cov1, "target covariance")
    if not torch.allclose(cov0, cov0.mT) or not torch.allclose(cov1, cov1.mT):
        raise ValueError("Covariances must be symmetric")
    i, j = matches.unbind(-1)
    losses, mahalanobis, residuals = [], [], []
    for a, b, ca, cb, h in (
        (points0[i], points1[j], cov0[i], cov1[j], homography),
        (points1[j], points0[i], cov1[j], cov0[i], torch.linalg.inv(homography)),
    ):
        projected, jac = project_with_jacobian(a, h)
        error = b - projected
        covariance = cb + jac @ ca @ jac.mT
        chol = torch.linalg.cholesky(covariance)
        q = (error[..., None] * torch.cholesky_solve(error[..., None], chol)).sum((-1, -2))
        logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        losses.append(0.5 * (q + logdet + 2 * math.log(2 * math.pi)))
        mahalanobis.append(q.detach())
        residuals.append(error.detach())
    loss = torch.stack(losses).mean()
    finite(loss, "covariance loss")
    q = torch.cat(mahalanobis)
    error = residuals[0]  # reverse residuals must not cancel forward bias in diagnostics
    eigenvalues = torch.linalg.eigvalsh(torch.cat((cov0[i], cov1[j])).detach())
    stats = {
        "mahalanobis": float(q.mean()),
        "coverage95": float((q <= 5.991464547).float().mean()),
        "residual_mean_x": float(error[:, 0].mean()),
        "residual_mean_y": float(error[:, 1].mean()),
        "min_eigenvalue": float(eigenvalues[:, 0].min()),
        "mean_condition": float((eigenvalues[:, 1] / eigenvalues[:, 0]).mean()),
    }
    return loss, stats
