"""Training-only multi-scale feature alignment for RaCo distillation."""

from __future__ import annotations

import math
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TwoScaleFeatureAlignment(nn.Module):
    """Align student H/8 and H/32 maps with teacher block3/block4 maps."""

    def __init__(self) -> None:
        super().__init__()
        self.student_h32_to_t128 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.student_h32_to_h8 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.student_h8_to_h8 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.h8_gate_logits = nn.Parameter(torch.zeros(1, 64, 1, 1))

    def forward(self, student_h8: Tensor, student_h32: Tensor, valid_mask: Tensor) -> dict[str, Tensor]:
        _validate_feature_inputs(student_h8, student_h32, None, None, valid_mask)
        batch, _, height, width = valid_mask.shape
        valid8 = F.adaptive_avg_pool2d(valid_mask.float(), (height // 8, width // 8))
        valid32 = F.adaptive_avg_pool2d(valid_mask.float(), (height // 32, width // 32))

        h32_for_teacher = self.student_h32_to_t128(student_h32)
        h32_for_h8 = self.student_h32_to_h8(student_h32)
        # Mask before interpolation so invalid coarse cells cannot leak into valid H/8 cells.
        masked_h32 = h32_for_h8 * valid32
        upsampled_support = F.interpolate(valid32, size=valid8.shape[-2:], mode="bilinear", align_corners=False)
        upsampled_h32 = F.interpolate(masked_h32, size=valid8.shape[-2:], mode="bilinear", align_corners=False)
        upsampled_h32 = upsampled_h32 / upsampled_support.clamp_min(1e-6)
        h8_support = valid8 * upsampled_support

        h8_local = self.student_h8_to_h8(student_h8)
        gate = torch.sigmoid(self.h8_gate_logits)
        fused_h8 = (gate * h8_local + (1.0 - gate) * upsampled_h32) * (valid8 > 0).to(h8_local.dtype)

        if h32_for_teacher.shape[0] != batch:
            raise ValueError("student feature batch does not match valid mask")
        return {
            "block3": fused_h8,
            "block4": h32_for_teacher,
            "valid8": valid8,
            "valid8_support": h8_support,
            "valid32": valid32,
        }


def two_scale_feature_loss(
    student_h8: Tensor,
    student_h32: Tensor,
    teacher_h8: Tensor,
    teacher_h32: Tensor,
    valid_mask: Tensor,
    adapter: TwoScaleFeatureAlignment,
) -> dict[str, Tensor]:
    """Compute equally weighted, mask-aware HCL losses at H/8 and H/32."""

    _validate_feature_inputs(student_h8, student_h32, teacher_h8, teacher_h32, valid_mask)
    aligned = adapter(student_h8, student_h32, valid_mask)
    h8_loss, h8_denominator = _masked_hcl_loss(aligned["block3"], teacher_h8.detach(), aligned["valid8_support"])
    h32_loss, h32_denominator = _masked_hcl_loss(aligned["block4"], teacher_h32.detach(), aligned["valid32"])
    return {
        "loss": 0.5 * (h8_loss + h32_loss),
        "h8_loss": h8_loss,
        "h32_loss": h32_loss,
        "h8_denominator": h8_denominator,
        "h32_denominator": h32_denominator,
        "aligned_h8": aligned["block3"],
        "aligned_h32": aligned["block4"],
        "valid8": aligned["valid8"],
        "valid32": aligned["valid32"],
    }


def _masked_hcl_loss(student: Tensor, teacher: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    student = student.float()
    teacher = teacher.detach().float()
    mask = mask.float()
    if student.shape != teacher.shape:
        raise ValueError(f"aligned feature shape mismatch: {student.shape} != {teacher.shape}")
    if mask.shape != (student.shape[0], 1, *student.shape[-2:]):
        raise ValueError("feature mask shape does not match aligned feature maps")
    _require_finite("student aligned feature", student)
    _require_finite("teacher feature", teacher)
    _require_finite("feature mask", mask)

    student = _masked_channel_standardize(student, mask)
    teacher = _masked_channel_standardize(teacher, mask)
    losses = []
    for size in (None, 1, 2, 4):
        if size is None:
            pooled_student, pooled_teacher, pooled_mask = student, teacher, mask
        else:
            output_size = (size, size)
            pooled_mask = F.adaptive_avg_pool2d(mask, output_size)
            student_sum = F.adaptive_avg_pool2d(student * mask, output_size)
            teacher_sum = F.adaptive_avg_pool2d(teacher * mask, output_size)
            divisor = pooled_mask.clamp_min(1e-6)
            pooled_student = student_sum / divisor
            pooled_teacher = teacher_sum / divisor
        denominator = pooled_mask.sum() * student.shape[1]
        if denominator.item() <= 0:
            raise ValueError("feature loss valid mask is empty")
        losses.append(((pooled_student - pooled_teacher).square() * pooled_mask).sum() / denominator)
    base_denominator = mask.sum() * student.shape[1]
    return torch.stack(losses).mean(), base_denominator.detach()


def _masked_channel_standardize(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.float()
    denominator = weight.sum(dim=(-2, -1), keepdim=True)
    if (denominator <= 0).any():
        raise ValueError("feature loss valid mask is empty for at least one sample")
    mean = (value * weight).sum(dim=(-2, -1), keepdim=True) / denominator
    variance = ((value - mean).square() * weight).sum(dim=(-2, -1), keepdim=True) / denominator
    return (value - mean) * torch.rsqrt(variance.clamp_min(1e-6))


def _validate_feature_inputs(
    student_h8: Tensor,
    student_h32: Tensor,
    teacher_h8: Tensor | None,
    teacher_h32: Tensor | None,
    valid_mask: Tensor,
) -> None:
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError("valid_mask must have shape B x 1 x H x W")
    batch, _, height, width = valid_mask.shape
    if height % 32 or width % 32:
        raise ValueError("input dimensions must be divisible by 32 for H/8 and H/32 alignment")
    expected8, expected32 = (height // 8, width // 8), (height // 32, width // 32)
    if student_h8.shape != (batch, 64, *expected8):
        raise ValueError(f"student H/8 features must have shape {(batch, 64, *expected8)}")
    if student_h32.shape != (batch, 64, *expected32):
        raise ValueError(f"student H/32 features must have shape {(batch, 64, *expected32)}")
    if teacher_h8 is not None and teacher_h8.shape != (batch, 64, *expected8):
        raise ValueError(f"teacher H/8 features must have shape {(batch, 64, *expected8)}")
    if teacher_h32 is not None and teacher_h32.shape != (batch, 128, *expected32):
        raise ValueError(f"teacher H/32 features must have shape {(batch, 128, *expected32)}")
    devices = {student_h8.device, student_h32.device, valid_mask.device}
    if teacher_h8 is not None:
        devices.add(teacher_h8.device)
    if teacher_h32 is not None:
        devices.add(teacher_h32.device)
    if len(devices) != 1:
        raise ValueError("all feature maps and valid_mask must be on the same device")
    if not valid_mask.is_floating_point() and valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean or floating point")
    if valid_mask.is_floating_point() and ((valid_mask < 0).any() or (valid_mask > 1).any()):
        raise ValueError("floating valid_mask values must be in [0, 1]")
    _require_finite("valid mask", valid_mask)


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


class LocalContrastProjection(nn.Module):
    """Project sparse block3 locations through a train-only contrastive head."""

    def __init__(self, input_channels: int = 64, hidden_dim: int = 128, embedding_dim: int = 128) -> None:
        super().__init__()
        if min(input_channels, hidden_dim, embedding_dim) <= 0:
            raise ValueError("projection dimensions must be positive")
        self.conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.mlp_in = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, feature_map: Tensor, points: Tensor, *, image_size: tuple[int, int]) -> Tensor:
        if feature_map.ndim != 4 or feature_map.shape[0] != 1:
            raise ValueError("local projection expects a single B x C x H x W feature map")
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape N x 2 in input-image pixel coordinates")
        if points.device != feature_map.device:
            raise ValueError("feature map and points must be on the same device")
        image_height, image_width = image_size
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_size dimensions must be positive")
        if len(points) and (
            (points[:, 0] < 0).any()
            or (points[:, 0] >= image_width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] >= image_height).any()
        ):
            raise ValueError("projection points must lie inside the input image")
        _require_finite("local projection feature map", feature_map)
        _require_finite("local projection points", points)

        projected = F.relu(self.conv(feature_map))
        grid = torch.stack(
            (
                2.0 * (points[:, 0].to(projected.dtype) + 0.5) / image_width - 1.0,
                2.0 * (points[:, 1].to(projected.dtype) + 0.5) / image_height - 1.0,
            ),
            dim=-1,
        ).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(projected, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.squeeze(2).transpose(1, 2).squeeze(0)
        embedding = self.mlp_out(F.relu(self.mlp_in(sampled)))
        return F.normalize(embedding, dim=-1)


@torch.no_grad()
def build_mutual_nearest_pairs(
    points_source: Tensor,
    points_target: Tensor,
    homography: Tensor,
    target_size: tuple[int, int],
    *,
    max_distance_px: float = 3.0,
    max_pairs: int = 256,
    seed: int = 0,
    ambiguity_margin_px: float = 0.5,
) -> dict[str, Tensor]:
    """Find deterministic mutual-nearest geometric pairs across two views."""

    if points_source.ndim != 2 or points_source.shape[-1] != 2:
        raise ValueError("points_source must have shape N x 2")
    if points_target.ndim != 2 or points_target.shape[-1] != 2:
        raise ValueError("points_target must have shape M x 2")
    if points_source.device != points_target.device or points_source.device != homography.device:
        raise ValueError("points and homography must be on the same device")
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape 3 x 3")
    target_height, target_width = target_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_size dimensions must be positive")
    if (
        not math.isfinite(max_distance_px)
        or not math.isfinite(ambiguity_margin_px)
        or max_distance_px < 0
        or ambiguity_margin_px < 0
        or type(max_pairs) is not int
        or max_pairs <= 0
        or type(seed) is not int
    ):
        raise ValueError("pair distance, ambiguity margin, max_pairs, or seed is invalid")
    _require_finite("source points", points_source)
    _require_finite("target points", points_target)
    _require_finite("homography", homography)

    empty_index = torch.empty(0, dtype=torch.long, device=points_source.device)
    empty_distance = points_source.new_empty((0,))
    if not len(points_source) or not len(points_target):
        return {"source_indices": empty_index, "target_indices": empty_index.clone(), "distances": empty_distance}

    source_homogeneous = torch.cat((points_source, torch.ones_like(points_source[:, :1])), dim=1).to(homography.dtype)
    mapped = source_homogeneous @ homography.T
    depth = mapped[:, 2]
    projected = mapped[:, :2] / depth[:, None].clamp_min(1e-8)
    valid_source = (
        torch.isfinite(projected).all(dim=1)
        & (depth > 1e-8)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < target_width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < target_height)
    )
    valid_target = (
        (points_target[:, 0] >= 0)
        & (points_target[:, 0] < target_width)
        & (points_target[:, 1] >= 0)
        & (points_target[:, 1] < target_height)
    )
    source_indices = torch.where(valid_source)[0]
    target_indices = torch.where(valid_target)[0]
    if not len(source_indices) or not len(target_indices):
        return {"source_indices": empty_index, "target_indices": empty_index.clone(), "distances": empty_distance}

    distances = torch.cdist(projected[source_indices].float(), points_target[target_indices].float())
    nearest_distance, nearest_target_local = distances.min(dim=1)
    nearest_source_local = distances.min(dim=0).indices
    source_rows = torch.arange(len(source_indices), device=points_source.device)
    mutual = nearest_source_local[nearest_target_local] == source_rows
    unambiguous = torch.ones_like(mutual)
    if len(target_indices) > 1:
        closest_two = distances.topk(2, dim=1, largest=False).values
        unambiguous = (closest_two[:, 1] - closest_two[:, 0]) > ambiguity_margin_px
    keep = mutual & unambiguous & (nearest_distance <= max_distance_px)
    matched_sources = source_indices[keep]
    matched_targets = target_indices[nearest_target_local[keep]]
    matched_distances = nearest_distance[keep]
    if len(matched_sources) > max_pairs:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selection = torch.randperm(len(matched_sources), generator=generator)[:max_pairs].to(matched_sources.device)
        matched_sources = matched_sources[selection]
        matched_targets = matched_targets[selection]
        matched_distances = matched_distances[selection]
    return {
        "source_indices": matched_sources,
        "target_indices": matched_targets,
        "distances": matched_distances,
    }


@torch.no_grad()
def local_false_negative_mask(points_source: Tensor, points_target: Tensor, *, exclusion_px: float = 16.0) -> Tensor:
    """Mark cross-view candidates that must not be used as negatives."""

    if points_source.ndim != 2 or points_source.shape[-1] != 2 or points_target.shape != points_source.shape:
        raise ValueError("paired source and target points must share shape N x 2")
    if points_source.device != points_target.device:
        raise ValueError("paired source and target points must be on the same device")
    if not math.isfinite(exclusion_px) or exclusion_px < 0:
        raise ValueError("exclusion_px must be nonnegative")
    _require_finite("paired source points", points_source)
    _require_finite("paired target points", points_target)
    source_distance = torch.cdist(points_source.float(), points_source.float())
    target_distance = torch.cdist(points_target.float(), points_target.float())
    excluded = (source_distance <= exclusion_px) | (target_distance <= exclusion_px)
    excluded.fill_diagonal_(False)
    return excluded


class LocalContrastResult(TypedDict):
    loss: Tensor
    active: bool
    pair_count: int
    valid_queries: int
    invalid_queries: int
    skip_reason: str | None


def symmetric_local_info_nce(
    embedding_source: Tensor,
    embedding_target: Tensor,
    points_source: Tensor,
    points_target: Tensor,
    *,
    temperature: float = 0.1,
    exclusion_px: float = 16.0,
    min_pairs: int = 32,
) -> LocalContrastResult:
    """Compute symmetric InfoNCE after dropping nearby false negatives."""

    if embedding_source.ndim != 2 or embedding_target.shape != embedding_source.shape:
        raise ValueError("paired embeddings must share shape N x D")
    if points_source.shape != (len(embedding_source), 2) or points_target.shape != points_source.shape:
        raise ValueError("point coordinates must match the embedding count and have shape N x 2")
    if len({embedding_source.device, embedding_target.device, points_source.device, points_target.device}) != 1:
        raise ValueError("paired embeddings and point coordinates must be on the same device")
    if (
        not math.isfinite(temperature)
        or not math.isfinite(exclusion_px)
        or temperature <= 0
        or exclusion_px < 0
        or type(min_pairs) is not int
        or min_pairs <= 0
    ):
        raise ValueError("temperature, exclusion_px, or min_pairs is invalid")
    _require_finite("source embeddings", embedding_source)
    _require_finite("target embeddings", embedding_target)
    _require_finite("source points", points_source)
    _require_finite("target points", points_target)

    pair_count = len(embedding_source)
    zero = (embedding_source.float().sum() + embedding_target.float().sum()) * 0.0
    if pair_count < min_pairs:
        return {
            "loss": zero,
            "active": False,
            "pair_count": pair_count,
            "valid_queries": 0,
            "invalid_queries": 0,
            "skip_reason": "fewer_than_min_pairs",
        }

    excluded = local_false_negative_mask(points_source, points_target, exclusion_px=exclusion_px)
    allowed = ~excluded
    allowed.fill_diagonal_(True)
    source = F.normalize(embedding_source.float(), dim=-1)
    target = F.normalize(embedding_target.float(), dim=-1)
    logits = source @ target.T / temperature

    def direction_loss(direction_logits: Tensor, direction_allowed: Tensor) -> tuple[Tensor | None, int]:
        valid_query = direction_allowed.sum(dim=1) > 1
        count = int(valid_query.sum().item())
        if count == 0:
            return None, 0
        masked_logits = direction_logits.masked_fill(~direction_allowed, torch.finfo(direction_logits.dtype).min)
        per_query = torch.logsumexp(masked_logits, dim=1) - direction_logits.diagonal()
        return per_query[valid_query].mean(), count

    source_loss, source_queries = direction_loss(logits, allowed)
    target_loss, target_queries = direction_loss(logits.T, allowed.T)
    if source_loss is None or target_loss is None:
        valid_queries = min(source_queries, target_queries)
        return {
            "loss": zero,
            "active": False,
            "pair_count": pair_count,
            "valid_queries": valid_queries,
            "invalid_queries": pair_count - valid_queries,
            "skip_reason": "no_valid_negative_queries",
        }
    valid_queries = min(source_queries, target_queries)
    return {
        "loss": 0.5 * (source_loss + target_loss),
        "active": True,
        "pair_count": pair_count,
        "valid_queries": valid_queries,
        "invalid_queries": pair_count - valid_queries,
        "skip_reason": None,
    }
