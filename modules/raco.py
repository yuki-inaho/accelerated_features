"""Frozen sparse XFeat with independent ranking and positional error heads.

Coordinates use integer pixel centers. Covariances are effective error matrices
in original-image pixel squared, conditional on the training correspondence rule.
This is an XFeat extension inspired by RaCo, not the RaCo architecture or weights.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.model import XFeatModel
from modules.utils import load_pretrained_weights, resolve_device
from modules.xfeat import DEFAULT_WEIGHTS


def finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each pixel; no spatial statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PixelHead(nn.Module):
    """Read selected integer pixels without constructing full-resolution outputs."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.trunk = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            ChannelNorm(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            ChannelNorm(64),
            nn.SiLU(),
        )
        self.projection = nn.Conv2d(64, 64 * channels, 1)

    def forward(self, z: Tensor, points: Tensor) -> Tensor:
        if z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != 128:
            raise ValueError("PixelHead expects one 128-channel feature map")
        if points.ndim != 2 or points.shape[-1] != 2 or points.dtype != torch.int64:
            raise ValueError("PixelHead expects int64 Nx2 pixel centers")
        finite(z, "head input")
        if (points < 0).any() or (points >= points.new_tensor([z.shape[3] * 8, z.shape[2] * 8])).any():
            raise ValueError("PixelHead coordinates outside image")
        hidden = self.trunk(z)
        x, y = points.unbind(-1)
        feature = hidden[0, :, y // 8, x // 8].T
        phase = 8 * (y % 8) + x % 8
        rows = phase[:, None] + 64 * torch.arange(self.channels, device=z.device)[None]
        weight = self.projection.weight[:, :, 0, 0][rows]
        assert self.projection.bias is not None
        result = (weight * feature[:, None]).sum(-1) + self.projection.bias[rows]
        finite(result, "head output")
        return result


class RacoHeads(nn.Module):
    def __init__(self, sigma_min: float = 0.05) -> None:
        super().__init__()
        if not math.isfinite(sigma_min) or not 0 < sigma_min < 1:
            raise ValueError("sigma_min must lie in (0, 1)")
        self.sigma_min = sigma_min
        self.ranker = PixelHead(1)
        self.covariance_head = PixelHead(3)
        nn.init.normal_(self.ranker.projection.weight, std=1e-3)
        assert self.ranker.projection.bias is not None
        nn.init.zeros_(self.ranker.projection.bias)
        nn.init.normal_(self.covariance_head.projection.weight, std=1e-3)
        with torch.no_grad():
            assert self.covariance_head.projection.bias is not None
            self.covariance_head.projection.bias.zero_()
            diagonal = math.log(math.expm1(math.sqrt(1 - sigma_min**2)))
            self.covariance_head.projection.bias[:64].fill_(diagonal)
            self.covariance_head.projection.bias[128:].fill_(diagonal)

    def rank(self, z: Tensor, points: Tensor, scores: Tensor) -> Tensor:
        finite(scores, "candidate scores")
        if scores.shape != (len(points),) or (scores < 0).any():
            raise ValueError("Candidate scores must be nonnegative and aligned")
        delta = self.ranker(z, points).squeeze(-1)
        if not len(scores):
            return delta
        logp = (scores + 1e-8).log()
        normalized = (logp - logp.mean()) / (logp.var(unbiased=False) + 1e-6).sqrt()
        return normalized.tanh() + 2 * delta.tanh()

    def covariance(self, z: Tensor, points: Tensor) -> Tensor:
        a, b, c = self.covariance_head(z, points).unbind(-1)
        zero = torch.zeros_like(a)
        lower = torch.stack((F.softplus(a), zero, b, F.softplus(c)), -1).reshape(-1, 2, 2)
        result = lower @ lower.mT + self.sigma_min**2 * torch.eye(2, device=z.device, dtype=z.dtype)
        finite(result, "covariance")
        return result


def resize_input(image: Tensor) -> tuple[Tensor, Tensor]:
    """Resize once to multiples of 32 and return original-to-network T."""
    if image.ndim != 4 or image.shape[1] not in (1, 3) or not image.is_floating_point():
        raise ValueError("Expected floating BCHW RGB/grayscale image")
    finite(image, "image")
    if image.shape[0] < 1 or min(image.shape[-2:]) < 2 or (image < 0).any() or (image > 1).any():
        raise ValueError("Image must be nonempty, at least 2x2 and in [0,1]")
    h, w = image.shape[-2:]
    nh, nw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    sx, sy = nw / w, nh / h
    transform = image.new_tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    if (nh, nw) != (h, w):
        image = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False)
    return image, transform


def restore_coordinates(points: Tensor, covariances: Tensor | None, transform: Tensor) -> tuple[Tensor, Tensor | None]:
    finite(transform, "coordinate transform")
    if transform.shape != (3, 3) or not torch.equal(transform[2], transform.new_tensor([0, 0, 1])):
        raise ValueError("Coordinate transform must be affine 3x3")
    inverse = torch.linalg.inv(transform[:2, :2])
    restored = (points - transform[:2, 2]) @ inverse.T
    cov = None if covariances is None else inverse @ covariances @ inverse.T
    return restored, cov


def sample_map(feature: Tensor, points: Tensor, height: int, width: int, mode: str = "bilinear") -> Tensor:
    grid = 2 * (points.to(feature.dtype) + 0.5) / feature.new_tensor([width, height]) - 1
    sampled = F.grid_sample(feature, grid[None, :, None], mode=mode, align_corners=False)
    return sampled[0, :, :, 0].T


class XFeatRaCo(nn.Module):
    """Separate API so original XFeat's pretrained sampling remains unchanged."""

    def __init__(
        self,
        weights: Any = DEFAULT_WEIGHTS,
        *,
        device: str | torch.device = "cpu",
        candidate_limit: int = 8192,
        detection_threshold: float = 0.05,
        border: int = 4,
        sigma_min: float = 0.05,
    ) -> None:
        super().__init__()
        if type(candidate_limit) is not int or candidate_limit < 1 or type(border) is not int or border < 0:
            raise ValueError("candidate_limit must be positive and border nonnegative integers")
        if not math.isfinite(detection_threshold) or not 0 <= detection_threshold <= 1:
            raise ValueError("Invalid detection threshold")
        self.net = XFeatModel()
        self.net.requires_grad_(False).eval()
        self.heads = RacoHeads(sigma_min)
        self.warm_start_report: dict[str, Any] = {}
        if weights is not None:
            self.warm_start_report = self.warm_start(weights)
        self.candidate_limit, self.detection_threshold, self.border = candidate_limit, detection_threshold, border
        self.trained_heads: set[str] = set()
        self.loaded_bundle = False
        self.enabled_ranking = self.enabled_covariance = True
        self.to(resolve_device(device))

    def warm_start(self, weights: Any) -> dict[str, Any]:
        """Non-strict transfer from XFeat or XFeat+LighterGlue exports.

        Missing new heads and unrelated matcher keys are recorded. Every frozen
        XFeat tensor must still be present: randomly frozen missing base weights
        would invalidate this task's initial condition.
        """
        raw = load_pretrained_weights(weights, torch.device("cpu"))
        state = {}
        for key, value in raw.items():
            if key.startswith("extractor.model.net."):
                key = "net." + key.removeprefix("extractor.model.net.")
            elif key in self.net.state_dict():
                key = "net." + key
            if key in state:
                raise ValueError("Duplicate transferred checkpoint key")
            finite(value, key)
            state[key] = value
        expected_base = {"net." + key for key in self.net.state_dict()}
        if expected_base - state.keys():
            raise ValueError("Warm start is missing frozen XFeat weights")
        incompatible = self.load_state_dict(state, strict=False)
        return {
            "strict": False,
            "loaded_keys": sorted(set(state) & set(self.state_dict())),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }

    def train(self, mode: bool = True) -> XFeatRaCo:
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def candidates(self, image: Tensor, valid_mask: Tensor | None = None) -> list[dict[str, Tensor]]:
        if image.ndim != 4 or not image.is_floating_point():
            raise ValueError("Expected floating BCHW image")
        if valid_mask is not None and valid_mask.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError("valid_mask must match original image dimensions")
        device = next(self.parameters()).device
        image = image.to(device=device, dtype=torch.float32)
        original_size = image.new_tensor([image.shape[-1], image.shape[-2]])
        image, transform = resize_input(image)
        b, _, h, w = image.shape
        if valid_mask is None:
            valid_mask = torch.ones((b, 1, h, w), device=device, dtype=torch.bool)
        else:
            if valid_mask.ndim != 4 or valid_mask.shape[:2] != (b, 1) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be B1HW bool")
            valid_mask = F.interpolate(valid_mask.float().to(device), size=(h, w), mode="nearest") > 0.5
        radius = self.border
        if radius:
            padded = F.pad(valid_mask.float(), (radius,) * 4, value=0)
            valid_mask = -F.max_pool2d(-padded, 2 * radius + 1, stride=1) > 0.5
        features, logits, reliability, detector = self.net.forward_with_features(image)
        for name, tensor in (
            ("features", features),
            ("detector", detector),
            ("logits", logits),
            ("reliability", reliability),
        ):
            finite(tensor, name)
        maps = F.pixel_shuffle(logits.softmax(1)[:, :64], 8)
        peaks = (maps == F.max_pool2d(maps, 5, stride=1, padding=2)) & (maps > self.detection_threshold) & valid_mask
        z = torch.cat((features, detector), 1)
        descriptors = F.normalize(features, dim=1)
        result = []
        for index in range(b):
            points = peaks[index, 0].nonzero()[:, [1, 0]]
            scores = maps[index, 0, points[:, 1], points[:, 0]]
            scores = scores * sample_map(reliability[index : index + 1], points, h, w).squeeze(-1)
            order = torch.argsort(scores, descending=True, stable=True)[: self.candidate_limit]
            points, scores = points[order], scores[order]
            desc = F.normalize(sample_map(descriptors[index : index + 1], points, h, w, "bicubic"), dim=-1)
            result.append(
                {
                    "keypoints": points,
                    "scores": scores,
                    "descriptors": desc,
                    "candidate_ids": torch.arange(len(points), device=device),
                    "z": z[index : index + 1],
                    "transform": transform,
                    "image_size": original_size,
                    "support": valid_mask[index : index + 1],
                }
            )
        return result

    def predict(
        self, candidates: dict[str, Tensor], *, top_k: int = 4096, ranking: bool = True, covariance: bool = True
    ) -> dict[str, Tensor]:
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        p, points, z = candidates["scores"], candidates["keypoints"], candidates["z"]
        ranks = self.heads.rank(z, points, p) if ranking else None
        order = torch.argsort(p if ranks is None else ranks, descending=True, stable=True)[:top_k]
        cov = self.heads.covariance(z, points[order]) if covariance else None
        coords, cov = restore_coordinates(points[order].to(z.dtype), cov, candidates["transform"])
        result = {
            "keypoints": coords,
            "descriptors": candidates["descriptors"][order],
            "scores": p[order],
            "keypoint_scores": p[order],
            "candidate_ids": candidates["candidate_ids"][order],
            "image_size": candidates["image_size"],
        }
        if ranks is not None:
            result["ranker_scores"] = ranks[order]
        if cov is not None:
            result["covariances"] = cov
        return result

    @torch.no_grad()
    def extract(
        self, image: Tensor, *, top_k: int = 4096, ranking: bool | None = None, covariance: bool | None = None
    ) -> list[dict[str, Tensor]]:
        rank = self.enabled_ranking if ranking is None else ranking
        cov = self.enabled_covariance if covariance is None else covariance
        if self.loaded_bundle and (
            (rank and "rank" not in self.trained_heads) or (cov and "covariance" not in self.trained_heads)
        ):
            raise ValueError("Requested untrained head; explicitly disable it")
        return [self.predict(c, top_k=top_k, ranking=rank, covariance=cov) for c in self.candidates(image)]

    def bundle(self, *, trained_heads: list[str]) -> dict[str, Any]:
        if set(trained_heads) - {"rank", "covariance"}:
            raise ValueError("Unknown trained head")
        return {
            "schema_version": 1,
            "architecture": "xfeat_raco_v1",
            "trained_heads": sorted(trained_heads),
            "config": {
                "candidate_limit": self.candidate_limit,
                "detection_threshold": self.detection_threshold,
                "border": self.border,
                "sigma_min": self.heads.sigma_min,
            },
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_bundle(
        cls, path: str | Path, *, device: str | torch.device = "cpu", ranking: bool = True, covariance: bool = True
    ) -> XFeatRaCo:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if set(bundle) != {"schema_version", "architecture", "trained_heads", "config", "state_dict"}:
            raise ValueError("Invalid RaCo bundle keys")
        if bundle["schema_version"] != 1 or bundle["architecture"] != "xfeat_raco_v1":
            raise ValueError("Unsupported RaCo bundle schema/architecture")
        trained = set(bundle["trained_heads"])
        if trained - {"rank", "covariance"}:
            raise ValueError("Invalid trained heads")
        if (ranking and "rank" not in trained) or (covariance and "covariance" not in trained):
            raise ValueError("Requested untrained head; explicitly disable it")
        model = cls(weights=None, device=device, **bundle["config"])
        for key, tensor in bundle["state_dict"].items():
            finite(tensor, key)
        model.load_state_dict(bundle["state_dict"], strict=True)
        model.trained_heads = trained
        model.loaded_bundle = True
        model.enabled_ranking, model.enabled_covariance = ranking, covariance
        return model.eval()
