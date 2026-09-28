"""Frozen adapter for the externally supplied, official CVG RaCo source/weights.

All maps and coordinates use the input image grid. No ``extract`` resize is
performed. Official sparse coordinates contain a +0.5 offset; map coordinates
are integer pixel centers. Cholesky elements are interpolated *before* LLᵀ.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from modules.utils import state_hash
from xfeat_training.data import file_sha256, json_hash

OFFICIAL_WEIGHT_SHA256 = "79f0d862bb41abedd8dda33a9b0120f4dcd619c43a90398175bad1a019192e3a"


class OfficialRacoTeacher:
    """Keep the teacher outside the student's optimizer/checkpoint module tree."""

    def __init__(self, source: str | Path, weights: str | Path, device: torch.device) -> None:
        root, weights = Path(source).expanduser().resolve(), Path(weights).expanduser().resolve()
        hashes = {name: file_sha256(root / "raco" / name) for name in ("__init__.py", "raco.py", "utils.py")}
        digest = json_hash(hashes)
        if file_sha256(weights) != OFFICIAL_WEIGHT_SHA256:
            raise ValueError("Teacher weights do not match the supplied official RaCo v1.0.0 checkpoint")
        # A private package name permits relative imports without modifying sys.path
        # or shadowing the student's modules.raco package.
        package = f"_cvg_raco_{digest[:16]}"
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package, root / "raco" / "__init__.py", submodule_search_locations=[str(root / "raco")]
            )
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load the supplied official RaCo source package")
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(package, None)
                raise
        self.upstream: Any = sys.modules[f"{package}.raco"]
        self.model = self.upstream.RaCo(weights=None, max_num_keypoints=512)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).requires_grad_(False).eval()
        self.frozen_hash = state_hash(self.model.state_dict())
        self.identity = {
            "weights_sha256": OFFICIAL_WEIGHT_SHA256,
            "source_files": hashes,
            "source_hash": digest,
            "state_hash": self.frozen_hash,
            "coordinates": "integer pixel centers; subtract 0.5 from official sparse output",
            "covariance_units": "input pixel squared",
            "resize": None,
        }

    def check_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Official teacher must remain frozen in eval mode")
        if state_hash(self.model.state_dict()) != self.frozen_hash:
            raise RuntimeError("Official teacher parameters or buffers changed")

    @torch.no_grad()
    def dense(self, image: Tensor, *, include_features: bool = False) -> dict[str, Any]:
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 32:
            raise ValueError("Teacher expects B x 3 x H x W RGB with H,W >= 32")
        if include_features and any(side % 32 for side in image.shape[-2:]):
            raise ValueError("Teacher feature maps require H and W divisible by 32")
        captured: dict[str, Tensor] = {}

        def capture(name: str):
            def hook(_module, _inputs, output):
                captured[name] = output.detach()

            return hook

        heads = [
            ("detector_logits", self.model.score_head),
            ("rank", self.model.ranker_head),
            ("cholesky_raw", self.model.covariance_estimator_head),
        ]
        if include_features:
            heads.extend((("block3", self.model.block3), ("block4", self.model.block4)))
        handles = [head.register_forward_hook(capture(name)) for name, head in heads]
        try:
            official = self.model({"image": image})
        finally:
            for handle in handles:
                handle.remove()
        padder = self.upstream.InputPadder(*image.shape[-2:], divis_by=32)
        maps = {name: padder.unpad(captured[name]) for name in ("detector_logits", "rank", "cholesky_raw")}
        if include_features:
            for name, stride in (("block3", 8), ("block4", 32)):
                feature = captured[name]
                expected = tuple(side // stride for side in image.shape[-2:])
                if feature.shape[-2:] != expected:
                    raise ValueError(f"Teacher {name} shape does not match the input grid")
                maps[name] = feature
        raw = maps.pop("cholesky_raw")
        maps["cholesky"] = torch.stack(
            (self.model.var_activation(raw[:, 0]), raw[:, 1], self.model.var_activation(raw[:, 2])), dim=1
        )
        maps["probability"] = maps["detector_logits"].flatten(1).softmax(-1).reshape_as(maps["detector_logits"])
        return {**maps, "official": official}

    @torch.no_grad()
    def sample(self, maps: dict[str, Any], points: Tensor) -> dict[str, Tensor]:
        """Sample B x N x 2 internal coordinates, preserving official interpolation."""
        height, width = maps["rank"].shape[-2:]
        values = {
            name: self.upstream._sample_at_keypoints(maps[name], points, height, width, True)
            for name in ("rank", "cholesky", "probability", "detector_logits")
        }
        values["covariance"] = self.upstream._covariance_matrix_from_cholesky_elements(values["cholesky"])
        return values

    @torch.no_grad()
    def at(self, image: Tensor, points: Tensor) -> dict[str, Tensor]:
        return self.sample(self.dense(image), points)
