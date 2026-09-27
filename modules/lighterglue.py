"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from kornia.feature.lightglue import LightGlue
from torch import Tensor, nn

from modules.model import XFeatModel
from modules.typecheck import typechecked
from modules.utils import load_pretrained_weights, resolve_device, state_hash, validate_state_dict

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "xfeat-lighterglue.pt"
DEFAULT_WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt"


def split_lighterglue_state(state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split the released bundle and translate only anchored legacy matcher names."""
    matcher, extractor = {}, {}
    for key, value in state.items():
        if key.startswith("extractor.model.net."):
            extractor[key.removeprefix("extractor.model.net.")] = value
        elif key.startswith("matcher."):
            name = key.removeprefix("matcher.")
            legacy = re.fullmatch(r"(self_attn|cross_attn)\.(\d+)\.(.+)", name)
            if legacy:
                kind, index, tail = legacy.groups()
                name = f"transformers.{index}.{kind}.{tail}"
            if name in matcher:
                raise ValueError(f"Colliding matcher weight: {key}")
            matcher[name] = value
        else:
            raise ValueError(f"Unexpected bundle key: {key}")
    return matcher, extractor


class LighterGlue(nn.Module):
    """
    Lighter version of LightGlue :)
    """

    default_conf_xfeat: ClassVar[dict[str, Any]] = {
        "name": "xfeat",  # just for interfacing
        "input_dim": 64,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 96,
        "add_scale_ori": False,
        "add_laf": False,  # for KeyNetAffNetHardNet
        "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
        "n_layers": 6,
        "num_heads": 1,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": 0.95,  # point pruning, disable with -1
        "filter_threshold": 0.1,  # match threshold
        "weights": None,
    }

    @typechecked
    def __init__(
        self,
        weights: str | os.PathLike[str] | Mapping[str, Tensor] | None = DEFAULT_WEIGHTS,
        device: str | torch.device | None = None,
        **config: Any,
    ) -> None:
        super().__init__()
        conf = {**self.default_conf_xfeat, **config}
        if conf["weights"] is not None:
            raise ValueError("Use the weights argument to select a LighterGlue bundle")
        # None selects a feature-extractor-free LightGlue; kornia's stub only accepts str
        self.net = LightGlue(None, **conf)  # ty: ignore[invalid-argument-type]
        self.dev = resolve_device(device)

        state_dict = self._load_state_dict(weights, torch.device("cpu"))
        matcher, extractor = split_lighterglue_state(state_dict)
        with torch.device("meta"):
            extractor_schema = XFeatModel().state_dict()
        validate_state_dict(extractor, extractor_schema)
        validate_state_dict(matcher, self.net.state_dict(), allow_missing=frozenset({"confidence_thresholds"}))
        matcher.setdefault("confidence_thresholds", cast(torch.Tensor, self.net.confidence_thresholds))
        self.net.load_state_dict(matcher, strict=True)
        self.extractor_state = extractor
        self.extractor_hash = state_hash(extractor)
        self.net.to(self.dev)

    @staticmethod
    def _load_state_dict(
        weights: str | os.PathLike[str] | Mapping[str, Tensor] | None,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        """Load pretrained weights from memory, disk, or the upstream release URL."""
        if weights is None:
            return torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS_URL, map_location=device, weights_only=True)
        return load_pretrained_weights(weights, device)

    @torch.inference_mode()
    @typechecked
    def forward(self, data: Mapping[str, Tensor], min_conf: float = 0.1) -> dict[str, Any]:
        if data["keypoints0"].shape[1] == 0 or data["keypoints1"].shape[1] == 0:
            batch = data["keypoints0"].shape[0]
            empty = torch.empty((0, 2), dtype=torch.long, device=data["keypoints0"].device)
            return {"matches": [empty.clone() for _ in range(batch)]}
        self.net.conf.filter_threshold = min_conf
        result = self.net(
            {
                "image0": {
                    "keypoints": data["keypoints0"],
                    "descriptors": data["descriptors0"],
                    "image_size": data["image_size0"],
                },
                "image1": {
                    "keypoints": data["keypoints1"],
                    "descriptors": data["descriptors1"],
                    "image_size": data["image_size1"],
                },
            }
        )
        return result
