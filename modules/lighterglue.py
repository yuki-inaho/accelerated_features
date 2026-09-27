"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch
from kornia.feature.lightglue import LightGlue
from torch import Tensor, nn

from modules.typecheck import typechecked
from modules.utils import load_pretrained_weights, resolve_device

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "xfeat-lighterglue.pt"
DEFAULT_WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt"


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
    ) -> None:
        super().__init__()
        LightGlue.default_conf = self.default_conf_xfeat
        # None selects a feature-extractor-free LightGlue; kornia's stub only accepts str
        self.net = LightGlue(None)  # ty: ignore[invalid-argument-type]
        self.dev = resolve_device(device)

        state_dict = self._load_state_dict(weights, self.dev)

        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace("matcher.", ""): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev)

    @staticmethod
    def _load_state_dict(
        weights: str | os.PathLike[str] | Mapping[str, Tensor] | None,
        device: torch.device,
    ) -> Mapping[str, Tensor]:
        """Load pretrained weights from memory, disk, or the upstream release URL."""
        if weights is not None and not isinstance(weights, Mapping) and not Path(weights).is_file():
            weights = None  # fall back to the released weights
        if weights is None:
            return torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS_URL, map_location=device, weights_only=True)
        return load_pretrained_weights(weights, device)

    @torch.inference_mode()
    @typechecked
    def forward(self, data: Mapping[str, Tensor], min_conf: float = 0.1) -> dict[str, Any]:
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
