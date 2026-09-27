"""Torch Hub entry point.

Usage:
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True)
"""

from __future__ import annotations

import torch
from torch import nn

from modules.xfeat import XFeat as _XFeat

dependencies = ["torch"]

DEFAULT_WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"


def XFeat(
    pretrained: bool = True,
    top_k: int = 4096,
    detection_threshold: float = 0.05,
    device: str | torch.device | None = None,
) -> nn.Module:
    """
    XFeat model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    weights = None
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(DEFAULT_WEIGHTS_URL, weights_only=True)

    return _XFeat(weights, top_k=top_k, detection_threshold=detection_threshold, device=device)
