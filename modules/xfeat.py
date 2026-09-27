"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from numpy import ndarray as NDArray
from torch import Tensor, nn

from modules.interpolator import InterpolateSparse2d
from modules.model import XFeatModel
from modules.typecheck import (
    DenseFeatures,
    ImageInput,
    KeypointsArray,
    SparseFeatures,
    SparseFeaturesWithSize,
    typechecked,
)
from modules.utils import load_pretrained_weights, resolve_device

if TYPE_CHECKING:
    from modules.lighterglue import LighterGlue

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "xfeat.pt"


class XFeat(nn.Module):
    """
    Implements the inference module for XFeat.

    It supports inference for both sparse and semi-dense feature extraction
    and matching. Weights default to the pretrained checkpoint shipped in
    ``weights/xfeat.pt``; pass ``weights=None`` for random initialization or
    ``device`` to pin the model to a specific torch device.
    """

    @typechecked
    def __init__(
        self,
        weights: str | os.PathLike[str] | Mapping[str, Tensor] | None = DEFAULT_WEIGHTS,
        top_k: int = 4096,
        detection_threshold: float = 0.05,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.dev = resolve_device(device)
        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        if weights is not None:
            state_dict = load_pretrained_weights(weights, self.dev)
            if not isinstance(weights, Mapping):
                print(f"loading weights from: {weights}")
            self.net.load_state_dict(state_dict)

        self.interpolator = InterpolateSparse2d("bicubic")
        self._nearest = InterpolateSparse2d("nearest")
        self._bilinear = InterpolateSparse2d("bilinear")

        # LighterGlue is provided by kornia, which is imported lazily
        self.kornia_available = importlib.util.find_spec("kornia") is not None
        self.lighterglue: LighterGlue | None = None

    @torch.inference_mode()
    @typechecked
    def detectAndCompute(
        self,
        x: ImageInput,
        top_k: int | None = None,
        detection_threshold: float | None = None,
    ) -> list[SparseFeatures]:
        """
        Compute sparse keypoints & descriptors. Supports batched mode.

        input:
            x -> torch.Tensor(B, C, H, W): grayscale or rgb image
        return:
            List[Dict]:
                'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                'scores'       ->   torch.Tensor(N,): keypoint scores
                'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        if top_k is None:
            top_k = self.top_k
        if detection_threshold is None:
            detection_threshold = self.detection_threshold
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape

        M1, K1, R1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        # Compute reliability scores
        scores = (self._nearest(K1h, mkpts, _H1, _W1) * self._bilinear(R1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0
        return [
            SparseFeatures(
                keypoints=mkpts[b][valid[b]],
                scores=scores[b][valid[b]],
                descriptors=feats[b][valid[b]],
            )
            for b in range(B)
        ]

    @torch.inference_mode()
    @typechecked
    def detectAndComputeDense(self, x: ImageInput, top_k: int | None = None, multiscale: bool = True) -> DenseFeatures:
        """
        Compute dense *and coarse* descriptors. Supports batched mode.

        input:
            x -> torch.Tensor(B, C, H, W): grayscale or rgb image
            top_k -> int: keep best k features
        return: features sorted by their reliability score -- from most to least
            Dict:
                'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                'scales'       ->   torch.Tensor(top_k,): extraction scale
                'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if top_k is None:
            top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extractDense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

        return DenseFeatures(keypoints=mkpts, descriptors=feats, scales=sc)

    @torch.inference_mode()
    @typechecked
    def match_lighterglue(
        self,
        d0: SparseFeaturesWithSize,
        d1: SparseFeaturesWithSize,
        min_conf: float = 0.1,
    ) -> tuple[KeypointsArray, KeypointsArray, Int[NDArray, "N 2"]]:
        """
        Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched
        inference because of padding, but its possible to implement easily.
        input:
            d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
        output:
            mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
            idx              -> np.ndarray (N,2) the indices of the matching features
        """
        if not self.kornia_available:
            raise RuntimeError("We rely on kornia for LightGlue. Install with: pip install kornia")
        if self.lighterglue is None:
            from modules.lighterglue import LighterGlue

            self.lighterglue = LighterGlue(device=self.dev)

        data: dict[str, Tensor] = {
            "keypoints0": d0["keypoints"][None, ...],
            "keypoints1": d1["keypoints"][None, ...],
            "descriptors0": d0["descriptors"][None, ...],
            "descriptors1": d1["descriptors"][None, ...],
            "image_size0": torch.tensor(d0["image_size"]).to(self.dev)[None, ...],
            "image_size1": torch.tensor(d1["image_size"]).to(self.dev)[None, ...],
        }

        out = self.lighterglue(data, min_conf=min_conf)

        idxs = out["matches"][0]

        return d0["keypoints"][idxs[:, 0]].cpu().numpy(), d1["keypoints"][idxs[:, 1]].cpu().numpy(), idxs.cpu().numpy()

    @torch.inference_mode()
    @typechecked
    def match_xfeat(
        self,
        img1: ImageInput,
        img2: ImageInput,
        top_k: int | None = None,
        min_cossim: float = -1,
    ) -> tuple[KeypointsArray, KeypointsArray]:
        """
        Simple extractor and MNN matcher.
        For simplicity it does not support batched mode due to possibly different number of kpts.
        input:
            img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            top_k -> int: keep best k features
        returns:
            mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if top_k is None:
            top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]

        idxs0, idxs1 = self.match(out1["descriptors"], out2["descriptors"], min_cossim=min_cossim)

        return out1["keypoints"][idxs0].cpu().numpy(), out2["keypoints"][idxs1].cpu().numpy()

    @torch.inference_mode()
    @typechecked
    def match_xfeat_star(
        self,
        im_set1: ImageInput,
        im_set2: ImageInput,
        top_k: int | None = None,
    ) -> list[Float[Tensor, "... 4"]] | tuple[KeypointsArray, KeypointsArray]:
        """
        Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
        input:
            im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            top_k -> int: keep best k features
        returns:
            matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
        """
        if top_k is None:
            top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        # Compute coarse feats
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

        # Match batches of pairs
        idxs_list = self.batch_match(out1["descriptors"], out2["descriptors"])
        B = len(im_set1)

        # Refine coarse matches
        # this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(self.refine_matches(out1, out2, matches=idxs_list, batch_idx=b))

        return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())

    @typechecked
    def preprocess_tensor(self, x: ImageInput) -> tuple[Float[Tensor, "B C H2 W2"], float, float]:
        """Guarantee that image is divisible by 32 to avoid aliasing artifacts."""
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2, 0, 1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2, 0, 1)[None]
            else:
                raise RuntimeError("For numpy arrays, only (H,W) or (H,W,C) format is supported.")

        if len(x.shape) != 4:
            raise RuntimeError("Input tensor needs to be in (B,C,H,W) format")

        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        if _H == 0 or _W == 0:
            raise RuntimeError(f"Input resolution ({H}, {W}) is too small, H and W must be at least 32 pixels.")
        rh, rw = H / _H, W / _W

        x = F.interpolate(x, (_H, _W), mode="bilinear", align_corners=False)
        return x, rh, rw

    @typechecked
    def get_kpts_heatmap(
        self, kpts: Float[Tensor, "B 65 H W"], softmax_temp: float = 1.0
    ) -> Float[Tensor, "B 1 H2 W2"]:
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    @typechecked
    def NMS(self, x: Float[Tensor, "B 1 H W"], threshold: float = 0.05, kernel_size: int = 5) -> Int[Tensor, "B N 2"]:
        B = x.shape[0]
        pad = kernel_size // 2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max(len(p) for p in pos_batched)
        padded_pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        # Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            padded_pos[b, : len(pos_batched[b]), :] = pos_batched[b]

        return padded_pos

    @torch.inference_mode()
    @typechecked
    def batch_match(
        self,
        feats1: Float[Tensor, "B N 64"],
        feats2: Float[Tensor, "B M 64"],
        min_cossim: float = -1,
    ) -> list[tuple[Int[Tensor, "..."], Int[Tensor, "..."]]]:
        B = len(feats1)
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)

        idx0 = torch.arange(len(match12[0]), device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0

            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    @typechecked
    def subpix_softmax2d(self, heatmaps: Float[Tensor, "N H W"], temp: float = 3) -> Float[Tensor, "N 2"]:
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(
            torch.arange(W, device=heatmaps.device),
            torch.arange(H, device=heatmaps.device),
            indexing="xy",
        )
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = x[None, ...] * heatmaps
        coords_y = y[None, ...] * heatmaps
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
        return coords.sum(1)

    @typechecked
    def refine_matches(
        self,
        d0: DenseFeatures,
        d1: DenseFeatures,
        matches: list[tuple[Int[Tensor, "..."], Int[Tensor, "..."]]],
        batch_idx: int,
        fine_conf: float = 0.25,
    ) -> Float[Tensor, "K 4"]:
        idx0, idx1 = matches[batch_idx]
        feats1 = d0["descriptors"][batch_idx][idx0]
        feats2 = d1["descriptors"][batch_idx][idx1]
        mkpts_0 = d0["keypoints"][batch_idx][idx0]
        mkpts_1 = d1["keypoints"][batch_idx][idx1]
        sc0 = d0["scales"][batch_idx][idx0]

        # Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        mkpts_0 += offsets * (sc0[:, None])
        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        return torch.cat([mkpts_0, mkpts_1], dim=-1)

    @torch.inference_mode()
    @typechecked
    def match(
        self,
        feats1: Float[Tensor, "N 64"],
        feats2: Float[Tensor, "M 64"],
        min_cossim: float = 0.82,
    ) -> tuple[Int[Tensor, "K"], Int[Tensor, "K"]]:  # noqa: F821 -- "K" is a jaxtyping axis
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    def create_xy(self, h: int, w: int, dev: torch.device) -> Int[Tensor, "N 2"]:
        y, x = torch.meshgrid(
            torch.arange(h, device=dev),
            torch.arange(w, device=dev),
            indexing="ij",
        )
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)
        return xy

    @typechecked
    def extractDense(
        self,
        x: ImageInput,
        top_k: int = 8_000,
    ) -> tuple[Float[Tensor, "B N 2"], Float[Tensor, "B N 64"]]:
        if top_k < 1:
            top_k = 100_000_000

        x, rh1, rw1 = self.preprocess_tensor(x)

        M1, _K1, R1 = self.net(x)

        B, C, _H1, _W1 = M1.shape

        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)

        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
        R1 = R1.permute(0, 2, 3, 1).reshape(B, -1)

        _, top_k_indices = torch.topk(R1, k=min(len(R1[0]), top_k), dim=-1)

        feats = torch.gather(M1, 1, top_k_indices[..., None].expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k_indices[..., None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, -1)

        return mkpts, feats

    @typechecked
    def extract_dualscale(
        self,
        x: ImageInput,
        top_k: int,
        s1: float = 0.6,
        s2: float = 1.3,
    ) -> tuple[Float[Tensor, "B N 2"], Float[Tensor, "B N"], Float[Tensor, "B N 64"]]:
        if isinstance(x, np.ndarray):
            x = self.parse_input(x)
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode="bilinear")
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode="bilinear")

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k * 0.20))
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k * 0.80))

        mkpts = torch.cat([mkpts_1 / s1, mkpts_2 / s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1 / s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1 / s2)
        sc = torch.cat([sc1, sc2], dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats

    @typechecked
    def parse_input(self, x: ImageInput) -> Float[Tensor, "B C H W"]:
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0, 3, 1, 2) / 255

        return x
