"""End-to-end tests for pretrained XFeat inference.

The tests run the full pipeline on the image pair bundled in ``assets/``:
checkpoint loading, sparse and dense feature extraction, sparse/semi-dense/
LightGlue matching, and RANSAC homography verification.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from jaxtyping import TypeCheckError
from numpy import ndarray as NDArray

from inference import MatchingResult, Method, load_image, run_inference, to_tensor
from modules.xfeat import XFeat


def test_pretrained_checkpoint_is_loaded_exactly(xfeat: XFeat, weights_file: Path) -> None:
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    state = {key: value.cpu() for key, value in xfeat.net.state_dict().items()}
    assert set(state) == set(checkpoint)
    for key, value in checkpoint.items():
        assert torch.equal(state[key], value), f"parameter mismatch: {key}"


def test_detect_and_compute_returns_valid_features(
    xfeat: XFeat,
    ref_image: NDArray,
    device: torch.device,
) -> None:
    features = xfeat.detectAndCompute(to_tensor(ref_image), top_k=1024)[0]
    keypoints, scores, descriptors = features["keypoints"], features["scores"], features["descriptors"]

    num_keypoints = keypoints.shape[0]
    assert 0 < num_keypoints <= 1024
    assert keypoints.shape == (num_keypoints, 2)
    assert scores.shape == (num_keypoints,)
    assert descriptors.shape == (num_keypoints, 64)
    assert keypoints.device.type == device.type

    assert torch.all(scores[1:] <= scores[:-1]), "features must be sorted by decreasing score"
    assert torch.allclose(descriptors.norm(dim=-1), torch.ones_like(scores), atol=1e-4)

    height, width = ref_image.shape[:2]
    assert torch.all(keypoints >= 0)
    assert torch.all(keypoints[:, 0] <= width)
    assert torch.all(keypoints[:, 1] <= height)


def test_dense_features_cover_both_scales(xfeat: XFeat, ref_image: NDArray) -> None:
    features = xfeat.detectAndComputeDense(to_tensor(ref_image), top_k=1024)
    keypoints, descriptors, scales = features["keypoints"], features["descriptors"], features["scales"]

    assert keypoints.shape[0] == 1
    num_keypoints = keypoints.shape[1]
    assert 1000 <= num_keypoints <= 1024  # 20% + 80% of top_k across the two scales
    assert keypoints.shape == (1, num_keypoints, 2)
    assert descriptors.shape == (1, num_keypoints, 64)
    assert scales.shape == (1, num_keypoints)
    assert float(scales.min()) < 1.0 < float(scales.max())


def test_detection_is_deterministic(xfeat: XFeat, ref_image: NDArray) -> None:
    first = xfeat.detectAndCompute(to_tensor(ref_image), top_k=512)[0]
    second = xfeat.detectAndCompute(to_tensor(ref_image), top_k=512)[0]

    assert torch.equal(first["keypoints"], second["keypoints"])
    assert torch.equal(first["descriptors"], second["descriptors"])
    assert torch.equal(first["scores"], second["scores"])


def test_batched_inference_matches_single_image_inference(
    xfeat: XFeat,
    ref_image: NDArray,
    tgt_image: NDArray,
) -> None:
    batch = torch.cat([to_tensor(ref_image), to_tensor(tgt_image)])
    batched = xfeat.detectAndCompute(batch, top_k=512)
    single_ref = xfeat.detectAndCompute(to_tensor(ref_image), top_k=512)[0]
    single_tgt = xfeat.detectAndCompute(to_tensor(tgt_image), top_k=512)[0]

    assert torch.equal(batched[0]["keypoints"], single_ref["keypoints"])
    assert torch.equal(batched[1]["keypoints"], single_tgt["keypoints"])
    assert torch.allclose(batched[0]["descriptors"], single_ref["descriptors"], atol=1e-4)
    assert torch.allclose(batched[1]["descriptors"], single_tgt["descriptors"], atol=1e-4)


@pytest.mark.parametrize(
    ("method", "min_matches", "min_inliers"),
    [
        ("xfeat", 400, 80),
        ("xfeat-star", 300, 200),
        ("lighterglue", 300, 300),
    ],
)
def test_matching_methods_recover_the_homography(
    xfeat: XFeat,
    ref_image: NDArray,
    tgt_image: NDArray,
    method: Method,
    min_matches: int,
    min_inliers: int,
) -> None:
    result = run_inference(xfeat, method, ref_image, tgt_image, top_k=4096)

    assert isinstance(result, MatchingResult)
    assert result.matches >= min_matches
    assert result.inliers >= min_inliers
    assert result.homography is not None

    points0 = result.points0[result.inlier_mask]
    points1 = result.points1[result.inlier_mask]
    projected = cv2.perspectiveTransform(points0.reshape(-1, 1, 2), result.homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - points1, axis=1)
    assert float(np.median(errors)) < 2.0
    assert float(errors.max()) < 3.5


def test_runtime_type_checking_rejects_invalid_inputs(xfeat: XFeat) -> None:
    with pytest.raises(TypeCheckError):
        xfeat.detectAndCompute(torch.randn(3, 480, 640))  # missing batch dimension
    with pytest.raises(TypeCheckError):
        xfeat.detectAndCompute(np.zeros((64, 64, 3), dtype=np.int32))  # unsupported dtype
    with pytest.raises(TypeCheckError):
        # top_k must be an int; the deliberate mismatch is what the runtime checker rejects
        xfeat.detectAndCompute(torch.randn(1, 3, 480, 640), top_k=1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(RuntimeError, match="too small"):
        xfeat.detectAndCompute(torch.randn(1, 3, 16, 16))  # below the 32 pixel stride


def test_torch_hub_entry_point_exposes_xfeat(repo_root: Path) -> None:
    model = torch.hub.load(str(repo_root), "XFeat", pretrained=False, source="local", trust_repo=True)
    features = model.detectAndCompute(torch.randn(1, 3, 480, 640), top_k=256)[0]
    assert features["descriptors"].shape == (features["keypoints"].shape[0], 64)


def test_missing_weights_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        XFeat(weights=tmp_path / "missing.pt", device="cpu")


def test_load_image_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_image_caps_the_max_size(repo_root: Path) -> None:
    scaled = load_image(repo_root / "assets" / "ref.png", max_size=200)
    assert max(scaled.shape[:2]) == 200


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_and_cpu_predictions_agree(ref_image: NDArray, weights_file: Path) -> None:
    image = to_tensor(ref_image)
    cpu_features = XFeat(weights=weights_file, device="cpu").detectAndCompute(image, top_k=256)[0]
    gpu_features = XFeat(weights=weights_file, device="cuda").detectAndCompute(image.cuda(), top_k=256)[0]

    assert torch.allclose(cpu_features["keypoints"], gpu_features["keypoints"].cpu(), atol=1e-3)
    cosine = (cpu_features["descriptors"] * gpu_features["descriptors"].cpu()).sum(dim=-1)
    assert bool((cosine > 0.999).all())
