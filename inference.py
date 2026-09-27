"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

End-to-end inference with the pretrained XFeat weights.

Example:
    uv run inference.py assets/ref.png assets/tgt.png --output matches.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy import ndarray as NDArray
from torch import Tensor

from modules.typecheck import InlierMask, KeypointsArray, SparseFeaturesWithSize, typechecked
from modules.xfeat import DEFAULT_WEIGHTS, XFeat

Method = Literal["xfeat", "xfeat-star", "lighterglue"]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class MatchingResult:
    """Matches between two images plus their geometric verification."""

    method: Method
    points0: KeypointsArray
    points1: KeypointsArray
    homography: Float[NDArray, "3 3"] | None
    inlier_mask: InlierMask

    @property
    def matches(self) -> int:
        """Number of mutual matches between the two images."""
        return int(self.points0.shape[0])

    @property
    def inliers(self) -> int:
        """Number of matches consistent with the estimated homography."""
        return int(self.inlier_mask.sum())


@typechecked
def load_image(path: Path, max_size: int | None = None) -> UInt8[NDArray, "H W 3"]:
    """Read an image as BGR uint8, optionally capping its largest side."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: '{path}'")
    if max_size is not None:
        height, width = image.shape[:2]
        scale = max_size / max(height, width)
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def to_tensor(image: UInt8[NDArray, "H W 3"]) -> Float[Tensor, "1 3 H W"]:
    """Convert a BGR uint8 image to a normalized float tensor of shape (1, 3, H, W)."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None]
    return tensor.float() / 255.0


@typechecked
def match_pair(
    xfeat: XFeat,
    method: Method,
    image0: UInt8[NDArray, "H0 W0 3"],
    image1: UInt8[NDArray, "H1 W1 3"],
    top_k: int = 4096,
) -> tuple[KeypointsArray, KeypointsArray]:
    """Run XFeat detection, description and matching for a single image pair."""
    if method == "xfeat":
        return xfeat.match_xfeat(image0, image1, top_k=top_k)

    if method == "xfeat-star":
        matches = xfeat.match_xfeat_star(image0, image1, top_k=top_k)
        if isinstance(matches, list):
            raise ValueError("xfeat-star expects exactly one image pair")
        return matches

    features0 = xfeat.detectAndCompute(to_tensor(image0), top_k=top_k)[0]
    features1 = xfeat.detectAndCompute(to_tensor(image1), top_k=top_k)[0]
    height0, width0 = image0.shape[:2]
    height1, width1 = image1.shape[:2]
    with_size0 = SparseFeaturesWithSize(
        keypoints=features0["keypoints"],
        scores=features0["scores"],
        descriptors=features0["descriptors"],
        image_size=(width0, height0),
    )
    with_size1 = SparseFeaturesWithSize(
        keypoints=features1["keypoints"],
        scores=features1["scores"],
        descriptors=features1["descriptors"],
        image_size=(width1, height1),
    )
    points0, points1, _ = xfeat.match_lighterglue(with_size0, with_size1)
    return points0, points1


@typechecked
def estimate_homography(
    points0: KeypointsArray,
    points1: KeypointsArray,
    ransac_threshold: float = 3.0,
) -> tuple[Float[NDArray, "3 3"] | None, InlierMask]:
    """Estimate a homography between two point sets with RANSAC."""
    empty_mask = np.zeros(points0.shape[0], dtype=bool)
    if points0.shape[0] < 4:
        return None, empty_mask

    homography, mask = cv2.findHomography(
        points0.reshape(-1, 1, 2),
        points1.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_threshold,
    )
    if homography is None or mask is None:
        return None, empty_mask
    return homography, mask.ravel().astype(bool)


@typechecked
def run_inference(
    xfeat: XFeat,
    method: Method,
    image0: UInt8[NDArray, "H0 W0 3"],
    image1: UInt8[NDArray, "H1 W1 3"],
    top_k: int = 4096,
    ransac_threshold: float = 3.0,
) -> MatchingResult:
    """Match an image pair end to end and verify the matches geometrically."""
    points0, points1 = match_pair(xfeat, method, image0, image1, top_k=top_k)
    homography, inlier_mask = estimate_homography(points0, points1, ransac_threshold)
    return MatchingResult(
        method=method,
        points0=points0,
        points1=points1,
        homography=homography,
        inlier_mask=inlier_mask,
    )


@typechecked
def draw_matches(
    image0: UInt8[NDArray, "H0 W0 3"],
    image1: UInt8[NDArray, "H1 W1 3"],
    points0: KeypointsArray,
    points1: KeypointsArray,
    inlier_mask: InlierMask,
) -> UInt8[NDArray, "H W 3"]:
    """Draw the geometrically verified matches between two images."""
    keypoints0 = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points0]
    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points1]
    matches = [cv2.DMatch(int(index), int(index), 0.0) for index in np.flatnonzero(inlier_mask)]
    # OpenCV allocates the canvas when outImg is None, which its stubs do not model
    return cv2.drawMatches(  # ty: ignore[no-matching-overload]
        image0,
        keypoints0,
        image1,
        keypoints1,
        matches,
        None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the inference entry point."""
    parser = argparse.ArgumentParser(description="Match two images with pretrained XFeat weights.")
    parser.add_argument("image1", nargs="?", type=Path, default=ASSETS_DIR / "ref.png", help="first image path")
    parser.add_argument("image2", nargs="?", type=Path, default=ASSETS_DIR / "tgt.png", help="second image path")
    parser.add_argument(
        "--method",
        choices=("xfeat", "xfeat-star", "lighterglue"),
        default="xfeat",
        help="matcher backend (xfeat-star is semi-dense, lighterglue needs kornia)",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="path to the pretrained checkpoint")
    parser.add_argument("--device", default=None, help="torch device such as cuda, cpu or mps (default: auto)")
    parser.add_argument("--top-k", type=int, default=4096, dest="top_k", help="maximum number of keypoints")
    parser.add_argument("--max-size", type=int, default=None, dest="max_size", help="cap the largest image side")
    parser.add_argument("--ransac-thr", type=float, default=3.0, dest="ransac_threshold", help="RANSAC threshold")
    parser.add_argument("--output", type=Path, default=None, help="optional path for the match visualization")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run pretrained XFeat inference on an image pair from the command line."""
    args = parse_args(argv)
    xfeat = XFeat(weights=args.weights, top_k=args.top_k, device=args.device)
    image0 = load_image(args.image1, max_size=args.max_size)
    image1 = load_image(args.image2, max_size=args.max_size)

    result = run_inference(
        xfeat,
        args.method,
        image0,
        image1,
        top_k=args.top_k,
        ransac_threshold=args.ransac_threshold,
    )

    print(f"device: {xfeat.dev} | weights: {args.weights}")
    print(f"image1: {args.image1} {image0.shape[1]}x{image0.shape[0]}")
    print(f"image2: {args.image2} {image1.shape[1]}x{image1.shape[0]}")
    print(f"method: {result.method} | matches={result.matches} | inliers={result.inliers}")
    if result.homography is not None:
        print("homography:\n" + np.array2string(result.homography, precision=3, suppress_small=True))

    if args.output is not None:
        canvas = draw_matches(image0, image1, result.points0, result.points1, result.inlier_mask)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), canvas)
        print(f"saved visualization: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
