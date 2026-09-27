"""Extract sparse XFeat descriptors, ranks, and positional error matrices to NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from modules.raco import XFeatRaCo


def extract_file(
    weights: str | Path,
    image: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    top_k: int = 1024,
    ranking: bool = True,
    covariance: bool = True,
) -> None:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError(f"Cannot decode image: {image}")
    tensor = torch.from_numpy(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
    model = XFeatRaCo.from_bundle(weights, device=device, ranking=ranking, covariance=covariance)
    features = model.extract(tensor, top_k=top_k)[0]
    arrays = {name: value.detach().cpu().numpy() for name, value in features.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {len(arrays['keypoints'])} keypoints to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--no-ranking", action="store_true")
    parser.add_argument("--no-covariance", action="store_true")
    args = parser.parse_args()
    extract_file(
        args.weights,
        args.image,
        args.output,
        device=args.device,
        top_k=args.top_k,
        ranking=not args.no_ranking,
        covariance=not args.no_covariance,
    )


if __name__ == "__main__":
    main()
