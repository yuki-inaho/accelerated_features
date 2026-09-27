"""Cache features only for the frames in explicitly prepared pair files."""

from __future__ import annotations

import argparse

from modules.xfeat import DEFAULT_WEIGHTS, XFeat
from xfeat_training.features import FeatureCache
from xfeat_training.mining import load_pairs, pair_dataset, pair_keys, repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="explicit relocation override; otherwise use pair manifest")
    parser.add_argument("--pairs", "--pairs-dir", dest="pairs", required=True)
    parser.add_argument("--output", "--output-dir", dest="output", required=True)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    dataset, _ = pair_dataset(args.pairs, args.data_root)
    extractor = XFeat(weights=repo_path(args.weights), top_k=args.top_k, device=args.device)
    keys = set()
    for split in ("train", "val", "test", "smoke"):
        arrays = load_pairs(repo_path(args.pairs) / f"{split}.npz")
        for index in range(len(arrays["frame0"])):
            keys.update(pair_keys(arrays, index))
    cache = FeatureCache(dataset, repo_path(args.output), extractor, top_k=args.top_k)
    cache.build(keys)
    print(f"Cached {len(keys)} frames: {cache.root} (identity={cache.identity_hash})")


if __name__ == "__main__":
    main()
