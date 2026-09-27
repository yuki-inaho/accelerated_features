"""Prepare immutable evaluation anchors, run val baselines, or compare a frozen selection."""

from __future__ import annotations

import argparse
import json
from typing import Any

from modules.lighterglue import LighterGlue
from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.data import file_sha256
from xfeat_training.evaluate import EvaluationSuite, evaluate_model, prepare_evaluation
from xfeat_training.mining import repo_path
from xfeat_training.trainer import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "baseline", "compare"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--pairs", default="temp/l76_run/pairs_v1")
    parser.add_argument("--cache", default="temp/l76_run/cache_official")
    parser.add_argument("--eval-cache", default="temp/l76_run/eval_official_5pct")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--selection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.mode == "prepare":
        result = prepare_evaluation(args.pairs, args.cache, output, data_root=args.data_root, device=args.device)
        print(f"Prepared fixed anchors/GT: {result['content_hash']}")
        return
    suite = EvaluationSuite(args.eval_cache, data_root=args.data_root)
    candidates: list[dict[str, Any]]
    if args.mode == "baseline":
        if args.split != "val":
            raise ValueError("Test is evaluated only by compare after selection has been frozen")
        candidates = [
            {"name": "official_lg", "xfeat_weights": "weights/xfeat.pt", "lg_weights": "weights/xfeat-lighterglue.pt"},
            {"name": "official_mnn", "xfeat_weights": "weights/xfeat.pt", "lg_weights": None},
        ]
        selection_hash = None
    else:
        if not args.selection:
            raise ValueError("compare requires --selection")
        path = repo_path(args.selection)
        selection = json.loads(path.read_text())
        if selection["selected_on"] != "val" or selection["evaluation_hash"] != suite.manifest["content_hash"]:
            raise ValueError("Selection must be based on val with this exact evaluation cache")
        candidates = selection["candidates"]
        selection_hash = file_sha256(path)
        for candidate in candidates:
            for key in ("xfeat_weights", "lg_weights"):
                if candidate.get(key) and file_sha256(repo_path(candidate[key])) != candidate["file_hashes"][key]:
                    raise ValueError(f"Selected candidate changed: {candidate['name']} {key}")
    # Validate every selected input before creating any result directory.
    for candidate in candidates:
        if not repo_path(candidate["xfeat_weights"]).is_file():
            raise FileNotFoundError(candidate["xfeat_weights"])
        if candidate.get("lg_weights") and not repo_path(candidate["lg_weights"]).is_file():
            raise FileNotFoundError(candidate["lg_weights"])
    results = {}
    for candidate in candidates:
        extractor = XFeat(weights=repo_path(candidate["xfeat_weights"]), device=args.device, top_k=1024)
        matcher = None
        if candidate.get("lg_weights"):
            matcher = LighterGlue(
                weights=repo_path(candidate["lg_weights"]),
                device=args.device,
                flash=False,
                width_confidence=-1,
                depth_confidence=-1,
                filter_threshold=0.1,
            )
            official_pair = (
                repo_path(candidate["xfeat_weights"]) == repo_path("weights/xfeat.pt")
                and repo_path(candidate["lg_weights"]) == repo_path("weights/xfeat-lighterglue.pt")
                and state_hash(extractor.net.state_dict()) == suite.identity["official_extractor_hash"]
            )
            if not official_pair and matcher.extractor_hash != state_hash(extractor.net.state_dict()):
                raise ValueError("Selected bundle/extractor mismatch")
        result = evaluate_model(
            suite,
            extractor,
            matcher.net if matcher is not None else None,
            args.split,
            output / candidate["name"],
            fine_diagnostic=matcher is None,
        )
        results[candidate["name"]] = result
        print(f"{candidate['name']}: primary F1={result['primary_f1']:.6f}", flush=True)
        del extractor, matcher
    atomic_json(
        output / "comparison.json",
        {
            "split": args.split,
            "selection_sha256": selection_hash,
            "evaluation_hash": suite.manifest["content_hash"],
            "candidates": results,
        },
    )


if __name__ == "__main__":
    main()
