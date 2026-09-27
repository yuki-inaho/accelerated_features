"""Immutable official anchors, count-based evaluation and inference X exports."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.data import Array, Frame, FrameKey, file_sha256, json_hash
from xfeat_training.geometry import matching_labels, project_points, sample_depth
from xfeat_training.mining import REPO_ROOT, load_pairs, pair_dataset, pair_keys, repo_path
from xfeat_training.trainer import atomic_json


def _metrics(counts: Mapping[str, Any]) -> dict[str, Any]:
    tp, p, g, a = (int(counts[key]) for key in ("TP", "P", "G", "A"))
    return {
        "TP": tp,
        "P": p,
        "G": g,
        "A": a,
        "precision": tp / p if p else 0.0,
        "recall": tp / g if g else None,
        "f1": 2 * tp / (p + g) if g else None,
        "precision_lower_bound": tp / a if a else None,
        "ignored_predictions": a - p,
    }


def score_predictions(matches: Array, gt0: Array, gt1: Array) -> dict[str, Any]:
    matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
    if len(matches) and ((matches < 0).any() or (matches[:, 0] >= len(gt0)).any() or (matches[:, 1] >= len(gt1)).any()):
        raise ValueError("Prediction indices outside GT arrays")
    if len(np.unique(matches[:, 0])) != len(matches) or len(np.unique(matches[:, 1])) != len(matches):
        raise ValueError("Predictions must be one-to-one")
    i, j = matches.T
    known = (gt0[i] != -2) & (gt1[j] != -2)
    correct = (gt0[i] == j) & (gt1[j] == i)
    result = _metrics({"TP": correct.sum(), "P": known.sum(), "G": (gt0 >= 0).sum(), "A": len(matches)})
    result.update(known_gt=int((gt0 != -2).sum() + (gt1 != -2).sum()), points=len(gt0) + len(gt1))
    result["gt_coverage"] = result["known_gt"] / result["points"] if result["points"] else 0.0
    return result


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], eligible_bins: Sequence[tuple[str, int]]) -> dict[str, Any]:
    if not eligible_bins:
        raise ValueError("No eligible primary bins")
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["subset"]), int(row["gap"])].append(row)
    bins = []
    by_key = {}
    for key, entries in sorted(groups.items()):
        counts = {name: sum(int(row[name]) for row in entries) for name in ("TP", "P", "G", "A")}
        result = _metrics(counts)
        result.update(
            subset=key[0],
            gap=key[1],
            pair_count=len(entries),
            known_gt=sum(int(row.get("known_gt", 0)) for row in entries),
            points=sum(int(row.get("points", 0)) for row in entries),
        )
        result["gt_coverage"] = result["known_gt"] / result["points"] if result["points"] else 0.0
        result["eligible"] = key in eligible_bins
        bins.append(result)
        by_key[key] = result
    if any(key not in by_key or by_key[key]["f1"] is None for key in eligible_bins):
        raise ValueError("Missing eligible bin or its fixed GT positives")
    total = _metrics({name: sum(int(row[name]) for row in rows) for name in ("TP", "P", "G", "A")})
    total.update(
        primary_f1=float(np.mean([by_key[key]["f1"] for key in eligible_bins])),
        bins=bins,
        eligible_bins=[list(key) for key in eligible_bins],
        pair_count=len(rows),
    )
    return total


def select_best(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates or any(not np.isfinite(row["primary_f1"]) for row in candidates):
        raise ValueError("Best selection requires finite evaluated candidates")
    return max(candidates, key=lambda row: (row["primary_f1"], row["TP"], -row["step"]))


@torch.no_grad()
def sample_descriptors(extractor: XFeat, image: Tensor, keypoints: Tensor) -> Tensor:
    prepared, rh, rw = extractor.preprocess_tensor(image)
    dense = F.normalize(extractor.net(prepared)[0], dim=1)
    points = keypoints.to(dense.device) / torch.tensor([rw, rh], device=dense.device)
    descriptors = extractor.interpolator(dense, points[None], H=prepared.shape[-2], W=prepared.shape[-1])[0]
    return F.normalize(descriptors, dim=-1)


def bundle_state(matcher: nn.Module, extractor: nn.Module) -> dict[str, Tensor]:
    state = {
        f"matcher.{key}": value.detach().cpu().clone()
        for key, value in matcher.state_dict().items()
        if key != "confidence_thresholds"
    }
    state.update(
        {f"extractor.model.net.{key}": value.detach().cpu().clone() for key, value in extractor.state_dict().items()}
    )
    if len(state) != 291:
        raise ValueError(f"Unexpected bundle schema: {len(state)} keys")
    return state


def _atomic_weights(path: Path, state: Mapping[str, Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(dict(state), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def export_task(task: Any, prefix: str, step: int) -> None:
    output = repo_path(task.config["run_dir"]) / "exports"
    output.mkdir(exist_ok=True)
    extractor_state = {key: value.detach().cpu().clone() for key, value in task.extractor.net.state_dict().items()}
    if len(extractor_state) != 122:
        raise ValueError("Unexpected XFeat export schema")
    path = output / f"{prefix}_xfeat.pt"
    _atomic_weights(path, extractor_state)
    files = {path.name: file_sha256(path)}
    matcher_hash = None
    if task.matcher is not None:
        path = output / f"{prefix}_lighterglue.pt"
        _atomic_weights(path, bundle_state(task.matcher.net, task.extractor.net))
        files[path.name] = file_sha256(path)
        matcher_hash = state_hash(task.matcher.net.state_dict())
    atomic_json(
        output / f"{prefix}_manifest.json",
        {
            "schema_version": 1,
            "step": step,
            "parameter_state": "X" if task.config["optimizer"]["name"] == "amuse" else "standard",
            "extractor_state_hash": state_hash(extractor_state),
            "matcher_state_hash": matcher_hash,
            "files": files,
            "top_k": task.config["top_k"],
            "matcher": task.config["matcher"],
            "preprocessing": "native RGB float32 [0,1]; XFeat channel mean",
            "batchnorm": "frozen",
            "identities": task.identities,
            **getattr(
                task,
                "export_origin",
                {
                    "checkpoint": f"checkpoints/step_{step:06d}.pt",
                    "checkpoint_model_key": "model+optimizer.eval"
                    if task.config["optimizer"]["name"] == "amuse"
                    else "model",
                    "checkpoint_parameter_state": "Y" if task.config["optimizer"]["name"] == "amuse" else "standard",
                    "checkpoint_file_sha256": None,
                    "checkpoint_status": "pending",
                },
            ),
        },
    )


def _save_npz(path: Path, **arrays: Array) -> None:
    if "allow_pickle" in arrays or any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("NPZ artifacts require non-object arrays and non-reserved names")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        # NumPy 1.x has no allow_pickle keyword; the schema check above forbids object data.
        np.savez_compressed(stream, **cast(dict[str, Any], arrays))


def evaluation_config() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "configs/evaluation/default.yaml"), resolve=True),
    )


def prepare_evaluation(
    pairs_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    data_root: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    from xfeat_training.features import FeatureCache

    output = repo_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Evaluation cache already exists: {output}")
    dataset, pair_manifest = pair_dataset(pairs_dir, data_root)
    cfg = evaluation_config()
    extractor = XFeat(weights=REPO_ROOT / "weights/xfeat.pt", device=device, top_k=cfg["top_k"])
    cache = FeatureCache(dataset, cache_dir, extractor, top_k=cfg["top_k"])
    splits = {split: load_pairs(repo_path(pairs_dir) / f"{split}.npz") for split in ("val", "test")}
    all_keys = {key for pairs in splits.values() for i in range(len(pairs["subset"])) for key in pair_keys(pairs, i)}
    features = {key: cache.get(key) for key in sorted(all_keys)}
    output.mkdir(parents=True)
    files, anchors, gt_files, eligible, all_bins = {}, {}, {}, {}, {}
    for key, feature in features.items():
        path = output / "anchors" / key.subset / key.scene / f"{key.frame_id:06d}.npz"
        _save_npz(
            path,
            **{name: feature[name].numpy() for name in ("keypoints", "descriptors", "scores")},
            image_size=np.array(feature["image_size"], dtype=np.int64),
        )
        relative = path.relative_to(output).as_posix()
        anchors[key.name()] = relative
        files[relative] = file_sha256(path)
    frame = lru_cache(maxsize=64)(dataset.load_frame)
    for split, pairs in splits.items():
        rows, gt_files[split] = [], []
        for i in range(len(pairs["subset"])):
            key0, key1 = pair_keys(pairs, i)
            labels = matching_labels(
                features[key0]["keypoints"].numpy(), features[key1]["keypoints"].numpy(), frame(key0), frame(key1)
            )
            path = output / "gt" / split / f"{i:06d}.npz"
            _save_npz(
                path,
                matches0=labels.matches0,
                matches1=labels.matches1,
                reasons0=labels.reasons0,
                reasons1=labels.reasons1,
            )
            relative = path.relative_to(output).as_posix()
            gt_files[split].append(relative)
            files[relative] = file_sha256(path)
            row = score_predictions(np.empty((0, 2), int), labels.matches0, labels.matches1)
            row.update(subset=str(pairs["subset"][i]), gap=int(pairs["gap"][i]))
            rows.append(row)
        groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row["subset"], row["gap"]].append(row)
        for subset in sorted(np.unique(pairs["subset"]).tolist()):
            for gap in pair_manifest["resolved_config"]["eval_gaps"]:
                groups.setdefault((subset, gap), [])
        bins = []
        for (subset, gap), entries in sorted(groups.items()):
            known = sum(row["known_gt"] for row in entries)
            points = sum(row["points"] for row in entries)
            positives = sum(row["G"] for row in entries)
            coverage = known / points if points else 0.0
            bins.append(
                {
                    "subset": subset,
                    "gap": gap,
                    "G": positives,
                    "known_gt": known,
                    "points": points,
                    "gt_coverage": coverage,
                    "pair_count": len(entries),
                    "eligible": gap in cfg["primary_gaps"] and coverage >= cfg["minimum_gt_coverage"] and positives > 0,
                }
            )
        eligible[split] = [[row["subset"], row["gap"]] for row in bins if row["eligible"]]
        all_bins[split] = bins
        if not eligible[split]:
            atomic_json(output / "failure.json", {"split": split, "reason": "No eligible primary bins", "bins": bins})
            raise ValueError(f"No eligible primary bins in {split}")
    identity = {
        "schema_version": 1,
        "pair_hash": pair_manifest["content_hash"],
        "top_k": cfg["top_k"],
        "official_feature_hash": cache.identity_hash,
        "official_extractor_hash": cache.identity["extractor_state_hash"],
        "config": cfg,
        "files": files,
        "anchors": anchors,
        "gt_files": gt_files,
        "eligible_bins": eligible,
        "bins": all_bins,
    }
    manifest = {
        "identity": identity,
        "content_hash": json_hash(identity),
        "pairs_dir": str(repo_path(pairs_dir)),
        "data_root": str(dataset.root),
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


class EvaluationSuite:
    def __init__(self, root: str | Path, *, data_root: str | Path | None = None) -> None:
        self.root = repo_path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.identity = self.manifest["identity"]
        if json_hash(self.identity) != self.manifest["content_hash"]:
            raise ValueError("Evaluation cache identity mismatch")
        if self.identity["config"] != evaluation_config():
            raise ValueError("Evaluation configuration changed since anchors were prepared")
        self.dataset, self.pair_manifest = pair_dataset(self.manifest["pairs_dir"], data_root)
        if self.pair_manifest["content_hash"] != self.identity["pair_hash"]:
            raise ValueError("Evaluation cache pair hash mismatch")
        self.frame = lru_cache(maxsize=64)(self.dataset.load_frame)
        self.anchor = lru_cache(maxsize=None)(self._anchor)

    def _read(self, relative: str) -> dict[str, Array]:
        path = self.root / relative
        if file_sha256(path) != self.identity["files"][relative]:
            raise ValueError(f"Evaluation artifact checksum mismatch: {relative}")
        with np.load(path, allow_pickle=False) as data:
            return dict(data)

    def _anchor(self, key: FrameKey) -> dict[str, Array]:
        return self._read(self.identity["anchors"][key.name()])

    def pairs(self, split: str) -> dict[str, Array]:
        if split not in {"val", "test"}:
            raise ValueError("Evaluation split must be val or test")
        return load_pairs(Path(self.manifest["pairs_dir"]) / f"{split}.npz")


@torch.no_grad()
def predict(extractor: XFeat, matcher: Any, feature0: Mapping[str, Any], feature1: Mapping[str, Any]) -> Array:
    if not len(feature0["keypoints"]) or not len(feature1["keypoints"]):
        return np.empty((0, 2), dtype=np.int64)
    device = extractor.dev
    if matcher is None:
        a, b = extractor.match(feature0["descriptors"].to(device), feature1["descriptors"].to(device), min_cossim=-1)
        return torch.stack((a, b), dim=-1).cpu().numpy()
    data = {}
    for name, feature in (("image0", feature0), ("image1", feature1)):
        data[name] = {key: feature[key].to(device)[None] for key in ("keypoints", "descriptors")}
        data[name]["image_size"] = torch.as_tensor(feature["image_size"], device=device)[None]
    return matcher(data)["matches"][0].cpu().numpy()


def reprojection_metrics(points0: Array, points1: Array, frame0: Frame, frame1: Frame) -> dict[str, Any]:
    p01, p10 = project_points(points0, frame0, frame1), project_points(points1, frame1, frame0)
    error = np.maximum(np.linalg.norm(p01.xy - points1, axis=1), np.linalg.norm(p10.xy - points0, axis=1))
    usable = p01.valid & p10.valid & (p01.z > 0) & (p10.z > 0) & np.isfinite(error)
    visible = usable.copy()
    interior = usable.copy()
    for points, projected, source, target in ((points0, p01, frame0, frame1), (points1, p10, frame1, frame0)):
        depth, valid = sample_depth(target.depth, projected.xy)
        visible &= valid & (np.abs(projected.z - depth) <= 0.03 * depth)
        _, a = sample_depth(source.depth, points, interior=True)
        _, b = sample_depth(target.depth, projected.xy, interior=True)
        interior &= a & b
    interior &= visible
    result = {}
    for name, mask in (("all", np.ones(len(error), dtype=bool)), ("visible", visible), ("interior", interior)):
        count = int(mask.sum())
        result[f"{name}_matches"] = count
        for threshold in (1, 3, 5):
            correct = int((mask & usable & (error <= threshold)).sum())
            result[f"{name}_correct_{threshold}px"] = correct
            result[f"{name}_precision_{threshold}px"] = correct / count if count else None
    return result


def _group_diagnostics(rows: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in dimensions)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        entry = dict(zip(dimensions, key, strict=True))
        entry.update(_metrics({name: sum(int(row[name]) for row in values) for name in ("TP", "P", "G", "A")}))
        entry.update(
            pair_count=len(values),
            known_gt=sum(row["known_gt"] for row in values),
            points=sum(row["points"] for row in values),
        )
        entry["gt_coverage"] = entry["known_gt"] / entry["points"] if entry["points"] else 0.0
        for scope in ("all", "visible", "interior"):
            count = sum(row[f"{scope}_matches"] for row in values)
            entry[f"{scope}_matches"] = count
            for threshold in (1, 3, 5):
                correct = sum(row[f"{scope}_correct_{threshold}px"] for row in values)
                entry[f"{scope}_correct_{threshold}px"] = correct
                entry[f"{scope}_precision_{threshold}px"] = correct / count if count else None
        result.append(entry)
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def synthetic_fine_diagnostic(suite: EvaluationSuite, extractor: XFeat, split: str) -> dict[str, Any]:
    from modules.typecheck import DenseFeatures
    from xfeat_training.augment import sample_homography, transform_points, warp_image
    from xfeat_training.objectives import synthetic_correspondences

    generator = torch.Generator().manual_seed(suite.identity["config"]["synthetic_seed"])
    pairs = suite.pairs(split)
    keys = sorted({pair_keys(pairs, i)[0] for i in range(len(pairs["subset"]))})
    error_sum, count, skipped, transforms = 0.0, 0, 0, {}
    for key in keys:
        frame = suite.frame(key)
        height, width = frame.depth.shape
        homography = sample_homography(height, width, generator)
        transforms[key.name()] = homography.tolist()
        image0 = frame.rgb[None].to(extractor.dev)
        image1, _ = warp_image(image0, homography)
        correspondence = synthetic_correspondences(homography, height, width, generator=generator)
        maps = extractor.net(image0)[0], extractor.net(image1)[0]
        for corr, map0, map1, transform in (
            (correspondence, maps[0], maps[1], homography),
            (correspondence.reverse, maps[1], maps[0], torch.linalg.inv(homography)),
        ):
            if corr is None or len(corr.source) < 2:
                skipped += 1
                continue
            a, b = (corr.source.to(extractor.dev) / 8).long(), (corr.target.to(extractor.dev) / 8).long()
            desc0 = F.normalize(map0[0, :, a[:, 1], a[:, 0]].T, dim=-1)
            desc1 = F.normalize(map1[0, :, b[:, 1], b[:, 0]].T, dim=-1)
            n = len(a)
            feature0 = DenseFeatures(
                keypoints=corr.source.to(extractor.dev)[None].clone(),
                descriptors=desc0[None],
                scales=torch.ones(1, n, device=extractor.dev),
            )
            feature1 = DenseFeatures(
                keypoints=corr.target.to(extractor.dev)[None].clone(),
                descriptors=desc1[None],
                scales=torch.ones(1, n, device=extractor.dev),
            )
            indices = torch.arange(n, device=extractor.dev)
            refined = extractor.refine_matches(feature0, feature1, [(indices, indices)], 0, fine_conf=-1)
            expected = transform_points(corr.target, torch.linalg.inv(transform))
            error_sum += float((refined[:, :2].cpu() - expected).norm(dim=-1).sum())
            count += n
    return {
        "mean_offset_error_px": error_sum / count if count else None,
        "correspondences": count,
        "frames": len(keys),
        "skipped_directions": skipped,
        "seed": suite.identity["config"]["synthetic_seed"],
        "homographies_hash": json_hash(transforms),
        "decoder": "existing refine_matches, fine_conf=-1",
    }


@torch.no_grad()
def evaluate_model(
    suite: EvaluationSuite,
    extractor: XFeat,
    matcher: Any,
    split: str,
    output_dir: str | Path,
    *,
    fine_diagnostic: bool = False,
) -> dict[str, Any]:
    output = repo_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    extractor.net.eval()
    if matcher is not None:
        matcher.eval()
        conf = matcher.conf
        if conf.width_confidence != -1 or conf.depth_confidence != -1 or conf.flash or conf.filter_threshold != 0.1:
            raise ValueError("Matcher evaluation configuration differs from fixed protocol")
    pairs = suite.pairs(split)
    eligible = [tuple(value) for value in suite.identity["eligible_bins"][split]]
    extractor_hash = state_hash(extractor.net.state_dict())
    matcher_hash = state_hash(matcher.state_dict()) if matcher is not None else None
    fixed_features, live_features = {}, {}
    rows, live_rows, predictions = [], [], {}
    for i in range(len(pairs["subset"])):
        key0, key1 = pair_keys(pairs, i)
        for key in (key0, key1):
            if key in fixed_features:
                continue
            anchor = suite.anchor(key)
            frame = suite.frame(key)
            points = torch.from_numpy(anchor["keypoints"])
            if extractor_hash == suite.identity["official_extractor_hash"]:
                descriptors = torch.from_numpy(anchor["descriptors"])
            else:
                descriptors = sample_descriptors(extractor, frame.rgb[None], points).cpu()
            fixed_features[key] = {"keypoints": points, "descriptors": descriptors, "image_size": anchor["image_size"]}
            live = extractor.detectAndCompute(frame.rgb[None], top_k=suite.identity["top_k"])[0]
            live_features[key] = {
                name: value.detach().cpu() for name, value in cast(Mapping[str, Tensor], live).items()
            }
            live_features[key]["image_size"] = anchor["image_size"]
        frame0, frame1 = suite.frame(key0), suite.frame(key1)
        metadata = {
            "pair_index": i,
            "subset": str(pairs["subset"][i]),
            "scene": str(pairs["scene"][i]),
            "frame0": key0.frame_id,
            "frame1": key1.frame_id,
            "gap": int(pairs["gap"][i]),
            "overlap": float(pairs["overlap"][i]),
            "difficulty": str(pairs["difficulty"][i]),
        }
        gt = suite._read(suite.identity["gt_files"][split][i])
        for kind, feature_map, destination in (("fixed", fixed_features, rows), ("live", live_features, live_rows)):
            f0, f1 = feature_map[key0], feature_map[key1]
            points0, points1 = f0["keypoints"].numpy(), f1["keypoints"].numpy()
            if kind == "live":
                labels = matching_labels(points0, points1, frame0, frame1)
                gt0, gt1 = labels.matches0, labels.matches1
            else:
                gt0, gt1 = gt["matches0"], gt["matches1"]
            matches = predict(extractor, matcher, f0, f1)
            predictions[f"{i:06d}_{kind}"] = matches
            row = {**metadata, **score_predictions(matches, gt0, gt1), "points0": len(points0), "points1": len(points1)}
            row.update(reprojection_metrics(points0[matches[:, 0]], points1[matches[:, 1]], frame0, frame1))
            destination.append(row)
        if (i + 1) % 50 == 0:
            print(f"Evaluation {split}: {i + 1}/{len(pairs['subset'])} pairs", flush=True)
    summary = aggregate_metrics(rows, eligible)
    summary.update(
        split=split,
        method="lighterglue" if matcher is not None else "mnn",
        pair_hash=suite.identity["pair_hash"],
        evaluation_hash=suite.manifest["content_hash"],
        extractor_state_hash=extractor_hash,
        matcher_state_hash=matcher_hash,
        config=suite.identity["config"],
        fixed_gt_bins=suite.identity["bins"][split],
        missing_bins=[row for row in suite.identity["bins"][split] if row["pair_count"] == 0],
        live={
            "by_subset_gap": _group_diagnostics(live_rows, ("subset", "gap")),
            "by_subset_overlap": _group_diagnostics(live_rows, ("subset", "difficulty")),
        },
        fixed_reprojection={
            "by_subset_gap": _group_diagnostics(rows, ("subset", "gap")),
            "by_subset_overlap": _group_diagnostics(rows, ("subset", "difficulty")),
        },
    )
    if fine_diagnostic:
        summary["synthetic_fine"] = synthetic_fine_diagnostic(suite, extractor, split)
    if state_hash(extractor.net.state_dict()) != extractor_hash or (
        matcher is not None and state_hash(matcher.state_dict()) != matcher_hash
    ):
        raise RuntimeError("Evaluation changed model parameters or buffers")
    output.mkdir(parents=True)
    atomic_json(output / "metrics.json", summary)
    atomic_json(output / "pairs_fixed.json", rows)
    atomic_json(output / "pairs_live.json", live_rows)
    _write_csv(output / "pairs_fixed.csv", rows)
    _write_csv(output / "pairs_live.csv", live_rows)
    _write_csv(output / "bins_fixed.csv", summary["bins"])
    for kind in ("live", "fixed_reprojection"):
        for grouping, values in summary[kind].items():
            _write_csv(output / f"{kind}_{grouping}.csv", values)
    _save_npz(output / "predictions.npz", **predictions)
    return summary


def evaluate_task(task: Any, step: int) -> dict[str, Any]:
    if not hasattr(task, "evaluation_suite"):
        task.evaluation_suite = EvaluationSuite(task.config["eval_cache_dir"], data_root=task.config.get("data_root"))
    return evaluate_model(
        task.evaluation_suite,
        task.extractor,
        task.matcher.net if task.matcher is not None else None,
        "val",
        repo_path(task.config["run_dir"]) / "validation" / f"step_{step:06d}",
        fine_diagnostic=task.config["task"] == "xfeat",
    )
