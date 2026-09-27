"""Offline pair mining, exhaustive cross-split auditing and content manifests."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from xfeat_training.data import Array, Frame, FrameKey, L76Dataset, file_sha256, json_hash
from xfeat_training.geometry import overlap, relative_motion

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_DTYPES = {
    "subset": "<U64",
    "scene": "<U64",
    "frame0": "<i8",
    "frame1": "<i8",
    "chunk0": "<i8",
    "chunk1": "<i8",
    "split": "<U8",
    "gap": "<i8",
    "overlap": "<f4",
    "rotation_deg": "<f4",
    "translation_m": "<f4",
    "difficulty": "<U32",
}


def repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def source_identity() -> dict[str, Any]:
    """Hash tracked files and new implementation files without private artifacts."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    paths = {REPO_ROOT / p for p in git("ls-files").splitlines() if (REPO_ROOT / p).is_file()}
    for folder in ("modules", "xfeat_training", "scripts", "configs", "tests"):
        paths.update(p for p in (REPO_ROOT / folder).rglob("*") if p.suffix in {".py", ".yaml"})
    paths.update(p for p in (REPO_ROOT / "third_party/amuse").glob("*") if p.is_file())
    files = {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty_diff_hash": json_hash(git("diff", "--binary")),
        "files": files,
        "source_hash": json_hash(files),
    }


def runtime_source_identity() -> dict[str, Any]:
    """Content used for resume compatibility, independent of Git and documentation.

    New runtime import roots must be added here. Python files are compared byte for
    byte; dependency files are conservative whole-file checks, not environment probes.
    """
    paths = {REPO_ROOT / name for name in ("pyproject.toml", "uv.lock", "requirements.txt")}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing runtime dependency contract: {path.name}")
    for folder in ("xfeat_training", "modules", "scripts", "third_party/amuse"):
        paths.update((REPO_ROOT / folder).rglob("*.py"))
    paths.update((REPO_ROOT / "configs").rglob("*.yaml"))
    paths.update((REPO_ROOT / "configs").rglob("*.yml"))
    for name in ("third_party/__init__.py", ".python-version"):
        path = REPO_ROOT / name
        if path.is_file():
            paths.add(path)
    files = {path.relative_to(REPO_ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}
    return {"schema_version": 1, "files": files, "content_hash": json_hash(files)}


def arrays_hash(arrays: Mapping[str, Array]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        if array.dtype.hasobject:
            raise ValueError("object arrays are forbidden")
        normalized = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(json.dumps([key, normalized.dtype.str, normalized.shape], separators=(",", ":")).encode())
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def load_pairs(path: str | Path) -> dict[str, Array]:
    with np.load(path, allow_pickle=False) as source:
        arrays = dict(source)
    if set(arrays) != set(PAIR_DTYPES):
        raise ValueError(f"{path}: invalid pair keys")
    n = len(arrays["frame0"])
    for key, dtype in PAIR_DTYPES.items():
        value = arrays[key]
        expected = np.dtype(dtype)
        if value.shape != (n,) or value.dtype.kind != expected.kind or value.dtype.itemsize != expected.itemsize:
            raise ValueError(f"{path}: invalid {key} dtype/shape")
        if expected.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{path}: nonfinite {key}")
    return arrays


def pair_keys(arrays: Mapping[str, Array], index: int) -> tuple[FrameKey, FrameKey]:
    subset, scene = str(arrays["subset"][index]), str(arrays["scene"][index])
    return FrameKey(subset, scene, int(arrays["frame0"][index])), FrameKey(subset, scene, int(arrays["frame1"][index]))


def pair_dataset(pairs_dir: str | Path, data_root: str | Path | None = None) -> tuple[L76Dataset, dict[str, Any]]:
    """Validate prepared inputs; relocate only with an explicit root or environment override."""
    directory = repo_path(pairs_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or json_hash(manifest["identity"]) != manifest["content_hash"]:
        raise ValueError(f"{directory}: invalid pair manifest schema/content hash")
    if manifest["array_hashes"] != manifest["identity"]["array_hashes"]:
        raise ValueError(f"{directory}: array hashes disagree with manifest identity")
    semantic_config = {
        k: v
        for k, v in manifest["resolved_config"].items()
        if k not in {"data_root", "output_dir", "archive_path", "hydra", "defaults"}
    }
    if semantic_config != manifest["identity"]["config"]:
        raise ValueError(f"{directory}: resolved configuration disagrees with manifest identity")
    for split, expected in manifest["array_hashes"].items():
        if arrays_hash(load_pairs(directory / f"{split}.npz")) != expected:
            raise ValueError(f"{directory}: pair hash mismatch for {split}")
    root = data_root or os.environ.get("L76_DATA_ROOT") or manifest["data_root"]
    if not repo_path(root).is_dir():
        raise ValueError("Recorded data root is missing; set L76_DATA_ROOT explicitly to relocate")
    dataset = L76Dataset(repo_path(root))
    if dataset.fingerprint() != manifest["identity"]["dataset_hash"]:
        raise ValueError(f"{directory}: dataset hash mismatch")
    return dataset, manifest


def _split_frames(data: L76Dataset, config: Mapping[str, Any]) -> dict[tuple[str, str, str], list[FrameKey]]:
    split_chunks = config["split_chunks"]
    reverse = {int(chunk): split for split, chunks in split_chunks.items() for chunk in chunks}
    if len(reverse) != sum(len(v) for v in split_chunks.values()):
        raise ValueError("split chunk membership overlaps")
    groups: dict[tuple[str, str, str], list[FrameKey]] = {}
    for (subset, scene_name), scene in data.scenes.items():
        for chunk, count in Counter(scene.chunk_ids.tolist()).items():
            if count != config["chunk_size"] or chunk not in reverse:
                raise ValueError(f"{scene.path}: chunk {chunk} size/membership mismatch")
        for fid, chunk in zip(scene.frame_ids, scene.chunk_ids, strict=True):
            if fid < config["exclude_before"].get(subset, 0):
                continue
            if any(e["subset"] == subset and e["start"] <= fid <= e["end"] for e in config["frame_exclusions"]):
                continue
            split = reverse[int(chunk)]
            if split != "guard":
                groups.setdefault((subset, scene_name, split), []).append(FrameKey(subset, scene_name, int(fid)))
        for split in ("train", "val", "test"):
            keys = groups.get((subset, scene_name, split), [])
            if len(keys) < config["min_frames"][split]:
                raise ValueError(f"{subset}/{scene_name}/{split}: insufficient frames ({len(keys)})")
            keys.sort()
    return groups


def _reference_overlap(data: L76Dataset, compute: Any) -> dict[str, Any]:
    differences = []
    checked = set()
    for (subset, name), scene in data.scenes.items():
        sequence_path, overlap_path = scene.path / "sequences.npz", scene.path / "overlap.npz"
        if not sequence_path.exists() and not overlap_path.exists():
            continue  # tiny schema fixtures may contain no source sequences
        with np.load(sequence_path, allow_pickle=False) as s, np.load(overlap_path, allow_pickle=False) as o:
            sequences, lengths = s["sequences"], s["lengths"]
            values, valid = o["all_depth"], o["pair_valid"]
            if values.shape != valid.shape or values.shape[:2] != sequences.shape or len(lengths) != len(sequences):
                raise ValueError(f"{overlap_path}: source overlap shape mismatch")
            for row, length in enumerate(lengths):
                for i, j in itertools.combinations(range(int(length)), 2):
                    if not valid[row, i, j]:
                        continue
                    a, b = (FrameKey(subset, name, int(sequences[row, k])) for k in (i, j))
                    if (a, b) in checked:
                        continue
                    checked.add((a, b))
                    difference = abs(compute(a, b) - float(values[row, i, j]))
                    if difference > 0.01:
                        raise ValueError(f"{overlap_path}: overlap reproduction differs by {difference}: {a}/{b}")
                    differences.append(difference)
    return {
        "count": len(differences),
        "max_abs_error": max(differences, default=0),
        "mean_abs_error": float(np.mean(differences)) if differences else 0,
    }


def mine_pairs(data_root: str | Path, output_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Generate all four indices only after schema, split, geometry and warmup checks."""
    output = repo_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new pair output directory: {output}")
    cfg = dict(config)
    if cfg["train_max_gap"] > cfg["candidate_max_gap"] or cfg["candidate_max_gap"] < 1:
        raise ValueError("invalid gap limits")
    if len(cfg["overlap_bins"]) != len(cfg["bin_weights"]) + 1 or any(w <= 0 for w in cfg["bin_weights"]):
        raise ValueError("invalid overlap bin weights")
    if np.any(np.diff(cfg["overlap_bins"]) <= 0):
        raise ValueError("overlap bins must increase")
    data = L76Dataset(repo_path(data_root))
    groups = _split_frames(data, cfg)
    stats: Counter[str] = Counter()

    @cache
    def frame(key: FrameKey) -> Frame:
        return data.load_frame(key, load_rgb=False)

    @cache
    def compute(a: FrameKey, b: FrameKey) -> float:
        value = overlap(frame(a), frame(b), stride=cfg["stride"], depth_tolerance=cfg["depth_tolerance"])
        stats["overlap_computations"] += 1
        if stats["overlap_computations"] % 5000 == 0:
            print(f"Depth overlap: {stats['overlap_computations']} pairs", flush=True)
        return value

    reference = _reference_overlap(data, compute)
    audit_counts: Counter[str] = Counter()
    leaks = []
    for subset, scene in sorted(data.scenes):
        for split0, split1 in itertools.combinations(("train", "val", "test"), 2):
            for a, b in itertools.product(groups[subset, scene, split0], groups[subset, scene, split1]):
                audit_counts[f"{subset}/{split0}-{split1}"] += 1
                value = compute(a, b)
                if value >= cfg["leak_overlap"]:
                    leaks.append({"frame0": a.name(), "frame1": b.name(), "overlap": value})
    if leaks:
        raise ValueError(f"cross-split leak: {len(leaks)} pairs; first examples={json.dumps(leaks[:20])}")

    rng = np.random.default_rng(cfg["seed"])
    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in ("train", "val", "test", "smoke")}
    drops: Counter[str] = Counter()
    bins = np.array(cfg["overlap_bins"])
    for (subset, scene, split), keys in sorted(groups.items()):
        candidates: dict[int, list[dict[str, Any]]] = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                gap = b.frame_id - a.frame_id
                if gap > cfg["candidate_max_gap"]:
                    break
                if (split == "train" and gap > cfg["train_max_gap"]) or (
                    split != "train" and gap not in cfg["eval_gaps"]
                ):
                    drops[f"{split}/gap"] += 1
                    continue
                rotation, translation = relative_motion(frame(a).w2c, frame(b).w2c)
                if (
                    rotation > cfg["max_rotation_deg"]
                    or not cfg["min_translation_m"] <= translation <= cfg["max_translation_m"]
                ):
                    drops[f"{split}/motion"] += 1
                    continue
                value = compute(a, b)
                if split == "train" and not cfg["train_overlap"][0] <= value <= cfg["train_overlap"][1]:
                    drops["train/overlap"] += 1
                    continue
                bin_index = min(int(np.searchsorted(bins, value, side="right")) - 1, len(bins) - 2)
                difficulty = f"overlap_{bin_index}" if bins[0] <= value <= bins[-1] else "out_of_train_range"
                row = {
                    "subset": subset,
                    "scene": scene,
                    "frame0": a.frame_id,
                    "frame1": b.frame_id,
                    "chunk0": frame(a).chunk_id,
                    "chunk1": frame(b).chunk_id,
                    "split": split,
                    "gap": gap,
                    "overlap": value,
                    "rotation_deg": rotation,
                    "translation_m": translation,
                    "difficulty": difficulty,
                }
                candidates.setdefault(gap, []).append(row)
        for _gap, items in sorted(candidates.items()):
            if split != "train" and len(items) > cfg["eval_pairs_per_bin"]:
                chosen = sorted(rng.choice(len(items), cfg["eval_pairs_per_bin"], replace=False).tolist())
                items = [items[i] for i in chosen]
            rows[split].extend(items)
    empty_bins = {}
    effective_weights = {}
    for subset in sorted({key[0] for key in groups}):
        train = [r for r in rows["train"] if r["subset"] == subset]
        warmup = [r for r in train if r["overlap"] >= 0.5]
        if len(warmup) < cfg["smoke_per_subset"]:
            raise ValueError(f"{subset}: insufficient warmup/smoke pairs ({len(warmup)})")
        selected = sorted(rng.choice(len(warmup), cfg["smoke_per_subset"], replace=False).tolist())
        rows["smoke"].extend({**warmup[i], "split": "smoke"} for i in selected)
        present = {r["difficulty"] for r in train}
        empty_bins[subset] = [i for i in range(len(bins) - 1) if f"overlap_{i}" not in present]
        weights = np.array([w if f"overlap_{i}" in present else 0.0 for i, w in enumerate(cfg["bin_weights"])])
        effective_weights[subset] = (weights / weights.sum()).tolist()
    arrays_by_split = {}
    for split, items in rows.items():
        if not items:
            raise ValueError(f"No candidate pairs for split {split}")
        items.sort(key=lambda r: (r["subset"], r["scene"], r["frame0"], r["frame1"]))
        arrays_by_split[split] = {
            key: np.array([row[key] for row in items], dtype=dtype) for key, dtype in PAIR_DTYPES.items()
        }
    files = data.file_manifest()
    source = source_identity()
    array_hashes = {split: arrays_hash(arrays) for split, arrays in arrays_by_split.items()}
    semantic_config = {
        k: v for k, v in cfg.items() if k not in {"data_root", "output_dir", "archive_path", "hydra", "defaults"}
    }
    archive_hash = file_sha256(repo_path(cfg["archive_path"])) if cfg.get("archive_path") else None
    identity = {
        "config": semantic_config,
        "dataset_hash": json_hash(files),
        "array_hashes": array_hashes,
        "archive_sha256": archive_hash,
        "source_hash": source["source_hash"],
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_config": cfg,
        "data_root": str(data.root),
        "output_dir": str(output),
        "identity": identity,
        "content_hash": json_hash(identity),
        "files": files,
        "source": source,
        "array_hashes": array_hashes,
        "frame_counts": {"/".join(k): len(v) for k, v in groups.items()},
        "pair_counts": {s: len(r) for s, r in rows.items()},
        "drop_reasons": dict(drops),
        "empty_train_bins": empty_bins,
        "effective_bin_weights": effective_weights,
        "split_audit": {"pairs_checked": dict(audit_counts), "leak_count": 0, "frame_overlap_count": 0},
        "source_overlap_check": reference,
        "computation_stats": dict(stats),
    }
    output.mkdir(parents=True, exist_ok=False)
    for split, arrays in arrays_by_split.items():
        # PAIR_DTYPES contains no object dtype or reserved savez keyword.
        np.savez_compressed(output / f"{split}.npz", **cast(dict[str, Any], arrays))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
