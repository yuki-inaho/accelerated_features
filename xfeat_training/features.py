"""Frozen XFeat feature caches identified by weights, data and preprocessing."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modules.typecheck import SparseFeaturesWithSize
from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.data import FrameKey, L76Dataset, file_sha256, json_hash
from xfeat_training.mining import repo_path


class FeatureCache:
    def __init__(self, dataset: L76Dataset, root: str | Path, extractor: XFeat, *, top_k: int = 1024) -> None:
        if top_k < 1:
            raise ValueError("cache top_k must be positive")
        self.dataset, self.root, self.extractor, self.top_k = dataset, repo_path(root), extractor, top_k
        self.identity = {
            "schema_version": 1,
            "extractor_state_hash": state_hash(extractor.net.state_dict()),
            "dataset_hash": dataset.fingerprint(),
            "top_k": top_k,
            "preprocessing": {
                "channels": "RGB",
                "range": [0, 1],
                "dtype": "float32",
                "size": "native",
                "normalization": "XFeat channel mean",
                "detection_threshold": extractor.detection_threshold,
            },
        }
        self.identity_hash = json_hash(self.identity)
        self.frames: dict[str, dict[str, Any]] = {}
        if self.root.exists():
            path = self.root / "manifest.json"
            if not path.is_file():
                raise ValueError(f"{self.root}: incomplete cache identity manifest")
            manifest = json.loads(path.read_text())
            if manifest.get("identity") != self.identity or manifest.get("identity_hash") != self.identity_hash:
                raise ValueError(f"{self.root}: feature cache identity mismatch")
            self.frames = manifest["frames"]

    def _path(self, key: FrameKey) -> Path:
        return self.root / key.subset / key.scene / f"frame_{key.frame_id:06d}.npz"

    def _save_manifest(self) -> None:
        manifest = {"identity": self.identity, "identity_hash": self.identity_hash, "frames": self.frames}
        temporary = self.root / "manifest.json.tmp"
        with temporary.open("w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.root / "manifest.json")

    def build(self, keys: Iterable[FrameKey]) -> None:
        if state_hash(self.extractor.net.state_dict()) != self.identity["extractor_state_hash"]:
            raise ValueError("extractor changed after cache identity was constructed")
        self.root.mkdir(parents=True, exist_ok=True)
        self._save_manifest()
        self.extractor.net.eval()
        for index, key in enumerate(sorted(set(keys)), start=1):
            if key.name() in self.frames:
                self.get(key)  # verify existing content before reusing it
                continue
            frame = self.dataset.load_frame(key)
            with torch.no_grad():
                features = self.extractor.detectAndCompute(frame.rgb[None], top_k=self.top_k)[0]
            arrays = {
                name: features[name].detach().cpu().numpy().copy() for name in ("keypoints", "descriptors", "scores")
            }
            arrays["image_size"] = np.array([frame.depth.shape[1], frame.depth.shape[0]], dtype=np.int64)
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self.frames[key.name()] = {"file_sha256": file_sha256(path), "points": len(arrays["keypoints"])}
            self._save_manifest()
            if index % 100 == 0:
                print(f"Cached features: {index} frames", flush=True)
        if state_hash(self.extractor.net.state_dict()) != self.identity["extractor_state_hash"]:
            raise RuntimeError("feature extraction changed model parameters or BN buffers")

    def get(self, key: FrameKey) -> SparseFeaturesWithSize:
        path = self._path(key)
        if key.name() not in self.frames or not path.is_file():
            raise FileNotFoundError(f"Missing cached features: {path}")
        if file_sha256(path) != self.frames[key.name()]["file_sha256"]:
            raise ValueError(f"{path}: feature cache checksum mismatch")
        with np.load(path, allow_pickle=False) as source:
            arrays = dict(source)
        if set(arrays) != {"keypoints", "descriptors", "scores", "image_size"}:
            raise ValueError(f"{path}: invalid feature keys")
        n = len(arrays["keypoints"])
        for name, shape in (("keypoints", (n, 2)), ("descriptors", (n, 64)), ("scores", (n,))):
            if arrays[name].dtype != np.float32 or arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{path}: invalid {name} shape/dtype/values")
        if n > self.top_k or arrays["image_size"].shape != (2,):
            raise ValueError(f"{path}: invalid feature count/image size")
        width, height = map(int, arrays["image_size"])
        # Explicitly leave inference mode, even when a caller evaluates under it.
        with torch.inference_mode(False):
            return SparseFeaturesWithSize(
                keypoints=torch.from_numpy(arrays["keypoints"].copy()),
                descriptors=torch.from_numpy(arrays["descriptors"].copy()),
                scores=torch.from_numpy(arrays["scores"].copy()),
                image_size=(width, height),
            )
