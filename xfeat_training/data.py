"""Validated L76 RGB-D reader. Depth is in metres, poses are OpenCV w2c."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

Array = NDArray[Any]
SUBSETS = ("colmap_rgbd_half_s0", "colmap_rgbd_half_s1")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, order=True)
class FrameKey:
    subset: str
    scene: str
    frame_id: int

    def name(self) -> str:
        return f"{self.subset}/{self.scene}/{self.frame_id:06d}"


@dataclass
class Frame:
    key: FrameKey
    rgb: Tensor
    depth: Array
    intrinsics: Array
    w2c: Array
    chunk_id: int


@dataclass
class Scene:
    path: Path
    metadata: dict[str, Any]
    frame_ids: Array
    intrinsics: Array
    extrinsics_w2c: Array
    chunk_ids: Array
    index: dict[int, int]


class L76Dataset:
    """Read IDs via camera metadata; never infer pose or silently discard frames."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.scenes: dict[tuple[str, str], Scene] = {}
        self.frame_keys: list[FrameKey] = []
        for subset in SUBSETS:
            metadata_path = self.root / subset / "dataset.json"
            try:
                metadata = json.loads(metadata_path.read_text())
                self._validate_metadata(metadata)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{metadata_path}: {exc}") from exc
            scene_paths = sorted((metadata_path.parent / "scenes").glob("scene_*"))
            if len(scene_paths) != metadata.get("scene_count"):
                raise ValueError(f"{metadata_path}: scene_count mismatch")
            count = 0
            for path in scene_paths:
                scene = self._read_scene(path, metadata)
                self.scenes[subset, path.name] = scene
                self.frame_keys.extend(FrameKey(subset, path.name, int(fid)) for fid in scene.frame_ids)
                count += len(scene.frame_ids)
            if count != metadata.get("frame_count"):
                raise ValueError(f"{metadata_path}: frame_count mismatch")
        self.frame_keys.sort()

    @staticmethod
    def _validate_metadata(meta: dict[str, Any]) -> None:
        if meta["format"] != "colmap_rgbd_v1" or meta["schema_version"] != 1:
            raise ValueError("unsupported RGB-D schema")
        if meta["camera"]["extrinsics"] != "opencv_world_to_camera" or meta["camera"]["intrinsics"] != "pixel_units":
            raise ValueError("expected OpenCV world-to-camera poses and pixel intrinsics")
        image, depth = meta["image"], meta["depth"]
        if image["dtype"] != "uint8" or image["channels"] != 3:
            raise ValueError("RGB must be uint8 with three channels")
        if depth["dtype"] != "uint16" or depth["unit"] != "millimeters" or depth["invalid_value"] != 0:
            raise ValueError("depth must be uint16 millimetres with zero invalid")
        if any(image[axis] != depth[axis] or image[axis] < 1 for axis in ("height", "width")):
            raise ValueError("image/depth dimensions must agree and be positive")

    @staticmethod
    def _read_scene(path: Path, metadata: dict[str, Any]) -> Scene:
        camera_path = path / "cameras.npz"
        try:
            with np.load(camera_path, allow_pickle=False) as source:
                ids = source["frame_ids"]
                K, poses = source["intrinsics"], source["extrinsics_w2c"]
                quality, chunks = source["quality_flags"], source["chunk_ids"]
            n = len(ids)
            if ids.shape != (n,) or ids.dtype.kind not in "iu" or n == 0 or len(np.unique(ids)) != n:
                raise ValueError("frame_ids must be unique nonempty integer IDs")
            if (ids < 0).any():
                raise ValueError("negative frame ID")
            if K.shape != (n, 3, 3) or poses.shape != (n, 3, 4) or chunks.shape != (n,) or quality.shape != (n,):
                raise ValueError("camera array shape/length mismatch")
            if chunks.dtype.kind not in "iu" or (chunks < 0).any():
                raise ValueError("invalid chunk IDs")
            if not np.isin(quality, [True]).all():
                raise ValueError("quality_flags contains false/unknown values; inspect export provenance")
            for i, fid in enumerate(ids):
                if not np.isfinite(K[i]).all() or not np.isfinite(poses[i]).all():
                    raise ValueError(f"frame {fid}: nonfinite camera")
                if K[i, 0, 0] <= 0 or K[i, 1, 1] <= 0 or abs(np.linalg.det(K[i])) < 1e-12:
                    raise ValueError(f"frame {fid}: invalid intrinsics")
                if not np.allclose(K[i, 2], [0, 0, 1], atol=1e-6):
                    raise ValueError(f"frame {fid}: invalid homogeneous intrinsics row")
                R = poses[i, :3, :3]
                if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(R), 1, atol=1e-3):
                    raise ValueError(f"frame {fid}: invalid rotation")
            return Scene(path, metadata, ids, K, poses, chunks, {int(fid): i for i, fid in enumerate(ids)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{camera_path}: {exc}") from exc

    def _lookup(self, key: FrameKey) -> tuple[Scene, int]:
        try:
            scene = self.scenes[key.subset, key.scene]
            return scene, scene.index[key.frame_id]
        except KeyError as exc:
            raise ValueError(f"{self.root}: unknown frame {key.name()}") from exc

    def frame_paths(self, key: FrameKey) -> tuple[Path, Path]:
        scene, _ = self._lookup(key)
        name = f"frame_{key.frame_id:06d}.png"
        return scene.path / "rgb" / name, scene.path / "depth" / name

    def camera(self, key: FrameKey) -> tuple[Array, Array, int]:
        scene, i = self._lookup(key)
        pose = np.eye(4, dtype=np.float64)
        pose[:3] = scene.extrinsics_w2c[i]
        return scene.intrinsics[i].astype(np.float64), pose, int(scene.chunk_ids[i])

    def load_depth(self, key: FrameKey) -> Array:
        scene, _ = self._lookup(key)
        _, path = self.frame_paths(key)
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        meta = scene.metadata["depth"]
        if depth is None or depth.dtype != np.uint16 or depth.shape != (meta["height"], meta["width"]):
            raise ValueError(f"{path}: missing/invalid uint16 depth shape")
        if int(depth.max()) > meta["max_depth_mm"]:
            raise ValueError(f"{path}: depth exceeds declared max_depth_mm")
        return depth.astype(np.float32) * 0.001

    def load_frame(self, key: FrameKey, *, load_rgb: bool = True) -> Frame:
        scene, _ = self._lookup(key)
        K, pose, chunk = self.camera(key)
        if load_rgb:
            path, _ = self.frame_paths(key)
            rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            meta = scene.metadata["image"]
            if rgb is None or rgb.dtype != np.uint8 or rgb.shape != (meta["height"], meta["width"], 3):
                raise ValueError(f"{path}: missing/invalid uint8 RGB shape")
            image = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
        else:
            image = torch.empty(0)
        return Frame(key, image, self.load_depth(key), K, pose, chunk)

    def file_manifest(self) -> dict[str, str]:
        paths = {self.root / subset / "dataset.json" for subset in SUBSETS}
        for scene in self.scenes.values():
            paths.update(scene.path.glob("*.npz"))
        for key in self.frame_keys:
            paths.update(self.frame_paths(key))
        return {str(path.relative_to(self.root)): file_sha256(path) for path in sorted(paths)}

    def fingerprint(self) -> str:
        return json_hash(self.file_manifest())
