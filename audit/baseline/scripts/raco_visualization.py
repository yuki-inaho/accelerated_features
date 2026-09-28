"""Reproducible TUM RGB-D figures for the released XFeat ranking/covariance heads."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
from itertools import pairwise
from pathlib import Path
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from modules.raco import XFeatRaCo

SEQUENCES = {
    "freiburg3_nostructure_texture_far": "e4a921052d342fd52fac74509bb75f65149acbda3d31b25f36402041625fdba9",
    "freiburg3_structure_texture_far": "3f58c707f54c93b68fecd77293630e96a76d7b0f2703eb8ad9cff45ba4bbb81a",
}
MODEL_NAME = "xfeat-raco-rgbd-best.pt"
MODEL_SHA = "fd02f809283132b905b7ad151fcabfc0c8d8ad613ed12d3fda589764e48e2575"
RELEASE_URL = "https://github.com/yuki-inaho/accelerated_features/releases/download/rgbd-raco-v1/"
DATA_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_URL = "https://cvg.cit.tum.de/data/datasets/rgbd-dataset"
FRAME_INDICES = tuple(range(90, 181, 3))
K_CAMERA = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]], dtype=np.float64)


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, path: Path, expected_sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=60) as source, partial.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if file_sha(partial) != expected_sha:
            raise ValueError(f"Downloaded checksum mismatch: {path.name}")
        partial.replace(path)
    if file_sha(path) != expected_sha:
        raise ValueError(f"Cached checksum mismatch: {path.name}")


def _table(text: str) -> list[list[str]]:
    return sorted(
        (line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")),
        key=lambda row: float(row[0]),
    )


def prepare_data(root: Path) -> dict:
    """Fetch fixed archives; extract only metadata and the declared public frames."""
    manifests = {}
    for sequence, digest in SEQUENCES.items():
        stem = "rgbd_dataset_" + sequence
        archive = root / "data" / (stem + ".tgz")
        download(DATA_URL + archive.name, archive, digest)
        destination = root / "data" / stem
        saved = destination / "selection.json"
        if saved.exists():
            manifest = json.loads(saved.read_text())
            if manifest["archive_sha256"] != digest or [r["index"] for r in manifest["frames"]] != list(FRAME_INDICES):
                raise ValueError("Cached selection differs from declared inputs")
            if not all(file_sha(destination / name) == value for name, value in manifest["file_sha256"].items()):
                raise ValueError("Cached selection checksum mismatch")
            manifests[sequence] = manifest
            print(f"Verified {sequence}: {len(manifest['frames'])} frames")
            continue
        with tarfile.open(archive, "r:gz") as package:

            def read_member(relative: str, prefix: str = stem) -> bytes:
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("Unsafe archive path")
                member = package.getmember(prefix + "/" + relative)
                if not member.isfile():
                    raise ValueError("Only regular archive files are accepted")
                stream = package.extractfile(member)
                assert stream is not None
                return stream.read()

            metadata = {name: read_member(name).decode() for name in ("rgb.txt", "depth.txt", "groundtruth.txt")}
            rgb, depth, poses = (_table(metadata[name]) for name in ("rgb.txt", "depth.txt", "groundtruth.txt"))
            depth_times = np.array([float(row[0]) for row in depth])
            pose_times = np.array([float(row[0]) for row in poses])
            selected = []
            files = set(metadata)
            for index in FRAME_INDICES:
                row = rgb[index]
                stamp = float(row[0])
                di, pi = int(np.abs(depth_times - stamp).argmin()), int(np.abs(pose_times - stamp).argmin())
                depth_delta, pose_delta = abs(depth_times[di] - stamp), abs(pose_times[pi] - stamp)
                selected.append(
                    {
                        "index": index,
                        "timestamp": stamp,
                        "rgb": row[1],
                        "depth": depth[di][1],
                        "pose": [float(value) for value in poses[pi][1:]],
                        "depth_delta_seconds": float(depth_delta),
                        "pose_delta_seconds": float(pose_delta),
                    }
                )
                files.update((row[1], depth[di][1]))
            hashes = {}
            # Read gzip members in archive order to avoid repeatedly decompressing it.
            ordered_files = sorted(files, key=lambda name: package.getmember(stem + "/" + name).offset_data)
            for name in ordered_files:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                content = read_member(name)
                target.write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
        manifest = {
            "sequence": sequence,
            "source": SOURCE_URL,
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "archive_url": DATA_URL + archive.name,
            "archive_sha256": digest,
            "frames": selected,
            "file_sha256": hashes,
        }
        (destination / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests[sequence] = manifest
        print(f"Prepared {sequence}: {len(selected)} frames")
    download(RELEASE_URL + MODEL_NAME, root / "models" / MODEL_NAME, MODEL_SHA)
    return manifests


def load_frames(root: Path, sequence: str) -> list[dict]:
    directory = root / "data" / ("rgbd_dataset_" + sequence)
    manifest = json.loads((directory / "selection.json").read_text())
    frames = []
    for row in manifest["frames"]:
        for field in ("rgb", "depth"):
            if file_sha(directory / row[field]) != manifest["file_sha256"][row[field]]:
                raise ValueError("Selected frame checksum mismatch")
        rgb = cv2.imread(str(directory / row["rgb"]), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(directory / row["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape != (480, 640, 3) or depth.shape != (480, 640):
            raise ValueError("Expected full-resolution TUM RGB/depth")
        if depth.dtype != np.uint16:
            raise ValueError("Expected uint16 depth")
        # Keep the declared RGB sequence intact. Unsynchronized frames have no GT diagnostic.
        synchronized = max(row["depth_delta_seconds"], row["pose_delta_seconds"]) <= 0.02
        frames.append(
            {
                **row,
                "image": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                "depth_m": depth.astype(float) / 5000,
                "synchronized": synchronized,
            }
        )
    return frames


def pose_matrix(pose) -> np.ndarray:
    """TUM tx ty tz qx qy qz qw pose to camera-to-world matrix."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Expected finite translation and quaternion")
    quaternion = values[3:]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Zero quaternion")
    x, y, z, w = quaternion / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = values[:3]
    return result


def project_points(points, source_depth, target_depth, transform, k=K_CAMERA):
    """Project source pixels into target; return pixels and depth-visible mask.

    transform maps source-camera coordinates into the target camera. Depth is
    sampled at the nearest pixel; invalid projections remain NaN.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError("Nonfinite points or transform")
    h, w = source_depth.shape
    ij = np.rint(points).astype(int)
    inside = (points[:, 0] >= 0) & (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
    z = source_depth[ij[:, 1].clip(0, h - 1), ij[:, 0].clip(0, w - 1)]
    valid = inside & np.isfinite(z) & (z > 0)
    xyz = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(k).T * z[:, None]
    target = xyz @ transform[:3, :3].T + transform[:3, 3]
    projected = target @ k.T
    uv = np.full_like(points, np.nan)
    positive = np.isfinite(target).all(axis=1) & (target[:, 2] > 0)
    uv[positive] = projected[positive, :2] / projected[positive, 2:3]
    h, w = target_depth.shape
    visible = positive & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    observed = np.zeros(len(points))
    xy = np.rint(uv[visible]).astype(int)
    observed[visible] = target_depth[xy[:, 1], xy[:, 0]]
    valid &= visible & np.isfinite(observed) & (observed > 0)
    valid &= np.abs(target[:, 2] - observed) <= np.maximum(0.05, 0.05 * target[:, 2])
    return uv, valid


def ellipse_parameters(covariance, scale=1.0):
    """Full width, height and angle (degrees) of a nominal 95% 2-D ellipse."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("Expected finite symmetric covariance")
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if eigenvalues[0] <= 0 or scale <= 0:
        raise ValueError("Covariance and ellipse scale must be positive")
    width, height = 2 * scale * np.sqrt(5.991 * eigenvalues[::-1])
    axis = vectors[:, -1]
    return width, height, np.degrees(np.arctan2(axis[1], axis[0]))


def cosine_mnn(source, target, threshold=0.8):
    """Cosine mutual-nearest-neighbor matches; index order breaks exact ties."""
    if not len(source) or not len(target):
        return np.empty((0, 2), dtype=int)
    source, target = np.asarray(source), np.asarray(target)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite descriptors")
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    similarity = source @ target.T
    forward, reverse = similarity.argmax(1), similarity.argmax(0)
    i = np.arange(len(source))
    keep = (reverse[forward] == i) & (similarity[i, forward] >= threshold)
    return np.column_stack((i[keep], forward[keep]))


def advance_tracks(current_ids, next_count, matches):
    """Propagate live initial-frame IDs; lost IDs cannot be restarted."""
    result = np.full(next_count, -1, dtype=int)
    if len(matches):
        source, target = np.asarray(matches).T
        live = np.asarray(current_ids)[source] >= 0
        result[target[live]] = np.asarray(current_ids)[source[live]]
    return result


@torch.no_grad()
def dense_fields(model: XFeatRaCo, image: torch.Tensor) -> dict[str, np.ndarray]:
    """Read native-pixel fields for one image with dimensions divisible by 32."""
    if image.ndim != 4 or image.shape[0] != 1 or any(size % 32 for size in image.shape[-2:]):
        raise ValueError("Dense visualization requires one image, dimensions divisible by 32")
    image = image.to(device=next(model.parameters()).device, dtype=torch.float32)
    features, logits, reliability, detector = model.net.forward_with_features(image)
    z = torch.cat((features, detector), 1)
    rank = model.heads.ranker
    correction = 2 * F.pixel_shuffle(rank.projection(rank.trunk(z)), 8).tanh()
    head = model.heads.covariance_head
    raw = F.pixel_shuffle(head.projection(head.trunk(z)), 8)[0].permute(1, 2, 0)
    a, b, c = raw.unbind(-1)
    lower = torch.stack((F.softplus(a), torch.zeros_like(a), b, F.softplus(c)), -1).reshape(*a.shape, 2, 2)
    covariance = lower @ lower.mT + model.heads.sigma_min**2 * torch.eye(2, device=z.device)
    probability = F.pixel_shuffle(logits.softmax(1)[:, :64], 8)
    score = probability * F.interpolate(reliability, size=image.shape[-2:], mode="bilinear", align_corners=False)
    result = {"rank_correction": correction[0, 0], "detector": score[0, 0], "covariance": covariance}
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite dense output")
    return {key: value.cpu().numpy() for key, value in result.items()}


def image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 255


@torch.no_grad()
def extract_frame(model: XFeatRaCo, image: np.ndarray, top_k=512) -> dict:
    """Evaluate both selectors on exactly one shared candidate pool."""
    candidate = model.candidates(image_tensor(image))[0]
    results = {}
    for mode in ("detector", "ranked"):
        prediction = model.predict(candidate, top_k=top_k, ranking=mode == "ranked", covariance=True)
        results[mode] = {key: value.cpu().numpy() for key, value in prediction.items()}
    return results


def match_diagnostics(source_frame, target_frame, source, target, matches):
    """Bidirectional depth-visible reprojection error, never used for tracking."""
    errors = np.full(len(matches), np.nan)
    if not source_frame["synchronized"] or not target_frame["synchronized"] or not len(matches):
        return errors
    a, b = source["keypoints"][matches[:, 0]], target["keypoints"][matches[:, 1]]
    transform = np.linalg.inv(pose_matrix(target_frame["pose"])) @ pose_matrix(source_frame["pose"])
    projected_b, visible_b = project_points(a, source_frame["depth_m"], target_frame["depth_m"], transform)
    projected_a, visible_a = project_points(
        b, target_frame["depth_m"], source_frame["depth_m"], np.linalg.inv(transform)
    )
    eligible = visible_a & visible_b
    error = np.maximum(np.linalg.norm(projected_b - b, axis=1), np.linalg.norm(projected_a - a, axis=1))
    errors[eligible] = error[eligible]
    return errors


def spatial_subset(points: np.ndarray) -> list[int]:
    """First (highest-ranked) point in each of the fixed 4x4 image cells."""
    cells = set()
    selected = []
    for index, (x, y) in enumerate(points):
        cell = (int(x // 160), int(y // 120))
        if cell not in cells:
            cells.add(cell)
            selected.append(index)
    return selected


def _ellipses(ax, prediction, scale):
    points, ranks = prediction["keypoints"], prediction["ranker_scores"]
    normalization = Normalize(-3, 3)
    palette = plt.get_cmap("RdYlGn")
    ax.scatter(*points.T, c=ranks, cmap=palette, norm=normalization, s=5, alpha=0.55, linewidths=0)
    for index in spatial_subset(points):
        width, height, angle = ellipse_parameters(prediction["covariances"][index], scale)
        color = palette(normalization(ranks[index]))
        ax.add_patch(Ellipse(points[index], width, height, angle=angle, facecolor="none", edgecolor="black", lw=2.8))
        ax.add_patch(Ellipse(points[index], width, height, angle=angle, facecolor="none", edgecolor=color, lw=1.5))
        ax.scatter(*points[index], c=[color], s=22, edgecolors="black", linewidths=0.5)
    return plt.cm.ScalarMappable(norm=normalization, cmap=palette)


def _image_axis(fig, bounds, image, title):
    ax = fig.add_axes(bounds)
    ax.imshow(image)
    ax.set_title(title, fontsize=12, pad=9)
    ax.set_axis_off()
    return ax


def render_pipeline(frame, fields, prediction, sequence, output: Path) -> list[Path]:
    """Render actual dense outputs and native-coordinate covariance ellipses."""
    output.mkdir(parents=True, exist_ok=True)
    name = "Textured plane" if "nostructure" in sequence else "Textured 3D structure"
    image = frame["image"]
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    fig.suptitle(f"XFeat + RaCo-inspired heads  |  {name}", x=0.03, ha="left", fontsize=22, weight="bold", y=0.98)
    fig.text(
        0.03,
        0.925,
        f"Released best model  /  RGB index {frame['index']}  /  640 × 480 pixels",
        fontsize=12,
        color="#555555",
    )
    _image_axis(fig, [0.015, 0.36, 0.17, 0.30], image, "Input RGB")
    fig.add_artist(Rectangle((0.215, 0.16), 0.052, 0.69, transform=fig.transFigure, fc="#e9edf1", ec="#cad0d7"))
    fig.text(0.241, 0.505, "Frozen XFeat features", rotation=90, ha="center", va="center", fontsize=16, weight="bold")

    def field(bounds, values, title, cmap, limits, label, extend="neither", gamma=1):
        ax = fig.add_axes(bounds)
        plotted = ax.imshow(values, cmap=cmap, norm=PowerNorm(gamma, vmin=limits[0], vmax=limits[1]))
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_axis_off()
        x, y, width, height = bounds
        bar_axis = fig.add_axes((x + width + 0.005, y, 0.005, height))
        bar = fig.colorbar(plotted, cax=bar_axis, orientation="vertical", extend=extend)
        bar.set_label(label, fontsize=9)
        bar.ax.tick_params(labelsize=8)
        return ax

    field(
        [0.34, 0.615, 0.21, 0.2835],
        fields["rank_correction"],
        "Ranker · learned correction  2 tanh(δ)",
        "coolwarm",
        (-2, 2),
        "dimensionless",
    )
    field(
        [0.34, 0.30, 0.21, 0.2835],
        fields["detector"],
        "Detector · probability × reliability",
        "magma",
        (0, 1),
        "score (power scale γ = 0.3)",
        gamma=0.3,
    )
    sigma = np.sqrt(np.diagonal(fields["covariance"], axis1=-2, axis2=-1))
    correlation = fields["covariance"][..., 0, 1] / (sigma[..., 0] * sigma[..., 1])
    fig.text(0.47, 0.255, "Covariance estimator · Σ in pixel²", ha="center", fontsize=13, weight="bold")
    for x, values, title, cmap, limits, label in (
        (0.295, sigma[..., 0], "σx", "viridis", (0, 3), "px"),
        (0.414, sigma[..., 1], "σy", "viridis", (0, 3), "px"),
        (0.533, correlation, "ρxy", "coolwarm", (-1, 1), "correlation"),
    ):
        field([x, 0.095, 0.085, 0.11475], values, title, cmap, limits, label, "max" if label == "px" else "neither")

    result_ax = _image_axis(fig, [0.705, 0.28, 0.28, 0.44], image, "Ranked keypoints + anisotropic ellipses")
    scalar = _ellipses(result_ax, prediction, scale=4)
    result_ax.set_xlim(-0.5, 639.5)
    result_ax.set_ylim(479.5, -0.5)
    color_ax = fig.add_axes((0.742, 0.25, 0.205, 0.015))
    bar = fig.colorbar(scalar, cax=color_ax, orientation="horizontal")
    bar.set_label("Final candidate rank score", fontsize=11)
    fig.text(
        0.845, 0.145, "Top 512 points · up to 16 ellipses\nEllipse axes ×4 for visibility", ha="center", fontsize=12
    )
    for start, end in (
        [((0.185, 0.51), (0.215, 0.51))]
        + [((0.267, y), (0.332, y)) for y in (0.755, 0.44)]
        + [((0.267, 0.16), (0.29, 0.16))]
    ):
        fig.add_artist(
            FancyArrowPatch(
                start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15, color="#384657", lw=1.5
            )
        )
    # Branch connector outside all image and colorbar bounds.
    for y in (0.755, 0.44, 0.16):
        fig.add_artist(
            FancyArrowPatch((0.65, y), (0.682, y), transform=fig.transFigure, arrowstyle="-", color="#384657", lw=1.5)
        )
    fig.add_artist(plt.Line2D([0.682, 0.682], [0.16, 0.755], transform=fig.transFigure, color="#384657", lw=1.5))
    fig.add_artist(
        FancyArrowPatch(
            (0.682, 0.51),
            (0.704, 0.51),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            color="#384657",
            lw=1.5,
        )
    )
    fig.text(
        0.03,
        0.035,
        "TUM RGB-D · Sturm et al., IROS 2012 · CC BY 4.0  |  Ellipses: nominal 95% Gaussian shape; calibration not established.",
        fontsize=10,
        color="#555555",
    )
    pipeline = output / f"{sequence}_pipeline.png"
    fig.savefig(pipeline, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    ax.imshow(image)
    _ellipses(ax, prediction, scale=1)
    ax.set_xlim(160, 480)
    ax.set_ylim(360, 120)
    ax.set_xlabel("x [image pixels]")
    ax.set_ylabel("y [image pixels]")
    ax.set_title(f"{name} · central crop\nEllipse axes ×1 (native pixel units); nominal 95% shape", fontsize=14)
    crop = output / f"{sequence}_ellipse_native.png"
    fig.savefig(crop, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)
    return [pipeline, crop]


def evaluate_sequence(model, frames, sequence, output: Path, top_k=512):
    """Run both selectors, adjacent matching and initial-frame-only tracks."""
    predictions = [extract_frame(model, frame["image"], top_k) for frame in frames]
    rows, summaries, tracks, edges = [], [], {}, {}
    for mode in ("detector", "ranked"):
        first = predictions[0][mode]
        ids = np.arange(len(first["keypoints"]))
        positions = np.full((len(frames), len(ids), 2), np.nan)
        positions[0] = first["keypoints"]
        mode_rows, errors_all, mode_edges = [], [], []
        for edge, (frame_a, frame_b) in enumerate(pairwise(frames)):
            a, b = predictions[edge][mode], predictions[edge + 1][mode]
            matches = cosine_mnn(a["descriptors"], b["descriptors"])
            # Tracking uses RGB descriptors only. GT diagnostics run afterwards.
            ids = advance_tracks(ids, len(b["keypoints"]), matches)
            live = ids >= 0
            positions[edge + 1, ids[live]] = b["keypoints"][live]
            errors = match_diagnostics(frame_a, frame_b, a, b, matches)
            eligible = errors[np.isfinite(errors)]
            correct = int((eligible <= 3).sum())
            row = {
                "sequence": sequence,
                "mode": mode,
                "source_index": frame_a["index"],
                "target_index": frame_b["index"],
                "synchronized": bool(frame_a["synchronized"] and frame_b["synchronized"]),
                "matches": len(matches),
                "eligible": len(eligible),
                "correct": correct,
                "precision": correct / len(eligible) if len(eligible) else None,
                "median_error_px": float(np.median(eligible)) if len(eligible) else None,
                "live_initial_tracks": int(live.sum()),
            }
            mode_rows.append(row)
            errors_all.extend(eligible.tolist())
            mode_edges.append({"matches": matches, "errors": errors})
        rows.extend(mode_rows)
        total_matches = sum(row["matches"] for row in mode_rows)
        total_eligible = sum(row["eligible"] for row in mode_rows)
        total_correct = sum(row["correct"] for row in mode_rows)
        summaries.append(
            {
                "sequence": sequence,
                "mode": mode,
                "edges": len(mode_rows),
                "synchronized_edges": sum(row["synchronized"] for row in mode_rows),
                "matches": total_matches,
                "mean_matches": total_matches / len(mode_rows),
                "eligible": total_eligible,
                "correct": total_correct,
                "precision": total_correct / total_eligible if total_eligible else None,
                "median_error_px": float(np.median(errors_all)) if errors_all else None,
                "initial_tracks": positions.shape[1],
                "final_tracks": int(np.isfinite(positions[-1, :, 0]).sum()),
            }
        )
        tracks[mode], edges[mode] = positions, mode_edges
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{sequence}_tracks.npz", **tracks)
    return {
        "frames": frames,
        "predictions": predictions,
        "rows": rows,
        "summary": summaries,
        "tracks": tracks,
        "edges": edges,
    }


def save_metrics(results: dict, output: Path):
    rows = [row for result in results.values() for row in result["rows"]]
    summaries = [row for result in results.values() for row in result["summary"]]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    return summaries


def render_matches(result: dict, sequence: str, output: Path) -> Path:
    """Fixed index105→108 pair, first80 MNNs in source selection order."""
    slot = 5
    frames, predictions = result["frames"], result["predictions"]
    canvas = np.concatenate((frames[slot]["image"], frames[slot + 1]["image"]), axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), layout="constrained")
    for ax, mode in zip(axes, ("detector", "ranked"), strict=True):
        ax.imshow(canvas)
        edge = result["edges"][mode][slot]
        a, b = predictions[slot][mode], predictions[slot + 1][mode]
        for (i, j), error in zip(edge["matches"][:80], edge["errors"][:80], strict=True):
            color = "#12c982" if error <= 3 else "#f44f5e" if np.isfinite(error) else "#b8bfc9"
            start, end = a["keypoints"][i], b["keypoints"][j] + [640, 0]
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=0.65, alpha=0.8)
            ax.scatter([start[0], end[0]], [start[1], end[1]], s=8, c=color)
        errors = edge["errors"]
        count = int(np.isfinite(errors).sum())
        correct = int((errors <= 3).sum())
        ax.set_title(
            f"{mode.capitalize()} top512  |  {len(errors)} matches, {correct}/{count} GT-eligible correct  |  first80 shown",
            fontsize=13,
        )
        ax.set_axis_off()
    fig.suptitle(
        f"{sequence}\nRGB 105 → 108 · cosine MNN ≥0.8 · green ≤3px, red >3px, gray GT unavailable", fontsize=14
    )
    path = output / f"{sequence}_matches.png"
    fig.savefig(path, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)
    return path


def render_tracks(result: dict, sequence: str, output: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), layout="constrained")
    palette = plt.get_cmap("tab20")
    for row, mode in enumerate(("detector", "ranked")):
        positions = result["tracks"][mode]
        for column, slot in enumerate((0, 10, 20, 30)):
            ax = axes[row, column]
            ax.imshow(result["frames"][slot]["image"])
            for identity in range(min(16, positions.shape[1])):
                if not np.isfinite(positions[slot, identity]).all():
                    continue
                history = positions[max(0, slot - 7) : slot + 1, identity]
                color = palette(identity)
                ax.plot(*history.T, color=color, lw=1.8)
                point = positions[slot, identity]
                ax.scatter(*point, c=[color], s=20, edgecolors="black", linewidths=0.5)
                ax.annotate(
                    str(identity),
                    point + np.array([3, -3]),
                    color="white",
                    fontsize=7,
                    bbox={"fc": "black", "alpha": 0.5, "pad": 0.3, "ec": "none"},
                )
            live = int(np.isfinite(positions[slot, :, 0]).sum())
            ax.set_title(
                f"{mode.capitalize()} · RGB {result['frames'][slot]['index']}\n{live}/{positions.shape[1]} initial tracks alive",
                fontsize=11,
            )
            ax.set_xlim(-0.5, 639.5)
            ax.set_ylim(479.5, -0.5)
            ax.set_axis_off()
    fig.suptitle(
        f"{sequence}\nFixed initial IDs 0–15 shown · trails up to8 frames · lost tracks never restart · RGB-only matching",
        fontsize=15,
    )
    path = output / f"{sequence}_tracks.png"
    fig.savefig(path, dpi=100, pil_kwargs={"optimize": True})
    plt.close(fig)
    return path


def render_survival(results: dict, output: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(12, 4.5), squeeze=False, layout="constrained")
    for ax, (sequence, result) in zip(axes[0], results.items(), strict=True):
        stamps = np.array([frame["timestamp"] for frame in result["frames"]])
        for mode, color in (("detector", "#2763a4"), ("ranked", "#d55e00")):
            live = np.isfinite(result["tracks"][mode][..., 0]).sum(axis=1)
            ax.plot(stamps - stamps[0], live, marker=".", lw=2, color=color, label=mode)
        name = "Textured plane" if "nostructure" in sequence else "Textured 3D structure"
        ax.set(title=name, xlabel="Elapsed time [s]", ylabel="Surviving initial tracks", ylim=(0, 530))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Track survival · same top512 budget and cosine MNN rule; no restart, no GT filtering", fontsize=13)
    path = output / "track_survival.png"
    fig.savefig(path, dpi=120, pil_kwargs={"optimize": True})
    plt.close(fig)
    return path
