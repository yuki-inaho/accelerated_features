"""Figure-13-style visualization, without changing XFeat/RaCo-inspired predictions.

Paper: arXiv:2602.15755v1, Appendix D. The paper specifies determinant-weighted
whitening but does not give its transfer function/scale there. Our explicit
choice is alpha = exp(-det(Sigma) / det_scale), common to every image.
No claim of reproducing the original RaCo model or its exact renderer is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize, hsv_to_rgb

from modules.raco import XFeatRaCo
from modules.utils import state_hash
from scripts.raco_visualization import (
    MODEL_NAME, MODEL_SHA, dense_fields, file_sha, image_tensor, render_pipeline,
)

# These colors reproduce the *visual convention* visible in the supplied Fig.13.
# Exact author-side lookup tables were not available; do not describe them as exact.
DETECTOR_CMAP = LinearSegmentedColormap.from_list(
    "paper_black_green_white", [(0, (0, 0, 0)), (0.5, (0, 1, 0)), (1, (1, 1, 1))], N=1024
)
PAPER_URL = "https://arxiv.org/html/2602.15755v1#A4"

def resolve_visualization_paths(root: Path, *, model_name: str = MODEL_NAME) -> tuple[Path, Path, Path]:
    """Resolve local inputs without embedding machine-specific paths in notebooks."""
    root = Path(root).expanduser()
    downloads = Path.home() / "Downloads"

    def choose(env_name: str, candidates: list[Path], *, file: bool = False) -> Path:
        configured = os.environ.get(env_name)
        if configured:
            return Path(configured).expanduser()
        for candidate in candidates:
            exists = candidate.is_file() if file else candidate.is_dir()
            if exists:
                return candidate
        return candidates[0]

    data = choose("RACO_DATA_DIR", [
        root / "data" / "tum_rgbd_pairs_10",
        downloads / "tum_rgbd_pairs_10",
    ])
    weights = choose("RACO_WEIGHTS", [
        root / "weights" / model_name,
        downloads / model_name,
    ], file=True)
    output = Path(os.environ.get(
        "RACO_OUTPUT_DIR", str(root / "outputs" / "raco-paper-visualization")
    )).expanduser()
    return data, weights, output




@dataclass(frozen=True)
class DisplayConfig:
    """Display-only constants. No per-image stretching or hidden thresholding."""
    detector_min: float = 0.0
    detector_max: float = 1.0
    rank_min: float = -3.0
    rank_max: float = 3.0
    det_scale_px4: float = 1.0
    isotropic_rtol: float = 1e-6
    top_k: int = 512
    dpi: int = 128

    def __post_init__(self) -> None:
        values = (self.detector_min, self.detector_max, self.rank_min, self.rank_max,
                  self.det_scale_px4, self.isotropic_rtol)
        if not np.isfinite(values).all():
            raise ValueError("Display parameters must be finite")
        if self.detector_min >= self.detector_max or self.rank_min >= self.rank_max:
            raise ValueError("Display limits must be increasing")
        if self.det_scale_px4 <= 0 or not 0 <= self.isotropic_rtol < 1:
            raise ValueError("Invalid determinant scale or isotropy tolerance")
        if type(self.top_k) is not int or self.top_k < 0 or type(self.dpi) is not int or self.dpi < 40:
            raise ValueError("top_k must be a nonnegative integer, dpi an integer >=40")


def orientation_rgb(theta: np.ndarray) -> np.ndarray:
    """Axis (not directed vector) colors, radians, modulo pi; x right, y up."""
    theta = np.asarray(theta, dtype=np.float64)
    if not np.isfinite(theta).all():
        raise ValueError("Angles must be finite")
    hue = np.remainder(theta, np.pi) / np.pi
    return hsv_to_rgb(np.stack((hue, np.ones_like(hue), np.ones_like(hue)), axis=-1))


def covariance_map(covariance: np.ndarray, config: DisplayConfig = DisplayConfig()) -> dict[str, np.ndarray]:
    """Long-axis hue + determinant-based whitening of an SPD covariance field.

    Sigma is in image coordinates (x right, y down), in px^2. The angle is
    counterclockwise in the x-right/y-up legend: atan2(-vy, vx) mod pi.
    Exact/nearly isotropic matrices have no stable axis; they are white and
    their angle is NaN. Nonfinite, asymmetric, and non-SPD inputs are errors.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape[-2:] != (2, 2) or not np.isfinite(cov).all():
        raise ValueError("Expected finite (...,2,2) covariance matrices")
    if not np.allclose(cov, np.swapaxes(cov, -1, -2), rtol=1e-7, atol=1e-9):
        raise ValueError("Covariances must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.any(eigenvalues[..., 0] <= 0):
        raise ValueError("Covariances must be positive definite")
    major = eigenvectors[..., :, -1]  # eigh is ascending: last is the long axis.
    theta = np.remainder(np.arctan2(-major[..., 1], major[..., 0]), np.pi)
    det = np.prod(eigenvalues, axis=-1)  # px^4, not trace / sigma / detector score.
    gap = eigenvalues[..., 1] - eigenvalues[..., 0]
    defined = gap > config.isotropic_rtol * eigenvalues[..., 1]
    alpha = np.exp(-det / config.det_scale_px4)
    alpha = np.where(defined, alpha, 0.0)
    rgb = alpha[..., None] * orientation_rgb(theta) + (1 - alpha[..., None])
    return {
        "rgb": rgb.astype(np.float32),
        "angle_degrees": np.where(defined, np.degrees(theta), np.nan),
        "determinant_px4": det,
        "opacity": alpha,
        "angle_defined": defined,
        "eigenvalues_px2": eigenvalues,
    }


def with_final_rank(fields: dict[str, np.ndarray], candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Extend the *actual* sparse ranking formula to every native pixel.

    Statistics come from the full, shared candidate pool BEFORE top-k, not
    from all pixels or the retained subset. Outside the pool this is explicitly
    a visualization extension of a sparsely trained head, not a probability.
    An empty pool has no reference statistics: rank_score is NaN, not invented.
    """
    scores = candidate["scores"]
    result: dict[str, Any] = dict(fields)
    if not len(scores):
        result["rank_score"] = np.full(fields["detector"].shape, np.nan, dtype=np.float32)
        result["rank_reference"] = {"count": 0, "log_mean": None, "log_std": None, "defined": False}
        return result
    transform = candidate["transform"].detach().cpu().numpy()
    if not np.allclose(transform, np.eye(3), atol=1e-8):
        raise ValueError("Native-pixel rank display requires an identity coordinate transform")
    with torch.no_grad():
        logp = (scores + 1e-8).log()
        mean = logp.mean()
        std = (logp.var(unbiased=False) + 1e-6).sqrt()
        score_map = torch.as_tensor(fields["detector"], device=scores.device, dtype=scores.dtype).clone()
        # The sparse API uses grid_sample, whereas the legacy dense map uses
        # interpolate. Their FP32 roundoff is amplified when log-score variance
        # is tiny. Honor the actual API scores at all candidate pixels, without
        # changing the legacy map or any model prediction.
        xy = candidate["keypoints"]
        score_map[xy[:, 1], xy[:, 0]] = scores
        correction = torch.as_tensor(fields["rank_correction"], device=scores.device, dtype=scores.dtype)
        rank = (((score_map + 1e-8).log() - mean) / std).tanh() + correction
    result["rank_score"] = rank.cpu().numpy()
    result["rank_reference"] = {"count": len(scores), "log_mean": float(mean), "log_std": float(std), "defined": True}
    return result


def load_paired_samples(directory: Path) -> list[dict[str, Any]]:
    """Load and hash-check the supplied 10-pair layout. Never downloads data."""
    directory = Path(directory)
    manifest_path = directory / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; extract tum_rgbd_pairs_10.tar.gz here")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        sample_dir = (directory / sid).resolve()
        if directory.resolve() not in sample_dir.parents:
            raise ValueError("Unsafe sample directory")
        pair = json.loads((sample_dir / "pair.json").read_text(encoding="utf-8"))
        for key in ("sample_id", "source_sequence", "views"):
            if pair[key] != sample[key]:
                raise ValueError(f"pair.json differs from dataset manifest: {sid}/{key}")
        for view in pair["views"]:
            paths = {}
            for kind in ("rgb", "depth"):
                path = (sample_dir / view[kind]["file"]).resolve()
                if sample_dir not in path.parents:
                    raise ValueError("Unsafe image path")
                digest = file_sha(path)
                if digest != view[kind]["sha256"] or digest != sample["files_sha256"][view[kind]["file"]]:
                    raise ValueError(f"Input checksum mismatch: {path}")
                paths[kind] = path
            bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
            if bgr is None or bgr.shape != (480, 640, 3) or depth is None or depth.shape != (480, 640) or depth.dtype != np.uint16:
                raise ValueError(f"Expected full-resolution TUM RGB-D: {sid}")
            frames.append({
                "sample_id": sid, "view": view["view"], "sequence": pair["source_sequence"],
                "index": view["rgb_index"], "timestamp": view["timestamp_seconds"],
                "image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                "rgb_sha256": view["rgb"]["sha256"], "depth_sha256": view["depth"]["sha256"],
                "rgb_path": paths["rgb"],
            })
    ids = [(f["sample_id"], f["view"]) for f in frames]
    if len(ids) != len(set(ids)) or len(frames) != 2 * manifest["image_pair_count"]:
        raise ValueError("Duplicate or missing views in supplied samples")
    return frames


def _gradient_axis(fig, bounds, cmap, low: float, high: float, labels: tuple[str, str]):
    ax = fig.add_axes(bounds)
    ax.imshow(np.linspace(low, high, 1024)[None, :], aspect="auto", cmap=cmap,
              norm=Normalize(low, high), extent=(low, high, 0, 1), interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([low, high], labels=labels, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.6)
    return ax


def _angle_legend(fig, bounds):
    ax = fig.add_axes(bounds)
    axis = np.linspace(-1.12, 1.12, 321)
    x, y = np.meshgrid(axis, np.linspace(-0.08, 1.12, 177))
    theta = np.remainder(np.arctan2(y, x), np.pi)
    rgb = orientation_rgb(theta)
    radius = np.hypot(x, y)
    mask = (radius >= .57) & (radius <= 1) & (y >= 0)
    rgba = np.dstack((rgb, mask.astype(float)))
    ax.imshow(rgba, origin="lower", extent=(-1.12, 1.12, -.08, 1.12), interpolation="nearest")
    t = np.linspace(0, np.pi, 400)
    for r in (.57, 1):
        ax.plot(r * np.cos(t), r * np.sin(t), color="black", lw=.65)
    for start, end in (((0, 0), (1.22, 0)), ((0, 0), (0, 1.18))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": .8, "color": "black"})
    ax.text(1.03, -.17, "0°", ha="center", fontsize=8)
    ax.text(-1.00, -.17, "180°", ha="center", fontsize=8)
    ax.text(.08, 1.08, "90°", fontsize=8)
    ax.set(xlim=(-1.23, 1.25), ylim=(-.26, 1.25), aspect="equal")
    ax.set_axis_off()
    return ax


def panel_arrays(record: dict, config: DisplayConfig) -> list[np.ndarray]:
    fields = record["fields"]
    detector = DETECTOR_CMAP(Normalize(config.detector_min, config.detector_max, clip=True)(fields["detector_probability"]))[..., :3]
    ranks = fields["rank_score"]
    rank_rgb = plt.get_cmap("turbo")(Normalize(config.rank_min, config.rank_max, clip=True)(np.nan_to_num(ranks)))[..., :3]
    rank_rgb[~np.isfinite(ranks)] = 1
    cov_rgb = covariance_map(fields["covariance"], config)["rgb"]
    return [record["frame"]["image"], detector, rank_rgb, cov_rgb]


def render_paper_grid(records: list[dict], path: Path, config: DisplayConfig = DisplayConfig()) -> Path:
    """Four shared columns, paper-like palettes and *fixed* scales for all rows."""
    if not records:
        raise ValueError("At least one image is needed")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h, gap, left = 4.0, 3.0, .035, .54
    footer, header = .92, 1.72
    width = left + 4 * tile_w + 3 * gap + .14
    height = footer + len(records) * tile_h + (len(records) - 1) * gap + header
    fig = plt.figure(figsize=(width, height), dpi=config.dpi, facecolor="white")
    top = height - header
    headings = ("Keypoints", "Detector Scoremap", "Ranker Scoremap", "Covariance Ellipse\nMajor Axis Angle")
    for col, title in enumerate(headings):
        x = left + col * (tile_w + gap)
        fig.text((x + tile_w / 2) / width, (top + 1.32) / height, title,
                 ha="center", va="center", fontsize=14, family="DejaVu Serif")
        if col == 1:
            _gradient_axis(fig, [(x + .4) / width, (top + .30) / height, 3.2 / width, .23 / height],
                           DETECTOR_CMAP, config.detector_min, config.detector_max,
                           (f"low  {config.detector_min:g}", f"high  {config.detector_max:g}"))
        elif col == 2:
            _gradient_axis(fig, [(x + .4) / width, (top + .30) / height, 3.2 / width, .23 / height],
                           "turbo", config.rank_min, config.rank_max,
                           (f"worse  {config.rank_min:g}", f"better  {config.rank_max:g}"))
        elif col == 3:
            _angle_legend(fig, [(x + .7) / width, (top + .03) / height, 2.6 / width, 1.0 / height])
        else:
            fig.text((x + tile_w / 2) / width, (top + .43) / height,
                     f"Ranked top {config.top_k}  ·  lime points", ha="center", fontsize=10, color=".30")
    for row, record in enumerate(records):
        y = footer + (len(records) - row - 1) * (tile_h + gap)
        frame = record["frame"]
        label = f"{frame['sample_id']} {frame['view']}  |  RGB {frame['index']}"
        fig.text(.25 / width, (y + tile_h / 2) / height, label, fontsize=9, rotation=90, ha="center", va="center")
        for col, array in enumerate(panel_arrays(record, config)):
            x = left + col * (tile_w + gap)
            ax = fig.add_axes([x / width, y / height, tile_w / width, tile_h / height])
            ax.imshow(array, origin="upper", interpolation="nearest")
            if col == 0:
                xy = record["prediction"]["keypoints"]
                if len(xy):
                    ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c="lime", linewidths=0)
            h, w = frame["image"].shape[:2]
            ax.set(xlim=(-.5, w - .5), ylim=(h - .5, -.5))
            ax.set_axis_off()
    notes = [
        "XFeat + RaCo-inspired heads / TUM RGB-D. Figure-13-style rendering, NOT original RaCo predictions. RGB and covariance are at native 640 × 480.",
        "Detector: XFeat cell-softmax probability (not probability × reliability). Ranker: full-score extension using shared candidate-pool statistics; not a probability.",
        f"Covariance: x right, y up; hue = major-axis angle mod 180°. White blend: α = exp[-det(Σ) / ({config.det_scale_px4:g} px⁴)]. Fixed scales; undefined isotropic axes are white.",
        "TUM RGB-D / Sturm et al., IROS 2012 / CC BY 4.0. Color overlays and layout added. Sparse training does not validate dense uncertainty away from candidates.",
    ]
    for i, text in enumerate(notes):
        fig.text(left / width, (.73 - i * .185) / height, text, fontsize=8.5, color=".25", va="center")
    fig.savefig(path, dpi=config.dpi, facecolor="white", pil_kwargs={"optimize": False})
    plt.close(fig)
    return path


def save_native(record: dict, directory: Path, config: DisplayConfig) -> list[Path]:
    """Native pixel evidence, without figure resampling or decorative overlays."""
    directory.mkdir(parents=True, exist_ok=True)
    arrays = panel_arrays(record, config)
    paths = []
    for name, array in zip(("rgb", "detector_probability", "rank_score", "covariance_angle"), arrays, strict=True):
        path = directory / (name + ".png")
        rgb = array if array.dtype == np.uint8 else np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {path}")
        paths.append(path)
    return paths


def _array_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        a = np.asarray(array)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def run(data: Path, weights: Path, output: Path, *, device: str = "cpu", config: DisplayConfig = DisplayConfig(),
        save_all_fields: bool = False, expected_model_sha256: str = MODEL_SHA) -> dict[str, Any]:
    """Offline inference of all supplied views, dense/sparse assertions + figures."""
    if file_sha(Path(weights)) != expected_model_sha256:
        raise ValueError("Model checkpoint SHA-256 does not match the selected release")
    torch.manual_seed(20260927)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = load_paired_samples(Path(data))
    model = XFeatRaCo.from_bundle(weights, device=device).eval()
    initial_state = state_hash(model.state_dict())
    audit, selected = [], []
    for frame in frames:
        tensor = image_tensor(frame["image"])
        candidate = model.candidates(tensor)[0]
        fields = with_final_rank(dense_fields(model, tensor), candidate)
        with torch.no_grad():
            sparse_rank = model.heads.rank(candidate["z"], candidate["keypoints"], candidate["scores"]).cpu().numpy()
            sparse_cov = model.heads.covariance(candidate["z"], candidate["keypoints"]).cpu().numpy()
            sparse_corr = (2 * model.heads.ranker(candidate["z"], candidate["keypoints"]).squeeze(-1).tanh()).cpu().numpy()
            pred = model.predict(candidate, top_k=config.top_k, ranking=True, covariance=True)
        prediction = {k: v.cpu().numpy() for k, v in pred.items()}
        x, y = candidate["keypoints"].T.cpu().numpy()
        errors = {}
        for name, dense, sparse in (
            ("rank_score", fields["rank_score"][y, x], sparse_rank),
            ("rank_correction", fields["rank_correction"][y, x], sparse_corr),
            ("covariance", fields["covariance"][y, x], sparse_cov),
            ("detector_effective", fields["detector"][y, x], candidate["scores"].cpu().numpy()),
        ):
            np.testing.assert_allclose(dense, sparse, rtol=1e-5, atol=1e-5)
            errors[name] = float(np.max(np.abs(dense - sparse))) if len(sparse) else 0.0
        cov = covariance_map(fields["covariance"], config)
        det = cov["determinant_px4"]
        ratio = cov["eigenvalues_px2"][..., 1] / cov["eigenvalues_px2"][..., 0]
        record = {"frame": frame, "fields": fields, "prediction": prediction}
        tag = frame["sample_id"] + "_" + frame["view"]
        before_hash = _array_digest(prediction)
        input_hash = _array_digest({k: v for k, v in fields.items() if isinstance(v, np.ndarray)})
        save_native(record, output / "native" / tag, config)
        if frame["view"] == "b" and frame["sample_id"] in ("scene_01", "scene_02", "scene_03", "scene_06", "scene_07", "scene_08"):
            selected.append(record)
        if frame["index"] == 105:
            render_paper_grid([record], output / f"{tag}_paper.png", config)
            render_pipeline(frame, fields, prediction, frame["sequence"], output / "legacy_diagnostic")
        if save_all_fields or frame["index"] == 105:
            (output / "raw").mkdir(exist_ok=True)
            np.savez_compressed(output / "raw" / f"{tag}_fields.npz",
                                **{k: v for k, v in fields.items() if isinstance(v, np.ndarray)},
                                **{"prediction_" + k: v for k, v in prediction.items()},
                                candidate_xy=candidate["keypoints"].cpu().numpy(),
                                candidate_scores=candidate["scores"].cpu().numpy(),
                                candidate_rank_scores=sparse_rank,
                                candidate_covariances=sparse_cov)
        assert before_hash == _array_digest(prediction), "Renderer modified sparse predictions"
        assert input_hash == _array_digest({k: v for k, v in fields.items() if isinstance(v, np.ndarray)}), "Renderer modified dense fields"
        row = {
            "sample": tag, "sequence": frame["sequence"], "rgb_index": frame["index"],
            "rgb_sha256": frame["rgb_sha256"], "depth_sha256": frame["depth_sha256"],
            "candidates": len(x), "selected": len(prediction["keypoints"]),
            "rank_reference": fields["rank_reference"], "dense_sparse_max_abs": errors,
            "min_eigenvalue_px2": float(cov["eigenvalues_px2"].min()),
            "determinant_quantiles_px4": np.quantile(det, [0, .25, .5, .75, 1]).tolist(),
            "median_eigenvalue_ratio": float(np.median(ratio)),
            "undefined_axis_pixels": int((~cov["angle_defined"]).sum()),
            "whitening_opacity_median": float(np.median(cov["opacity"])),
            "prediction_sha256_before_and_after": before_hash,
            "native_dense_sha256_before_and_after": input_hash,
        }
        audit.append(row)
        print(f"{tag}: N={len(x)}, top={len(prediction['keypoints'])}, rank err={errors['rank_score']:.3g}, min eig={row['min_eigenvalue_px2']:.4f}", flush=True)
    render_paper_grid(selected, output / "paper_style_gallery.png", config)
    assert state_hash(model.state_dict()) == initial_state, "Model state changed during visualization"
    metadata = {
        "paper": PAPER_URL, "paper_version": "2602.15755v1", "model_kind": "xfeat_raco_v1, NOT original RaCo",
        "model_file": Path(weights).name, "model_sha256": expected_model_sha256, "model_state_sha256_before_and_after": initial_state,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                        "opencv": cv2.__version__, "matplotlib": matplotlib.__version__, "device": device},
        "seed": 20260927, "threads": 4, "deterministic_algorithms": True,
        "display_config": asdict(config),
        "whitening_provenance": "Our disclosed fixed exponential transfer; Appendix D does not specify this exact function/scale",
        "rank_map_provenance": "Full candidate-score formula extended to noncandidate pixels using full candidate-pool statistics",
        "dataset_manifest_sha256": file_sha(Path(data) / "dataset_manifest.json"),
        "views": len(frames), "metrics_scope": "Visualization consistency only. No tracking / pose / calibrated-coverage claim.",
        "frames": audit,
    }
    (output / "verification.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"verification": metadata, "selected_records": selected}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    default_data, default_weights, default_output = resolve_visualization_paths(root)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--det-scale", type=float, default=1.0, help="Fixed exponential whitening scale in px^4, not author-specified")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--save-all-fields", action="store_true")
    args = parser.parse_args()
    run(args.data, args.weights, args.output, device=args.device,
        config=DisplayConfig(det_scale_px4=args.det_scale, top_k=args.top_k), save_all_fields=args.save_all_fields)


if __name__ == "__main__":
    main()
