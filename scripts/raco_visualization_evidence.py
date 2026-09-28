"""Supplemental visual evidence and executable old/new parity checks."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw, ImageFont

from modules.raco import XFeatRaCo
from scripts.raco_paper_visualization import (
    DisplayConfig, covariance_map, load_paired_samples, resolve_visualization_paths,
)
from scripts.raco_visualization import dense_fields, file_sha, image_tensor


def before_after(output: Path, tag: str, sequence: str) -> Path:
    """Same input/checkpoint before/after. BEFORE is the legacy renderer, rerun."""
    before = Image.open(output / "legacy_diagnostic" / f"{sequence}_pipeline.png").convert("RGB")
    after = Image.open(output / f"{tag}_paper.png").convert("RGB")
    width = 2000
    def resize(image):
        return image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
    before, after = resize(before), resize(after)
    header, gap = 78, 78
    canvas = Image.new("RGB", (width, before.height + after.height + header + gap), "white")
    canvas.paste(before, (0, header))
    canvas.paste(after, (0, header + before.height + gap))
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 25)
    small = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 19)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 12), f"BEFORE / original diagnostic renderer rerun on CPU / {tag} / RGB 105", font=font, fill="black")
    draw.text((30, 47), "Same released weights, image, candidate pool and ranked top512. No training or geometric changes.", font=small, fill="#555555")
    draw.text((30, header + before.height + 10), "AFTER / Figure 13 visual conventions / same numerical model outputs", font=font, fill="black")
    draw.text((30, header + before.height + 44), "New displayed quantities: detector probability, complete rank score, dense covariance long-axis hue + determinant whitening.", font=small, fill="#555555")
    path = output / f"before_after_{tag}.png"
    canvas.save(path)
    return path


def synthetic_orientation_proof(path: Path, config: DisplayConfig = DisplayConfig()) -> Path:
    """An explicitly synthetic test of angle signs, long vs short axis, whitening."""
    angles = [0, 30, 45, 60, 90, 120, 135, 150]
    determinants = [.04, 1., 4.]
    fig = plt.figure(figsize=(14.5, 5.8), facecolor="white")
    fig.text(.02, .94, "Synthetic verification / major-axis angle and determinant whitening", fontsize=18, weight="bold")
    fig.text(.02, .89, "These are analytic test matrices, NOT model predictions. Angles increase counterclockwise in the x-right/y-up legend.", fontsize=10)
    checks = []
    for row, det in enumerate(determinants):
        for col, angle in enumerate(angles):
            t = np.radians(angle)
            rotation = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            cov = rotation @ np.diag([4., 1.]) @ rotation.T * np.sqrt(det / 4)
            result = covariance_map(cov, config)
            got = float(result["angle_degrees"])
            error = abs((got - angle + 90) % 180 - 90)
            assert error < 1e-10
            assert np.isclose(result["determinant_px4"], det)
            ax = fig.add_axes([.12 + col * .107, .12 + (2 - row) * .25, .10, .20])
            rgb = result["rgb"]
            ax.set_facecolor(rgb)
            ax.add_patch(Ellipse((0, 0), 1.6, .8, angle=angle, fill=False, edgecolor="black", lw=1.0))
            ax.plot([-.72*np.cos(t), .72*np.cos(t)], [-.72*np.sin(t), .72*np.sin(t)], color="black", lw=1)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_color(".65")
                spine.set_linewidth(.5)
            ax.set_title(f"{angle}°", fontsize=10, pad=3)
            if col == 0:
                fig.text(.014, .215 + (2 - row) * .25, f"det Σ = {det:g} px⁴\nα = {float(result['opacity']):.3f}", fontsize=10, va="center")
            checks.append({"angle_expected_degrees": angle, "angle_measured_degrees": got, "absolute_axis_error_degrees": error,
                           "determinant_px4": float(result["determinant_px4"]), "opacity": float(result["opacity"])})
    fig.text(.02, .052, "Hue is 180°-periodic (v and -v are the same axis). Larger det Σ is whiter. Isotropic axes are undefined and rendered white.", fontsize=10)
    fig.text(.02, .018, "Ellipse outlines are normalized 2:1 orientation glyphs, not metric sizes. Matrices have eigenvalue ratio 4:1 at all three determinant levels.", fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
    return path


def determinant_sensitivity(output: Path) -> Path:
    with np.load(output / "raw/scene_01_b_fields.npz", allow_pickle=False) as arrays:
        cov = arrays["covariance"].copy()
    fig = plt.figure(figsize=(16, 5.3), facecolor="white")
    fig.text(.02, .95, "Whitening sensitivity / exactly the same covariance tensor", fontsize=18, weight="bold")
    for col, scale in enumerate((.25, 1., 4.)):
        result = covariance_map(cov, replace(DisplayConfig(), det_scale_px4=scale))
        ax = fig.add_axes([.02 + col * .325, .16, .315, .71])
        ax.imshow(result["rgb"], interpolation="nearest")
        ax.set_title(f"τ = {scale:g} px⁴" + (" / delivered default" if scale == 1 else ""), fontsize=12)
        ax.set_axis_off()
    fig.text(.02, .08, "α = exp[-det(Σ)/τ]. τ is an explicitly chosen display parameter, not a value recovered from the paper or a calibrated probability.", fontsize=11)
    fig.text(.02, .035, "No per-image min/max, percentile normalization, blur, anisotropy-based contrast, detector mask, or post-hoc whitening was applied.", fontsize=10)
    path = output / "whitening_sensitivity.png"
    fig.savefig(path, dpi=128)
    plt.close(fig)
    return path


def rank_semantics(output: Path) -> Path:
    fig = plt.figure(figsize=(16, 8.8), facecolor="white")
    fig.text(.025, .955, "Ranker correction is not the final ranking score", fontsize=20, weight="bold")
    for row, tag in enumerate(("scene_01_b", "scene_06_b")):
        with np.load(output / "raw" / f"{tag}_fields.npz", allow_pickle=False) as arrays:
            correction, final = arrays["rank_correction"], arrays["rank_score"]
        y = .15 + (1 - row) * .365
        for col, (value, cmap, limits, title) in enumerate((
            (correction, "coolwarm", (-2, 2), "Before: correction only / coolwarm [-2,2]"),
            (correction, "turbo", (-2, 2), "Style-only change: correction / turbo [-2,2]"),
            (final, "turbo", (-3, 3), "Delivered: final-score extension / turbo [-3,3]"),
        )):
            ax = fig.add_axes([.04 + col * .318, y, .308, .33])
            ax.imshow(value, cmap=cmap, norm=Normalize(*limits), interpolation="nearest")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title, fontsize=10, pad=6)
            if col == 0:
                fig.text(.017, y + .165, tag, rotation=90, fontsize=10, va="center")
    fig.text(.04, .09, "Left and middle use the same tensor. Right adds the normalized log detector-score prior used by the actual candidate selector.", fontsize=11)
    fig.text(.04, .045, "Candidate-pool statistics are held fixed before top-k. Noncandidate pixels are a display extension of the sparse formula, not validated dense predictions.", fontsize=10)
    path = output / "rank_semantics.png"
    fig.savefig(path, dpi=128)
    plt.close(fig)
    return path


def verify_baseline(root: Path, output: Path, *, weights: Path | None = None) -> dict:
    """Compare the uploaded helper's original numeric paths to the new ones."""
    source = root / "audit/baseline/scripts/raco_visualization.py"
    spec = importlib.util.spec_from_file_location("raco_viz_original_snapshot", source)
    assert spec and spec.loader
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data, default_weights, _ = resolve_visualization_paths(root)
    weights = default_weights if weights is None else weights
    model = XFeatRaCo.from_bundle(weights, device="cpu").eval()
    rows = []
    for frame in load_paired_samples(data):
        tensor = image_tensor(frame["image"])
        before = old.dense_fields(model, tensor)
        after = dense_fields(model, tensor)
        errors = {}
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
            errors[key] = float(np.max(np.abs(before[key] - after[key])))
        before_pred = old.extract_frame(model, frame["image"], top_k=512)["ranked"]
        with torch.no_grad():
            candidate = model.candidates(tensor)[0]
            after_pred = model.predict(candidate, top_k=512, ranking=True, covariance=True)
        for key, value in before_pred.items():
            np.testing.assert_array_equal(value, after_pred[key].cpu().numpy())
        rows.append({"sample": frame["sample_id"] + "_" + frame["view"],
                     "old_new_dense_max_absolute_errors": errors,
                     "all_sparse_prediction_arrays_bitwise_equal": True})
    result = {"scope": "Actual uploaded original helper vs modified helper, same CPU model and each of 20 RGBs",
              "baseline_helper_sha256": file_sha(source), "new_helper_sha256": file_sha(root / "scripts/raco_visualization.py"),
              "comparison_scope": "Legacy dense fields and sparse predictions are checked against the preserved pre-change helper.",
              "views": len(rows), "results": rows}
    (output / "baseline_parity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_evidence(root: Path, output: Path, verify: bool = True, *, weights: Path | None = None) -> list[Path]:
    paths = [
        before_after(output, "scene_01_b", "freiburg3_nostructure_texture_far"),
        before_after(output, "scene_06_b", "freiburg3_structure_texture_far"),
        synthetic_orientation_proof(output / "orientation_whitening_proof.png"),
        determinant_sensitivity(output), rank_semantics(output),
    ]
    if verify:
        verify_baseline(root, output, weights=weights)
    return paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    paths = build_evidence(root, root / "evidence")
    print("Saved supplemental evidence:", "\n".join(str(p) for p in paths), sep="\n")
