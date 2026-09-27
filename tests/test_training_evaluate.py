"""Fixed GT denominators, eligible bins and inference-compatible exports."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from modules.lighterglue import LighterGlue
from modules.utils import state_hash
from modules.xfeat import XFeat
from xfeat_training.evaluate import aggregate_metrics, bundle_state, sample_descriptors, score_predictions, select_best


def test_ignore_predictions_and_zero_denominators() -> None:
    gt0, gt1 = np.array([0, -1, -2]), np.array([0, -1, -2])
    score = score_predictions(np.array([[0, 0], [1, 1], [2, 2]]), gt0, gt1)
    assert (score["TP"], score["P"], score["G"], score["A"]) == (1, 2, 1, 3)
    assert score["precision"] == 0.5 and score["recall"] == 1
    assert score["f1"] == pytest.approx(2 / 3)
    assert score["precision_lower_bound"] == pytest.approx(1 / 3)
    assert score["ignored_predictions"] == 1
    empty = score_predictions(np.empty((0, 2), int), gt0, gt1)
    assert empty["f1"] == empty["precision"] == empty["recall"] == 0
    assert empty["precision_lower_bound"] is None
    no_gt = score_predictions(np.array([[0, 0]]), np.array([-1]), np.array([-1]))
    assert no_gt["f1"] is None


def test_bins_sum_counts_before_macro_and_keep_eligible_set() -> None:
    rows = [
        {"subset": "s0", "gap": 5, "TP": 1, "P": 2, "G": 1, "A": 2},
        {"subset": "s0", "gap": 5, "TP": 0, "P": 0, "G": 9, "A": 0},
        {"subset": "s1", "gap": 10, "TP": 3, "P": 3, "G": 3, "A": 3},
        {"subset": "s0", "gap": 40, "TP": 10, "P": 10, "G": 10, "A": 10},
    ]
    result = aggregate_metrics(rows, eligible_bins=[("s0", 5), ("s1", 10)])
    assert result["primary_f1"] == pytest.approx((2 / 12 + 1) / 2)
    assert len(result["bins"]) == 3
    with pytest.raises(ValueError, match="eligible"):
        aggregate_metrics(rows, eligible_bins=[])
    with pytest.raises(ValueError, match=r"missing|eligible"):
        aggregate_metrics(rows[:2], eligible_bins=[("s0", 5), ("s1", 10)])
    candidates = [
        {"step": 10, "primary_f1": 0.5, "TP": 9},
        {"step": 5, "primary_f1": 0.5, "TP": 9},
        {"step": 3, "primary_f1": 0.4, "TP": 20},
    ]
    assert select_best(candidates)["step"] == 5


def test_fixed_anchor_descriptors_equal_live_extraction() -> None:
    torch.manual_seed(20)
    model = XFeat(device="cpu", top_k=64)
    image = torch.rand(1, 3, 96, 128)
    features = model.detectAndCompute(image)[0]
    descriptors = sample_descriptors(model, image, features["keypoints"])
    torch.testing.assert_close(descriptors, features["descriptors"], atol=1e-6, rtol=1e-5)


def test_export_bundle_has_all_strict_keys_and_actual_extractor() -> None:
    extractor = XFeat(device="cpu").net
    matcher = LighterGlue(device="cpu", flash=False).net
    with torch.no_grad():
        next(extractor.parameters()).add_(0.001)
    state = bundle_state(matcher, extractor)
    assert len(state) == 291 and "matcher.confidence_thresholds" not in state
    restored = LighterGlue(weights=state, device="cpu", flash=False)
    assert restored.extractor_hash == state_hash(extractor.state_dict())
    for key, tensor in matcher.state_dict().items():
        assert torch.equal(tensor, restored.net.state_dict()[key])


def test_prepare_evaluate_export_and_hash_guards(l76_factory, tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from modules.typecheck import SparseFeatures
    from xfeat_training import mining
    from xfeat_training.data import L76Dataset, file_sha256
    from xfeat_training.evaluate import EvaluationSuite, evaluate_model, export_task, prepare_evaluation
    from xfeat_training.features import FeatureCache

    root = l76_factory(ids=tuple(range(0, 60, 5)))
    cfg = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/pair_mining/default.yaml")))
    cfg.update(
        chunk_size=2,
        exclude_before={},
        min_frames={"train": 4, "val": 2, "test": 2},
        split_chunks={"train": [0, 1], "guard": [2, 4], "val": [3], "test": [5]},
        archive_path=None,
    )
    cfg.pop("defaults")
    cfg.pop("hydra")
    cfg.pop("data_root")
    monkeypatch.setattr(
        mining,
        "overlap",
        lambda a, b, **kw: 0.6 if (a.chunk_id == b.chunk_id or max(a.chunk_id, b.chunk_id) < 2) else 0.0,
    )
    mining.mine_pairs(root, tmp_path / "pairs", cfg)
    extractor = XFeat(device="cpu", top_k=1024)

    def grid_features(self, image, **kwargs):
        points = torch.tensor([[16.0, 16.0], [32.0, 16.0], [48.0, 16.0], [16.0, 32.0], [32.0, 32.0], [48.0, 32.0]])
        return [
            SparseFeatures(keypoints=points, descriptors=sample_descriptors(self, image, points), scores=torch.ones(6))
        ]

    monkeypatch.setattr(XFeat, "detectAndCompute", grid_features)
    dataset = L76Dataset(root)
    cache = FeatureCache(dataset, tmp_path / "cache", extractor)
    cache.build(dataset.frame_keys)
    manifest = prepare_evaluation(tmp_path / "pairs", tmp_path / "cache", tmp_path / "evaluation")
    original = {name: file_sha256(tmp_path / "evaluation" / name) for name in manifest["identity"]["files"]}
    assert len(manifest["identity"]["eligible_bins"]["val"]) == 2
    assert any(row["pair_count"] == 0 for row in manifest["identity"]["bins"]["val"])
    suite = EvaluationSuite(tmp_path / "evaluation")
    result = evaluate_model(suite, extractor, None, "val", tmp_path / "result", fine_diagnostic=True)
    assert result["primary_f1"] == 1.0 and result["pair_count"] == 2
    assert result["synthetic_fine"]["correspondences"] > 0
    assert result["evaluation_hash"] == manifest["content_hash"]
    assert (tmp_path / "result/pairs_live.csv").is_file() and result["missing_bins"]
    with torch.no_grad():
        next(extractor.net.parameters()).add_(0.001)
    changed = evaluate_model(suite, extractor, None, "val", tmp_path / "candidate")
    assert changed["evaluation_hash"] == result["evaluation_hash"]
    assert original == {name: file_sha256(tmp_path / "evaluation" / name) for name in original}
    matcher = LighterGlue(device="cpu", flash=False, width_confidence=-1)
    run = tmp_path / "run"
    run.mkdir()
    task = SimpleNamespace(
        config={"run_dir": str(run), "optimizer": {"name": "amuse"}, "top_k": 1024, "matcher": {"flash": False}},
        extractor=extractor,
        matcher=matcher,
        identities={"fixture": True},
    )
    export_task(task, "best", 5)
    restored = LighterGlue(weights=run / "exports/best_lighterglue.pt", device="cpu", flash=False)
    assert restored.extractor_hash == state_hash(extractor.net.state_dict())
    assert len(torch.load(run / "exports/best_xfeat.pt", weights_only=True)) == 122
    assert json.loads((run / "exports/best_manifest.json").read_text())["parameter_state"] == "X"
    name = next(iter(original))
    (tmp_path / "evaluation" / name).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        suite._read(name)


def test_reprojection_scopes_and_unknown_depth(plane_frame):
    from xfeat_training.evaluate import reprojection_metrics

    frame0, frame1 = plane_frame(), plane_frame(tx=0.01)
    points0 = np.array([[20.0, 20.0], [30.0, 30.0]])
    points1 = points0 + np.array([1.0, 0.0])
    result = reprojection_metrics(points0, points1, frame0, frame1)
    assert result["all_correct_1px"] == result["interior_matches"] == 2
    frame1.depth[:] = 0
    result = reprojection_metrics(points0, points1, frame0, frame1)
    assert result["all_matches"] == 2 and result["visible_matches"] == 0
    assert result["all_correct_3px"] == 0 and result["visible_precision_3px"] is None
