"""Both real task branches, including evaluation, export and boundary resume."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from modules.typecheck import SparseFeatures
from modules.xfeat import XFeat
from xfeat_training import mining
from xfeat_training.data import L76Dataset, file_sha256
from xfeat_training.evaluate import prepare_evaluation, sample_descriptors
from xfeat_training.features import FeatureCache
from xfeat_training.trainer import run_training


@pytest.mark.parametrize("target_device", ["cpu", "cuda"])
def test_both_training_tasks_resume_with_real_evaluation(l76_factory, tmp_path, monkeypatch, target_device):
    if target_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    root = l76_factory(ids=tuple(range(0, 60, 5)), width=96)
    cfg = cast(dict[str, Any], OmegaConf.to_container(OmegaConf.load("configs/pair_mining/default.yaml")))
    cfg.update(
        chunk_size=2,
        exclude_before={},
        min_frames={"train": 4, "val": 2, "test": 2},
        split_chunks={"train": [0, 1], "guard": [2, 4], "val": [3], "test": [5]},
        archive_path=None,
    )
    for key in ("defaults", "hydra", "data_root"):
        cfg.pop(key)
    monkeypatch.setattr(
        mining,
        "overlap",
        lambda a, b, **kw: 0.6 if (a.chunk_id == b.chunk_id or max(a.chunk_id, b.chunk_id) < 2) else 0.0,
    )
    mining.mine_pairs(root, tmp_path / "pairs", cfg)

    def grid_features(self, image, **kwargs):
        points = torch.tensor(
            [[16.0, 16.0], [32.0, 16.0], [48.0, 16.0], [16.0, 32.0], [32.0, 32.0], [48.0, 32.0]], device=self.dev
        )
        return [
            SparseFeatures(
                keypoints=points,
                descriptors=sample_descriptors(self, image, points),
                scores=torch.ones(6, device=self.dev),
            )
        ]

    monkeypatch.setattr(XFeat, "detectAndCompute", grid_features)
    dataset = L76Dataset(root)
    extractor = XFeat(device="cpu", top_k=1024)
    FeatureCache(dataset, tmp_path / "cache", extractor).build(dataset.frame_keys)
    prepare_evaluation(tmp_path / "pairs", tmp_path / "cache", tmp_path / "eval")
    official = {key: value.clone() for key, value in extractor.net.state_dict().items()}
    for task in ("lighterglue", "xfeat"):
        with initialize_config_dir(version_base="1.3", config_dir=str(mining.REPO_ROOT / "configs")):
            resolved = OmegaConf.to_container(
                compose(
                    config_name="train",
                    overrides=[
                        "experiment=smoke",
                        f"task={task}",
                        f"device={target_device}",
                        f"pairs_dir={tmp_path / 'pairs'}",
                        f"cache_dir={tmp_path / 'cache'}",
                        f"eval_cache_dir={tmp_path / 'eval'}",
                        "max_steps=2",
                        "save_every=1",
                        "eval_every=1",
                        f"run_dir={tmp_path / (task + '_full')}",
                    ],
                ),
                resolve=True,
            )
        config = cast(dict[str, Any], resolved)
        run_training(config)
        config.update(run_dir=str(tmp_path / (task + "_cut")), stop_after_steps=1)
        run_training(config)
        before = {p.name: p.read_bytes() for p in (tmp_path / (task + "_cut") / "checkpoints").iterdir()}
        config.update(
            run_dir=str(tmp_path / (task + "_resume")),
            stop_after_steps=None,
            resume_from=str(tmp_path / (task + "_cut") / "checkpoints/step_000001.pt"),
        )
        run_training(config)
        assert before == {p.name: p.read_bytes() for p in (tmp_path / (task + "_cut") / "checkpoints").iterdir()}
        a = torch.load(tmp_path / (task + "_full") / "checkpoints/step_000002.pt", weights_only=False)
        b = torch.load(tmp_path / (task + "_resume") / "checkpoints/step_000002.pt", weights_only=False)

        def compare(x, y):
            if isinstance(x, torch.Tensor):
                if target_device == "cpu":
                    assert torch.equal(x, y)
                else:
                    torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
            elif isinstance(x, dict):
                for key in x:
                    compare(x[key], y[key])
            elif isinstance(x, (list, tuple)):
                for xx, yy in zip(x, y, strict=True):
                    compare(xx, yy)
            else:
                assert x == y

        compare(a["model"], b["model"])
        compare(a["optimizer"], b["optimizer"])
        assert a["state"]["sampler"]["history"] == b["state"]["sampler"]["history"]
        assert a["state"]["accepted_microbatches"] == 8
        assert a["signature"] == b["signature"]
        for suffix in ("_full", "_cut", "_resume"):
            directory = tmp_path / (task + suffix)
            for prefix in ("best", "last"):
                exported = json.loads((directory / "exports" / f"{prefix}_manifest.json").read_text())
                reference = directory / exported["checkpoint"]
                assert exported["checkpoint_status"] == "complete"
                assert exported["checkpoint_file_sha256"] == file_sha256(reference)
        rows = [json.loads(line) for line in (tmp_path / (task + "_full") / "metrics.jsonl").read_text().splitlines()]
        if task == "xfeat":
            assert all(row["losses"]["synthetic"] == 0.5 for row in rows)
            for key in a["model"]:
                if key not in dict(extractor.net.named_parameters()):
                    assert torch.equal(a["model"][key], official[key])
            for prefix in ("block_fusion", "heatmap_head", "keypoint_head", "fine_matcher"):
                assert any(
                    norm > 0
                    for row in rows
                    for name, norm in row["parameter_gradient_norms"].items()
                    if name.startswith(prefix)
                )
