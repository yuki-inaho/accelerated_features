import json

import numpy as np
import pytest
import torch
from torch import nn


@pytest.mark.parametrize(
    "mode,values,stop",
    [
        ("max", [0.1, 0.1, 0.1, 0.12, 0.13, 0.14], False),
        ("max", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], True),
        ("max", [0.1, 0.1, 0.1, 0.09, 0.08, 0.07], True),
        ("min", [4.0, 4.0, 4.0, 3.9, 3.8, 3.7], False),
        ("min", [4.0, 4.0, 4.0, 4.0, 4.0, 4.0], True),
        ("min", [4.0, 4.0, 4.0, 4.1, 4.2, 4.3], True),
    ],
)
def test_plateau_direction_mean_peak_and_tolerance(mode, values, stop):
    from xfeat_training.convergence import plateau_decision

    result = plateau_decision(values, mode=mode, window=3, minimum_gain=0.01)
    assert result["stop"] is stop
    assert result["reason"] == ("validation_plateau" if stop else "validation_improving")
    with pytest.raises(ValueError, match="enough"):
        plateau_decision(values[:2], mode=mode, window=3, minimum_gain=0.01)
    with pytest.raises(ValueError, match="finite"):
        plateau_decision([float("nan")] * 6, mode=mode, window=3, minimum_gain=0.01)


def test_stop_callback_runs_after_checkpoint_and_flushed_logs(tmp_path):
    from xfeat_training.optim import build_optimizer
    from xfeat_training.trainer import PairSampler, train_loop

    class Task:
        def __init__(self):
            self.model = nn.Linear(2, 2)
            self.calls = 0

        def loss(self, index, accepted):
            self.calls += 1
            if self.calls == 1:
                self.skip_reason = "fixture_skip"
                return None
            return {"loss": self.model(torch.rand(1, 2)).square().mean()}

        def evaluate(self, step):
            return {"primary_f1": 0.5, "TP": 1}

        def export(self, prefix, step):
            path = tmp_path / "run/exports"
            path.mkdir(exist_ok=True)
            torch.save(self.model.state_dict(), path / f"{prefix}.pt")

        def stop_reason(self, step):
            if step != 4:
                return None
            path = tmp_path / "run"
            assert (path / "checkpoints/step_000004.pt.json").exists()
            rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
            assert rows[-1]["step"] == step
            from xfeat_training.trainer import pair_index_digest

            assert rows[-1]["attempted_pair_digest"] == pair_index_digest(rows[-1]["pair_indices"])
            assert rows[-1]["accepted_pair_digest"] == pair_index_digest([0])
            assert rows[0]["attempted_pair_digest"] == pair_index_digest([0, 0])
            assert rows[0]["accepted_pair_digest"] == pair_index_digest([0])
            assert rows[0]["microbatch_skip_rate"] == 0.5
            assert rows[-1]["microbatch_skip_rate"] == 0.0
            assert rows[-1]["cumulative_microbatch_skip_rate"] == 0.2
            return "validation_plateau"

    config = {
        "run_dir": str(tmp_path / "run"),
        "resume_from": None,
        "max_steps": 8,
        "seed": 2,
        "accumulation": 1,
        "grad_clip": 1.0,
        "precision": "fp32",
        "save_every": 2,
        "eval_every": 2,
        "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
    }
    task = Task()
    optimizer = build_optimizer(task.model, "xfeat", config["optimizer"])
    pairs = {"subset": np.array(["s"]), "difficulty": np.array(["overlap_0"]), "overlap": np.array([0.9])}
    sampler = PairSampler(pairs, {"bin_weights": [1.0]}, 2)
    result = train_loop(config, task, optimizer, sampler, {})
    assert result["successful_step"] == 4
    assert json.loads((tmp_path / "run/completed.json").read_text())["stop_reason"] == "validation_plateau"


def test_continuation_history_is_checkpointed_across_resume(tmp_path, assert_nested_equal):
    from xfeat_training.optim import build_optimizer
    from xfeat_training.trainer import PairSampler, train_loop

    class Task:
        def __init__(self):
            torch.manual_seed(99)
            self.model = nn.Linear(2, 2)
            self.continuation_history = {"validation": [], "losses": []}

        def loss(self, index, accepted):
            return {"loss": self.model(torch.rand(1, 2)).square().mean()}

        def evaluate(self, step):
            return {"primary_f1": 0.5, "TP": 1, "step": step}

        def export(self, prefix, step):
            pass

        def stop_reason(self, step):
            if step == 6:
                assert [v["step"] for v in self.continuation_history["validation"]] == [2, 4, 6]
                assert len(self.continuation_history["losses"]) == 6
                return "validation_plateau"
            return None

    def run(name, stop=None, resume=None):
        config = {
            "run_dir": str(tmp_path / name),
            "resume_from": str(resume) if resume else None,
            "stop_after_steps": stop,
            "max_steps": 8,
            "seed": 2,
            "auto_stop": {"enabled": True},
            "accumulation": 1,
            "grad_clip": 1.0,
            "precision": "fp32",
            "save_every": 2,
            "eval_every": 2,
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
        }
        task = Task()
        optimizer = build_optimizer(task.model, "xfeat", config["optimizer"])
        pairs = {"subset": np.array(["s"]), "difficulty": np.array(["overlap_0"]), "overlap": np.array([0.9])}
        sampler = PairSampler(pairs, {"bin_weights": [1.0]}, 2)
        return train_loop(config, task, optimizer, sampler, {})

    full = run("full")
    run("cut", stop=4)
    resumed = run("resumed", resume=tmp_path / "cut/checkpoints/step_000004.pt")
    assert_nested_equal(full, resumed)
