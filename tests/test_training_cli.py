"""Hydra read-only config, deliberate rejection and cwd-independent paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["xfeat", "lighterglue"])
def test_hydra_resolved_config_outside_repository_is_read_only(tmp_path, task) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "xfeat_training.train", "experiment=smoke", f"task={task}", "--cfg", "job", "--resolve"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"task: {task}" in result.stdout and "max_steps: 12" in result.stdout
    assert "flash: false" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("override", ["typo=1", "--multirun"])
def test_invalid_cli_never_creates_run(tmp_path, override) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "xfeat_training.train", override, f"run_dir={tmp_path / 'run'}"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "experiment,optimizer,expected",
    [
        ("lg_a", "amuse", 1000),
        ("xfeat_b", "amuse", 500),
        ("lg_c", "amuse", 1000),
        ("lg_a", "adamw", 1000),
    ],
)
def test_experiment_configs(tmp_path, experiment, optimizer, expected):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xfeat_training.train",
            f"experiment={experiment}",
            f"optimizer={optimizer}",
            "--cfg",
            "job",
            "--resolve",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"max_steps: {expected}" in result.stdout and f"name: {optimizer}" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments", [[], ["experiment=lg_c"], ["pairs_dir=missing"], ["-m"]])
def test_required_paths_fail_without_output(tmp_path, arguments):
    result = subprocess.run(
        [sys.executable, "-m", "xfeat_training.train", *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


def test_existing_run_and_resume_input_untouched_on_cli_rejection(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    checkpoint = old / "checkpoint.pt"
    checkpoint.write_bytes(b"sentinel")
    new.mkdir()
    (new / "sentinel").write_bytes(b"existing run")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = subprocess.run(
        [sys.executable, "-m", "xfeat_training.train", f"run_dir={new}", f"resume_from={checkpoint}"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "already exists" in result.stderr
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
