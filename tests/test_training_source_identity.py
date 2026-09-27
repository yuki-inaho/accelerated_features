"""Runtime identity is independent of documentation and Git bookkeeping."""

from __future__ import annotations

import copy
import json

import pytest
import torch
from torch import nn

from xfeat_training import mining
from xfeat_training.data import file_sha256, json_hash
from xfeat_training.optim import build_optimizer
from xfeat_training.trainer import checkpoint_signature, preflight_run, save_checkpoint


@pytest.fixture
def runtime_tree(tmp_path, monkeypatch):
    files = {
        "xfeat_training/train.py": "train",
        "modules/model.py": "model",
        "scripts/cache_features.py": "cache",
        "third_party/__init__.py": "",
        "third_party/amuse/AMUSE.py": "optimizer",
        "configs/train.yaml": "training: true",
        "pyproject.toml": "[project]",
        "uv.lock": "version = 1",
        "requirements.txt": "torch",
        ".python-version": "3.12",
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(mining, "REPO_ROOT", tmp_path)

    def reject_git(*args, **kwargs):
        raise AssertionError("Runtime identity must not depend on Git")

    monkeypatch.setattr(mining.subprocess, "check_output", reject_git)
    return tmp_path, set(files)


def test_runtime_identity_excludes_docs_tests_and_git(runtime_tree):
    root, expected = runtime_tree
    original = mining.runtime_source_identity()
    assert set(original["files"]) == expected
    for name in ("README.md", "tests/test_model.py", "docs/run.md", "third_party/amuse/LICENSE", ".git/HEAD"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("non-runtime change")
    assert mining.runtime_source_identity() == original
    first = {"source": {"head": "old", "dirty_diff_hash": "unstaged"}, "runtime_source": original, "pair_hash": "pairs"}
    second = {**first, "source": {"head": "new", "dirty_diff_hash": "staged", "files": {"README.md": "new"}}}
    assert first != second
    assert checkpoint_signature({}, first) == checkpoint_signature({}, second)


@pytest.mark.parametrize("path", ["modules/model.py", "configs/train.yaml", "third_party/amuse/AMUSE.py"])
@pytest.mark.parametrize("operation", ["edit", "add", "delete"])
def test_runtime_identity_detects_content_and_membership(runtime_tree, path, operation):
    root, _ = runtime_tree
    original = mining.runtime_source_identity()
    target = root / path
    if operation == "edit":
        target.write_text("changed")
    elif operation == "add":
        target.with_name("new_" + target.name).write_text("new")
    else:
        target.unlink()
    changed = mining.runtime_source_identity()
    assert changed != original
    a = {"source": {}, "runtime_source": original}
    b = {"source": {}, "runtime_source": changed}
    assert checkpoint_signature({}, a) != checkpoint_signature({}, b)


@pytest.mark.parametrize("name", ["pyproject.toml", "uv.lock", "requirements.txt", ".python-version"])
def test_runtime_dependency_file_changes(runtime_tree, name):
    root, _ = runtime_tree
    original = mining.runtime_source_identity()
    (root / name).write_text("changed")
    assert mining.runtime_source_identity() != original
    (root / name).unlink()
    if name == ".python-version":
        assert mining.runtime_source_identity() != original
    else:
        with pytest.raises(FileNotFoundError):
            mining.runtime_source_identity()


def test_source_provenance_requires_valid_runtime_identity(runtime_tree):
    with pytest.raises(ValueError, match="runtime_source"):
        checkpoint_signature({}, {"source": {"head": "old"}})
    runtime = mining.runtime_source_identity()
    corrupted = copy.deepcopy(runtime)
    corrupted["files"]["modules/model.py"] = "changed"
    with pytest.raises(ValueError, match="runtime_source"):
        checkpoint_signature({}, {"source": {}, "runtime_source": corrupted})
    assert runtime["content_hash"] == json_hash(runtime["files"])


@pytest.fixture
def source_checkpoint(runtime_tree, amuse_config):
    root, _ = runtime_tree
    config = {"max_steps": 12, "optimizer": amuse_config}
    identities = {
        "source": {"head": "original", "files": {"README.md": "old"}},
        "runtime_source": mining.runtime_source_identity(),
        "pair_hash": "pairs",
        "cache_hash": "cache",
        "evaluation_hash": "evaluation",
        "source_weights": {"extractor": "weights"},
    }
    model = nn.Linear(4, 4)
    optimizer = build_optimizer(model, "xfeat", amuse_config)
    path = root / "old/checkpoints/step_000000.pt"
    signature = checkpoint_signature(config, identities)
    save_checkpoint(
        path,
        model,
        optimizer,
        {"successful_step": 0, "microstep": 0},
        signature,
        config=config,
        identities=identities,
    )
    return root, path, config, identities, signature


def test_checkpoint_metadata_snapshot_and_provenance_only_resume(source_checkpoint):
    root, path, config, identities, signature = source_checkpoint
    original = copy.deepcopy(identities)
    identities["source"]["head"] = "committed"
    identities["source"]["files"]["README.md"] = "edited"
    assert checkpoint_signature(config, identities) == signature
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    loaded = preflight_run(root / "new", path, checkpoint_signature(config, identities))
    assert loaded is not None and loaded["schema_version"] == 2
    assert loaded["identities"] == original
    assert loaded["config"] == config
    assert checkpoint_signature(loaded["config"], loaded["identities"]) == signature
    assert not (root / "new").exists()
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}


@pytest.mark.parametrize(
    "kind", ["code", "config", "dependency", "pair_hash", "cache_hash", "evaluation_hash", "source_weights"]
)
def test_runtime_or_input_change_rejected_before_output(source_checkpoint, kind):
    root, path, config, identities, _ = source_checkpoint
    if kind in {"code", "config", "dependency"}:
        name = {"code": "modules/model.py", "config": "configs/train.yaml", "dependency": "uv.lock"}[kind]
        (root / name).write_text("modified")
        identities["runtime_source"] = mining.runtime_source_identity()
    else:
        identities[kind] = "modified"
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(ValueError, match="signature"):
        preflight_run(root / "new", path, checkpoint_signature(config, identities))
    assert not (root / "new").exists()
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}


@pytest.mark.parametrize("change", ["old_schema", "missing_metadata", "inconsistent_metadata"])
def test_checkpoint_schema_and_metadata_checked(source_checkpoint, change):
    root, path, _, _, signature = source_checkpoint
    sidecar_path = path.with_suffix(".pt.json")
    sidecar = json.loads(sidecar_path.read_text())
    if change == "old_schema":
        sidecar["schema_version"] = 1
    else:
        payload = torch.load(path, weights_only=False)
        if change == "missing_metadata":
            del payload["identities"]
        else:
            payload["config"]["max_steps"] += 1
        torch.save(payload, path)
        sidecar["file_sha256"] = file_sha256(path)
    sidecar_path.write_text(json.dumps(sidecar))
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    with pytest.raises(ValueError, match=r"schema|metadata"):
        preflight_run(root / "new", path, signature)
    assert not (root / "new").exists()
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
