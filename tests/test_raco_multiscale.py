"""Differentiable multiscale candidate path used only while training."""

import torch

from modules.raco import XFeatRaCo


def _enable_multiscale_parameters(model: XFeatRaCo) -> None:
    trainable = model.configure_multiscale_training()
    assert trainable
    assert all(
        name.startswith(("net.block3.", "net.block5.", "net.block_fusion.", "heads.ranker.", "heads.covariance_head."))
        for name in trainable
    )


def test_training_candidate_gradient_and_inference_parity():
    torch.manual_seed(20260928)
    model = XFeatRaCo(
        weights=None,
        device="cpu",
        candidate_limit=64,
        detection_threshold=0.001,
        border=4,
    )
    _enable_multiscale_parameters(model)
    model.train()
    assert not model.net.training

    image = torch.rand(1, 3, 64, 96)
    support = torch.ones(1, 1, 64, 96, dtype=torch.bool)
    buffers_before = {name: value.clone() for name, value in model.net.named_buffers()}
    with torch.no_grad():
        inference_candidates = model.candidates(image, support)[0]
        inference_output = model.predict(inference_candidates, top_k=64)

    training_candidates = model.training_candidates(image, support, include_feature_maps=True)[0]
    training_output = model.predict(training_candidates, top_k=64)
    assert len(training_candidates["keypoints"]) > 0

    for name in ("keypoints", "scores", "candidate_ids", "descriptors", "z"):
        torch.testing.assert_close(training_candidates[name], inference_candidates[name], atol=1e-6, rtol=1e-6)
    for name in ("keypoints", "descriptors", "scores", "candidate_ids", "ranker_scores", "covariances"):
        torch.testing.assert_close(training_output[name], inference_output[name], atol=1e-6, rtol=1e-6)
    assert training_candidates["z"].requires_grad
    assert training_candidates["descriptors"].requires_grad
    assert training_candidates["block3_features"].shape == (len(training_candidates["keypoints"]), 64)
    assert training_candidates["block5_features"].shape == (len(training_candidates["keypoints"]), 64)
    assert training_candidates["block3_features"].requires_grad
    assert training_candidates["block5_features"].requires_grad
    assert training_candidates["block3_map"].shape == (1, 64, 8, 12)
    assert training_candidates["block5_map"].shape == (1, 64, 2, 3)
    assert training_candidates["block3_map"].requires_grad
    assert training_candidates["block5_map"].requires_grad

    n = len(training_candidates["keypoints"])
    weight = torch.linspace(0.1, 1.0, n)[:, None]
    loss = (
        (training_candidates["descriptors"] * weight).sum()
        + training_candidates["block3_features"].square().mean()
        + training_candidates["block5_features"].square().mean()
        + training_candidates["block3_map"].square().mean()
        + training_candidates["block5_map"].square().mean()
        + training_output["ranker_scores"].square().mean()
        + 0.001 * training_output["covariances"].square().mean()
    )
    loss.backward()
    assert torch.isfinite(loss)

    allowed = ("block3.", "block5.", "block_fusion.")
    for name, parameter in model.net.named_parameters():
        if name.startswith(allowed):
            assert parameter.requires_grad and parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
        else:
            assert not parameter.requires_grad and parameter.grad is None, name
    for head in (model.heads.ranker, model.heads.covariance_head):
        for name, parameter in head.named_parameters():
            assert parameter.requires_grad and parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for name, value in model.net.named_buffers():
        torch.testing.assert_close(value, buffers_before[name], atol=0, rtol=0)


def _new_training_wrapper():
    from xfeat_training.raco_multiscale_task import RacoMultiscaleTrainingModel

    student = XFeatRaCo(
        weights=None,
        device="cpu",
        candidate_limit=64,
        detection_threshold=0.001,
    )
    auxiliary = {"probe": torch.nn.Linear(4, 3)}
    wrapper = RacoMultiscaleTrainingModel(student, auxiliary)
    wrapper.configure_multiscale_training()
    return wrapper


def test_aux_checkpoint_and_clean_bundle(tmp_path):
    import pytest

    from xfeat_training.optim import build_optimizer
    from xfeat_training.trainer import (
        checkpoint_signature,
        preflight_run,
        restore_checkpoint,
        save_checkpoint,
    )

    optimizer_config = {
        "name": "adamw",
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    config = {"task": "raco_multiscale", "optimizer": optimizer_config, "max_steps": 10}
    identities = {"runtime_id": "fixed-test-runtime"}
    signature = checkpoint_signature(config, identities)

    wrapper = _new_training_wrapper()
    optimizer = build_optimizer(wrapper, "raco_multiscale", optimizer_config)
    parameter_names = {name for group in optimizer.param_groups for name in group["param_names"]}
    assert any(name.startswith("student.net.block3.") for name in parameter_names)
    assert any(name.startswith("student.net.block5.") for name in parameter_names)
    assert any(name.startswith("student.net.block_fusion.") for name in parameter_names)
    assert any(name.startswith("student.heads.ranker.") for name in parameter_names)
    assert any(name.startswith("student.heads.covariance_head.") for name in parameter_names)
    assert any(name.startswith("auxiliary_heads.probe.") for name in parameter_names)

    probe = wrapper.auxiliary_heads["probe"](torch.ones(1, 4)).square().sum()
    ranker = wrapper.student.heads.ranker.projection.weight.square().mean()
    (probe + ranker).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    image = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        before_resume = wrapper.student.extract(image, top_k=64)[0]
    saved_model = {name: value.detach().clone() for name, value in wrapper.state_dict().items()}
    checkpoint_path = tmp_path / "step_000001.pt"
    state = {"successful_step": 1, "microstep": 0}
    save_checkpoint(
        checkpoint_path,
        wrapper,
        optimizer,
        state,
        signature,
        config=config,
        identities=identities,
    )

    resumed = _new_training_wrapper()
    resumed_optimizer = build_optimizer(resumed, "raco_multiscale", optimizer_config)
    checkpoint = preflight_run(tmp_path / "resume_run", checkpoint_path, signature)
    assert checkpoint is not None
    assert restore_checkpoint(checkpoint, resumed, resumed_optimizer) == state
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, saved_model[name], atol=0, rtol=0)
    with torch.no_grad():
        after_resume = resumed.student.extract(image, top_k=64)[0]
    for name, value in before_resume.items():
        torch.testing.assert_close(after_resume[name], value, atol=0, rtol=0)

    wrong_signature = checkpoint_signature(config, {"runtime_id": "different-runtime"})
    with pytest.raises(ValueError, match="signature"):
        preflight_run(tmp_path / "wrong_runtime", checkpoint_path, wrong_signature)
    assert not (tmp_path / "wrong_runtime").exists()
    wrong_optimizer_spec = {**optimizer_config, "lr": 5e-4}
    wrong_optimizer = build_optimizer(resumed, "raco_multiscale", wrong_optimizer_spec)
    with pytest.raises(ValueError, match="optimizer signature"):
        restore_checkpoint(checkpoint, resumed, wrong_optimizer)

    bundle = resumed.bundle(trained_heads=["rank", "covariance"])
    assert set(bundle) == {"schema_version", "architecture", "trained_heads", "config", "state_dict"}
    assert not any(key.startswith("auxiliary_heads.") for key in bundle["state_dict"])
    bundle_path = tmp_path / "student_only_bundle.pt"
    torch.save(bundle, bundle_path)
    restored_student = XFeatRaCo.from_bundle(bundle_path, device="cpu")
    with torch.no_grad():
        expected = resumed.student.extract(image, top_k=64)[0]
        actual = restored_student.extract(image, top_k=64)[0]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_two_scale_shapes_mask_and_gradient():
    import pytest

    from xfeat_training.raco_multiscale_losses import TwoScaleFeatureAlignment, two_scale_feature_loss

    torch.manual_seed(23)
    student8 = torch.randn(1, 64, 60, 80, requires_grad=True)
    student32 = torch.randn(1, 64, 15, 20, requires_grad=True)
    teacher8 = torch.randn(1, 64, 60, 80, requires_grad=True)
    teacher32 = torch.randn(1, 128, 15, 20, requires_grad=True)
    valid = torch.ones(1, 1, 480, 640, dtype=torch.bool)
    valid[:, :, :64, :128] = False
    adapter = TwoScaleFeatureAlignment()

    aligned = adapter(student8, student32, valid)
    assert aligned["block3"].shape == teacher8.shape
    assert aligned["block4"].shape == teacher32.shape
    assert aligned["valid8"].shape == (1, 1, 60, 80)
    assert aligned["valid32"].shape == (1, 1, 15, 20)
    losses = two_scale_feature_loss(student8, student32, teacher8, teacher32, valid, adapter)
    assert losses["loss"].ndim == 0
    assert losses["h8_denominator"] > 0 and losses["h32_denominator"] > 0
    assert torch.isfinite(torch.stack((losses["loss"], losses["h8_loss"], losses["h32_loss"]))).all()
    losses["loss"].backward()
    assert student8.grad is not None and torch.isfinite(student8.grad).all()
    assert student32.grad is not None and torch.isfinite(student32.grad).all()
    assert student8.grad.abs().sum() > 0 and student32.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()
    )
    assert teacher8.grad is None and teacher32.grad is None

    student8_changed = student8.detach().clone()
    student32_changed = student32.detach().clone()
    teacher8_changed = teacher8.detach().clone()
    teacher32_changed = teacher32.detach().clone()
    valid8 = torch.nn.functional.adaptive_avg_pool2d(valid.float(), (60, 80)) == 0
    valid32 = torch.nn.functional.adaptive_avg_pool2d(valid.float(), (15, 20)) == 0
    student8_changed.masked_fill_(valid8.expand_as(student8_changed), 1e4)
    teacher8_changed.masked_fill_(valid8.expand_as(teacher8_changed), -1e4)
    student32_changed.masked_fill_(valid32.expand_as(student32_changed), 1e4)
    teacher32_changed.masked_fill_(valid32.expand_as(teacher32_changed), -1e4)
    changed = two_scale_feature_loss(
        student8_changed, student32_changed, teacher8_changed, teacher32_changed, valid, adapter
    )
    torch.testing.assert_close(changed["loss"], losses["loss"].detach(), atol=1e-5, rtol=1e-5)

    empty = torch.zeros_like(valid)
    with pytest.raises(ValueError, match="empty"):
        two_scale_feature_loss(student8.detach(), student32.detach(), teacher8, teacher32, empty, adapter)
    with pytest.raises(ValueError, match="student H/32 features"):
        two_scale_feature_loss(student8.detach(), student32[:, :, :-1], teacher8, teacher32, valid, adapter)
    nonfinite = student8.detach().clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        two_scale_feature_loss(nonfinite, student32.detach(), teacher8, teacher32, valid, adapter)


def test_teacher_dense_feature_hooks_preserve_output_contract():
    import pytest
    from torch import nn

    from xfeat_training.raco_teacher import OfficialRacoTeacher

    class IdentityPadder:
        def __init__(self, *_args, **_kwargs):
            pass

        def unpad(self, value):
            return value

    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.score_head = nn.Conv2d(3, 1, 1)
            self.ranker_head = nn.Conv2d(3, 1, 1)
            self.covariance_estimator_head = nn.Conv2d(3, 3, 1)
            self.block3 = nn.Sequential(nn.AvgPool2d(8), nn.Conv2d(3, 64, 1))
            self.block4 = nn.Sequential(nn.AvgPool2d(32), nn.Conv2d(3, 128, 1))
            self.var_activation = nn.Softplus()

        def forward(self, data):
            image = data["image"]
            self.score_head(image)
            self.ranker_head(image)
            self.covariance_estimator_head(image)
            self.block3(image)
            self.block4(image)
            return {"official": torch.ones(1)}

    teacher = object.__new__(OfficialRacoTeacher)
    teacher.model = FakeTeacher().eval().requires_grad_(False)
    teacher.upstream = type("Upstream", (), {"InputPadder": IdentityPadder})
    image = torch.rand(1, 3, 64, 96, requires_grad=True)

    output_only = teacher.dense(image)
    assert "block3" not in output_only and "block4" not in output_only
    output_with_features = teacher.dense(image, include_features=True)
    assert output_with_features["block3"].shape == (1, 64, 8, 12)
    assert output_with_features["block4"].shape == (1, 128, 2, 3)
    assert not output_with_features["block3"].requires_grad
    assert not output_with_features["block4"].requires_grad
    assert torch.isfinite(output_with_features["cholesky"]).all()

    with pytest.raises(ValueError, match="divisible by 32"):
        teacher.dense(torch.rand(1, 3, 65, 96), include_features=True)


def test_local_contrast_pairs_and_false_negative_mask():
    from modules.raco import XFeatRaCo
    from xfeat_training.raco_multiscale_losses import (
        LocalContrastProjection,
        build_mutual_nearest_pairs,
        local_false_negative_mask,
        symmetric_local_info_nce,
    )
    from xfeat_training.raco_multiscale_task import RacoMultiscaleTrainingModel

    torch.manual_seed(20260928)
    points_a = torch.tensor([[24.0 + 56 * col, 24.0 + 68 * row] for row in range(5) for col in range(8)])
    points_a[1] = torch.tensor([32.0, 24.0])
    homography = torch.tensor([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    points_b = points_a + torch.tensor([5.0, 3.0])
    matches = build_mutual_nearest_pairs(
        points_a, points_b, homography, (480, 640), max_distance_px=2.0, max_pairs=256, seed=19
    )
    assert len(matches["source_indices"]) == 40
    assert torch.equal(matches["source_indices"], torch.arange(40))
    assert torch.equal(matches["target_indices"], torch.arange(40))
    assert matches["distances"].max() < 1e-5

    dense_points = torch.tensor([[20.0 + 20 * col, 20.0 + 28 * row] for row in range(15) for col in range(20)])
    dense_targets = dense_points + torch.tensor([5.0, 3.0])
    sampled_a = build_mutual_nearest_pairs(
        dense_points, dense_targets, homography, (480, 640), max_distance_px=2.0, max_pairs=256, seed=19
    )
    sampled_b = build_mutual_nearest_pairs(
        dense_points, dense_targets, homography, (480, 640), max_distance_px=2.0, max_pairs=256, seed=19
    )
    assert len(sampled_a["source_indices"]) == 256
    assert torch.equal(sampled_a["source_indices"], sampled_b["source_indices"])
    assert torch.equal(sampled_a["target_indices"], sampled_b["target_indices"])

    ambiguous = build_mutual_nearest_pairs(
        points_a[:1],
        torch.cat((points_b[:1], points_b[:1]), dim=0),
        homography,
        (480, 640),
        max_distance_px=2.0,
    )
    assert len(ambiguous["source_indices"]) == 0

    paired_a = points_a[matches["source_indices"]]
    paired_b = points_b[matches["target_indices"]]
    excluded = local_false_negative_mask(paired_a, paired_b, exclusion_px=16.0)
    assert excluded.shape == (40, 40)
    assert excluded[0, 1] and excluded[1, 0]
    assert not excluded[0, 2] and not excluded[2, 0]
    assert not excluded.diagonal().any()

    projector = LocalContrastProjection(embedding_dim=32)
    feature_a = torch.randn(1, 64, 60, 80, requires_grad=True)
    feature_b = torch.randn(1, 64, 60, 80, requires_grad=True)
    embedding_a = projector(feature_a, paired_a, image_size=(480, 640))
    embedding_b = projector(feature_b, paired_b, image_size=(480, 640))
    result = symmetric_local_info_nce(embedding_a, embedding_b, paired_a, paired_b, temperature=0.1)
    assert result["active"] and result["pair_count"] == 40
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert feature_a.grad is not None and torch.isfinite(feature_a.grad).all()
    assert feature_b.grad is not None and torch.isfinite(feature_b.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in projector.parameters()
    )

    skipped = symmetric_local_info_nce(
        embedding_a[:31], embedding_b[:31], paired_a[:31], paired_b[:31], temperature=0.1
    )
    assert not skipped["active"] and skipped["pair_count"] == 31
    assert skipped["skip_reason"] == "fewer_than_min_pairs"
    assert skipped["loss"].item() == 0.0

    cluster_points = torch.tensor([[24.0 + col, 24.0 + row] for row in range(4) for col in range(8)])
    cluster_embeddings = torch.randn(32, 32)
    no_negative = symmetric_local_info_nce(
        cluster_embeddings, cluster_embeddings, cluster_points, cluster_points, temperature=0.1
    )
    assert not no_negative["active"] and no_negative["valid_queries"] == 0
    assert no_negative["invalid_queries"] == 32
    assert no_negative["skip_reason"] == "no_valid_negative_queries"

    mixed_points = torch.tensor([[10.0, 10.0]] * 11 + [[20.0, 10.0]] * 10 + [[30.0, 10.0]] * 11)
    mixed_embeddings = torch.randn(32, 32)
    mixed = symmetric_local_info_nce(mixed_embeddings, mixed_embeddings, mixed_points, mixed_points, temperature=0.1)
    assert mixed["active"] and mixed["valid_queries"] == 22 and mixed["invalid_queries"] == 10

    student = XFeatRaCo(weights=None, device="cpu", candidate_limit=32)
    local_only = RacoMultiscaleTrainingModel(student, {"local_contrast": LocalContrastProjection(embedding_dim=32)})
    trainable_names = local_only.configure_multiscale_training()
    assert any(name.startswith("auxiliary_heads.local_contrast.") for name in trainable_names)
    assert not any("feature_alignment" in name for name in trainable_names)


def test_ablation_config_identity_and_selection(tmp_path):
    import json

    import numpy as np
    import pytest
    from omegaconf import OmegaConf
    from torch import nn

    from xfeat_training.data import file_sha256
    from xfeat_training.optim import build_optimizer
    from xfeat_training.raco_multiscale_task import calibrate_gradient_weights, calibrate_multiscale_weights
    from xfeat_training.raco_multiscale_train import (
        build_variant_config,
        multiscale_selection_score,
        validate_multiscale_config,
    )
    from xfeat_training.retention import prune_checkpoints
    from xfeat_training.trainer import (
        PairSampler,
        atomic_json,
        checkpoint_signature,
        pair_index_digest,
        save_checkpoint,
    )

    common = OmegaConf.to_container(OmegaConf.load("configs/raco.yaml"), resolve=True, throw_on_missing=False)
    assert isinstance(common, dict)
    common.pop("defaults", None)
    common.pop("hydra", None)
    template = OmegaConf.load("configs/raco_multiscale.yaml")
    assert template.val_frames == 16
    common["multiscale"] = OmegaConf.to_container(template.multiscale, resolve=True)
    common["seed"] = template.seed
    common["val_frames"] = template.val_frames
    common["image_size"] = list(template.image_size)
    common["deterministic_warn_only"] = template.deterministic_warn_only
    common.update(
        run_dir="placeholder-run",
        pairs_dir="placeholder-pairs",
        init_bundle="placeholder-student.pt",
        max_steps=5000,
        stop_after_steps=1000,
        checkpoint_keep_best=3,
        auto_stop={**common["auto_stop"], "enabled": False},
    )
    variants = {name: build_variant_config(common, name) for name in ("A", "B", "C")}
    for config in variants.values():
        validate_multiscale_config(config)
        assert config["seed"] == 20260928
        assert config["val_frames"] == 16
        assert config["image_size"] == [480, 640]
        assert config["deterministic_warn_only"] is True
        assert config["max_steps"] == 5000 and config["stop_after_steps"] == 1000
        assert config["checkpoint_keep_best"] == 3 and not config["auto_stop"]["enabled"]
        assert config["init_bundle"] == common["init_bundle"]
        assert config["optimizer"] == common["optimizer"]
    assert not variants["A"]["multiscale"]["feature_alignment"]["enabled"]
    assert variants["B"]["multiscale"]["feature_alignment"]["enabled"]
    assert variants["C"]["multiscale"]["local_contrast"]["enabled"]
    assert not variants["A"]["multiscale"]["local_contrast"]["enabled"]
    for config in variants.values():
        assert config["multiscale"]["rank_kd"]["enabled"]
        assert not config["multiscale"]["covariance_kd"]["enabled"]
    invalid_val_count = {**variants["A"], "val_frames": 4}
    with pytest.raises(ValueError, match="16-frame fixed validation"):
        validate_multiscale_config(invalid_val_count)
    with pytest.raises(ValueError, match="baseline 480x640 image size"):
        validate_multiscale_config({**variants["A"], "image_size": None})
    with pytest.raises(ValueError, match="deterministic warn-only"):
        validate_multiscale_config({**variants["A"], "deterministic_warn_only": False})
    signatures = {checkpoint_signature(config, {"pair_hash": "fixed-pairs"}) for config in variants.values()}
    assert len(signatures) == 3, "A/B/C must not accept each other's strict-resume checkpoints"
    assert pair_index_digest([2, 7, 2]) == pair_index_digest([2, 7, 2])
    assert pair_index_digest([2, 7, 2]) != pair_index_digest([7, 2, 2])
    sampler_pairs = {
        "subset": np.array(["left", "right"] * 4),
        "difficulty": np.array(["overlap_0"] * 8),
        "overlap": np.full(8, 0.9, dtype=np.float32),
    }
    sampler_a = PairSampler(sampler_pairs, {"bin_weights": [1.0]}, 20260928)
    sampler_b = PairSampler(sampler_pairs, {"bin_weights": [1.0]}, 20260928)
    assert [sampler_a.sample(step) for step in range(12)] == [sampler_b.sample(step) for step in range(12)]

    baseline = {"rank_utility": 0.3264389, "covariance_nll": 3.49498697}
    improved = {"rank_utility": 0.34, "covariance_nll": 3.40}
    expected_q = min(
        (improved["rank_utility"] - baseline["rank_utility"]) / baseline["rank_utility"],
        (baseline["covariance_nll"] - improved["covariance_nll"]) / baseline["covariance_nll"],
    )
    assert multiscale_selection_score(improved, baseline) == pytest.approx(expected_q)
    with pytest.raises(ValueError, match="positive"):
        multiscale_selection_score({"rank_utility": 0.0, "covariance_nll": 3.4}, baseline)

    shared_parameter = nn.Parameter(torch.tensor(2.0))
    provider_calls = []

    def calibration_losses(index):
        provider_calls.append(index)
        offset = index / 100.0
        value = shared_parameter
        return {
            "rank_geometry": (value - 1.0 - offset).square(),
            "covariance_geometry": 3.0 * (value + 0.5 + offset).square(),
            "descriptor": 2.0 * (value - 0.5 - offset).square(),
            "rank_kd": 0.5 * (value + 0.25 + offset).square(),
            "feature": 8.0 * (value - 0.25 - offset).square(),
            "local": 4.0 * (value + 0.75 + offset).square(),
        }

    calibration = calibrate_gradient_weights(
        calibration_losses,
        {"shared_stage.weight": shared_parameter},
        batch_count=8,
        covariance_bounds=(0.1, 10.0),
        variant_aux_key="feature",
    )
    assert provider_calls == list(range(8)) * 3
    assert calibration["common_parameter_names"] == ["shared_stage.weight"]
    assert 0.1 <= calibration["lambda_cov"] <= 10.0
    assert 0.0 < calibration["lambda_shared_aux"] <= 1.0
    assert 0.0 < calibration["lambda_variant_aux"] <= 1.0
    assert len(calibration["rank_gradient_norms"]) == len(calibration["task_gradient_norms"]) == 8

    common_calibration = calibrate_gradient_weights(
        calibration_losses, {"shared_stage.weight": shared_parameter}, batch_count=8
    )
    calibrated_weights, calibration_records = calibrate_multiscale_weights(
        calibration_losses,
        {"shared_stage.weight": shared_parameter},
        {"feature": "lambda_feature", "local": "lambda_local"},
        batch_count=8,
    )
    assert calibrated_weights["lambda_cov"] == common_calibration["lambda_cov"]
    assert calibrated_weights["lambda_shared_aux"] == common_calibration["lambda_shared_aux"]
    assert 0.0 < calibrated_weights["lambda_feature"] <= 1.0
    assert 0.0 < calibrated_weights["lambda_local"] <= 1.0
    assert set(calibration_records) == {"shared", "feature", "local"}

    run_dir = tmp_path / "retention"
    model = nn.Linear(4, 4)
    optimizer_spec = {"name": "adamw", "lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    checkpoint_config = {
        "optimizer": optimizer_spec,
        "max_steps": 100,
        "eval_every": 10,
        "save_every": 10,
        "selection_metric": {"name": "multiscale_q", "mode": "max"},
    }
    identities = {"evaluation_hash": "fixed-eval", "pair_hash": "fixed-pairs"}
    signature = checkpoint_signature(checkpoint_config, identities)
    optimizer = build_optimizer(model, "xfeat", optimizer_spec)
    score_by_step = {10: -0.3, 20: 0.1, 30: 0.0, 40: 0.2, 50: -0.1, 60: 0.3}
    rows = []
    for step in (10, 20, 30, 40, 50, 60, 61):
        save_checkpoint(
            run_dir / "checkpoints" / f"step_{step:06d}.pt",
            model,
            optimizer,
            {"successful_step": step, "microstep": 0},
            signature,
            config=checkpoint_config,
            identities=identities,
        )
        metric = None
        if step in score_by_step:
            metric = {
                "split": "val",
                "evaluation_hash": identities["evaluation_hash"],
                "pair_hash": identities["pair_hash"],
                "multiscale_q": score_by_step[step],
            }
            atomic_json(run_dir / "validation" / f"step_{step:06d}" / "metrics.json", metric)
        rows.append({"step": step, "validation": metric})
    atomic_json(
        run_dir / "run.json",
        {"last_checkpoint": "checkpoints/step_000061.pt", "completed_steps": 61, "signature": signature},
    )
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    retained = prune_checkpoints(run_dir, keep_best=3)
    assert retained["top_k"] == ["step_000060.pt", "step_000040.pt", "step_000020.pt"]
    assert set(retained["kept"]) == {"step_000020.pt", "step_000040.pt", "step_000060.pt", "step_000061.pt"}
    assert len(list((run_dir / "checkpoints").glob("*.pt"))) == 4
    assert all(
        file_sha256(run_dir / "checkpoints" / name) == entry["file_sha256"] for name, entry in retained["kept"].items()
    )
