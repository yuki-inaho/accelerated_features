"""Sparse RaCo-style heads: coordinates, identities, gradients and compatibility."""

import pytest
import torch
import torch.nn.functional as F

from modules.model import XFeatModel


def test_feature_tap_preserves_forward_and_checkpoint_keys():
    model = XFeatModel().eval()
    keys = list(model.state_dict())
    image = torch.rand(1, 3, 64, 96)
    with torch.no_grad():
        before = model(image)
        features = model.forward_with_features(image)
    assert len(before) == 3 and len(features) == 4
    for old, new in zip(before, features[:3], strict=True):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    assert features[3].shape == (1, 64, 8, 12)
    assert list(model.state_dict()) == keys


@pytest.mark.parametrize("channels", [1, 3])
def test_sparse_head_equals_pixel_shuffle(channels):
    from modules.raco import PixelHead

    head = PixelHead(channels).double()
    z = torch.randn(1, 128, 4, 5, dtype=torch.float64)
    points = torch.tensor([[0, 0], [39, 31], [8, 15], [9, 2]])
    dense = F.pixel_shuffle(head.projection(head.trunk(z)), 8)
    sparse = head(z, points)
    torch.testing.assert_close(sparse, dense[0, :, points[:, 1], points[:, 0]].T)
    assert head(z, points[:0]).shape == (0, channels)


def test_heads_parameter_count_spd_and_independent_gradients():
    from modules.raco import RacoHeads

    heads = RacoHeads().double()
    assert sum(p.numel() for p in heads.parameters()) == 34688
    z = torch.randn(1, 128, 4, 4, dtype=torch.float64)
    points = torch.tensor([[8, 9], [16, 16], [22, 25]])
    p = torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)
    ranks = heads.rank(z, points, p)
    ranks.square().sum().backward()
    assert all(p.grad is not None for p in heads.ranker.parameters())
    assert all(p.grad is None for p in heads.covariance_head.parameters())
    heads.zero_grad(set_to_none=True)
    cov = heads.covariance(z, points)
    torch.testing.assert_close(cov, cov.mT)
    assert torch.linalg.eigvalsh(cov).min() >= 0.05**2
    torch.testing.assert_close(cov, torch.eye(2, dtype=cov.dtype).expand(3, 2, 2), atol=0.02, rtol=0.02)
    cov.sum().backward()
    assert all(p.grad is not None for p in heads.covariance_head.parameters())
    assert all(p.grad is None for p in heads.ranker.parameters())


def test_resize_pixel_centers_and_covariance_restore():
    from modules.raco import resize_input, restore_coordinates

    image, transform = resize_input(torch.zeros(1, 3, 33, 65))
    assert image.shape[-2:] == (64, 96)
    points = torch.tensor([[0.0, 0.0], [64.0, 32.0]])
    net = points @ transform[:2, :2].T + transform[:2, 2]
    cov = torch.eye(2).expand(2, 2, 2).clone()
    restored, restored_cov = restore_coordinates(net, cov, transform)
    torch.testing.assert_close(restored, points, atol=1e-5, rtol=1e-5)
    inverse = torch.linalg.inv(transform[:2, :2])
    torch.testing.assert_close(restored_cov, (inverse @ inverse.T).expand(2, 2, 2))
    scale = torch.diag(torch.tensor([0.5, 0.5, 1.0]))
    _, four = restore_coordinates(points, cov, scale)
    torch.testing.assert_close(four, 4 * cov)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    asymmetric = torch.tensor([[[4.0, 1.0], [1.0, 2.0]]])
    _, rotated = restore_coordinates(points[:1], asymmetric, rotation)
    torch.testing.assert_close(rotated, rotation[:2, :2].T @ asymmetric @ rotation[:2, :2])


def test_frozen_extractor_and_aligned_candidate_attributes():
    from modules.raco import XFeatRaCo

    model = XFeatRaCo(device="cpu", candidate_limit=64, detection_threshold=0.001)
    before = {k: v.clone() for k, v in model.net.state_dict().items()}
    model.train()
    assert not model.net.training
    candidates = model.candidates(torch.rand(1, 3, 64, 96))[0]
    assert len(candidates["keypoints"]) > 0
    original = model.predict(candidates, top_k=1000, ranking=False, covariance=False)
    ranked = model.predict(candidates, top_k=1000, ranking=True, covariance=True)
    ids = ranked["candidate_ids"]
    for key in ("keypoints", "descriptors", "scores"):
        torch.testing.assert_close(ranked[key], original[key][ids])
    assert torch.equal(ids.sort().values, original["candidate_ids"])
    assert torch.all(ranked["ranker_scores"][:-1] >= ranked["ranker_scores"][1:])
    ranked["ranker_scores"].square().sum().backward()
    assert all(p.grad is None for p in model.net.parameters())
    for key, value in model.net.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_empty_candidates_negative_ranks_and_nonfinite_rejection():
    from modules.raco import XFeatRaCo

    model = XFeatRaCo(device="cpu", detection_threshold=1.0)
    result = model.extract(torch.rand(2, 3, 65, 97), top_k=10)
    assert len(result) == 2
    for out in result:
        assert out["covariances"].shape == (0, 2, 2)
        assert out["descriptors"].shape == (0, 64)
    with pytest.raises(ValueError, match="finite"):
        model.extract(torch.full((1, 3, 64, 64), float("nan")))
    model.detection_threshold = 0.001
    with torch.no_grad():
        model.heads.ranker.projection.weight.zero_()
        assert model.heads.ranker.projection.bias is not None
        model.heads.ranker.projection.bias.fill_(-10)
    output = model.extract(torch.rand(1, 3, 64, 64), top_k=10)[0]
    assert len(output["keypoints"]) > 0
    assert (output["ranker_scores"] < 0).all()


def test_bundle_roundtrip_strict_and_untrained_head_guard(tmp_path):
    from modules.raco import XFeatRaCo

    model = XFeatRaCo(device="cpu", candidate_limit=64)
    bundle = model.bundle(trained_heads=["rank", "covariance"])
    path = tmp_path / "raco.pt"
    torch.save(bundle, path)
    restored = XFeatRaCo.from_bundle(path, device="cpu")
    image = torch.rand(1, 3, 64, 96)
    for key, value in model.extract(image)[0].items():
        torch.testing.assert_close(value, restored.extract(image)[0][key], atol=0, rtol=0)
    bundle["trained_heads"] = ["rank"]
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="untrained"):
        XFeatRaCo.from_bundle(path, device="cpu")
    XFeatRaCo.from_bundle(path, device="cpu", covariance=False)
    bundle["state_dict"].pop(next(iter(bundle["state_dict"])))
    torch.save(bundle, path)
    with pytest.raises(RuntimeError, match="Missing"):
        XFeatRaCo.from_bundle(path, device="cpu", covariance=False)


def test_non_strict_transfer_initializes_only_missing_heads():
    from modules.raco import XFeatRaCo

    base = XFeatModel().eval().state_dict()
    combined = {"extractor.model.net." + k: v.clone() for k, v in base.items()}
    combined["matcher.unused.weight"] = torch.ones(2, 2)
    model = XFeatRaCo(weights=combined)
    report = model.warm_start_report
    assert report["strict"] is False and len(report["loaded_keys"]) == len(base)
    assert report["missing_keys"] and all(k.startswith("heads.") for k in report["missing_keys"])
    assert report["unexpected_keys"] == ["matcher.unused.weight"]
    for key, value in base.items():
        torch.testing.assert_close(model.net.state_dict()[key], value, atol=0, rtol=0)
    combined.pop("extractor.model.net." + next(iter(base)))
    with pytest.raises(ValueError, match="missing frozen"):
        XFeatRaCo(weights=combined)
