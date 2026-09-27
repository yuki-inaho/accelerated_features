"""Known-value and numerical-gradient contracts for ranking and error matrices."""

import math

import pytest
import torch


def test_soft_topk_mass_shift_and_implicit_gradient():
    from xfeat_training.raco_objectives import soft_topk

    scores = torch.randn(11, dtype=torch.float64, requires_grad=True)
    alpha = soft_topk(scores, 4, 0.2)
    torch.testing.assert_close(alpha.sum(), torch.tensor(4.0, dtype=scores.dtype))
    torch.testing.assert_close(alpha, soft_topk(scores + 100, 4, 0.2))
    assert torch.autograd.gradcheck(lambda x: soft_topk(x, 4, 0.2), (scores,))
    assert torch.equal(soft_topk(scores, 11, 0.2), torch.ones_like(scores))
    assert torch.equal(soft_topk(scores, 0, 0.2), torch.zeros_like(scores))
    assert soft_topk(scores[:0], 4, 0.2).numel() == 0
    gradient = torch.autograd.grad((alpha * torch.arange(11)).sum(), scores)[0]
    assert gradient.norm() > 0
    assert abs(float(gradient.sum())) < 1e-10


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_objective_rejects_nonfinite(bad):
    from xfeat_training.raco_objectives import soft_topk

    with pytest.raises(ValueError, match="finite"):
        soft_topk(torch.tensor([0.0, bad]), 1, 0.5)


def test_budget_ranking_uses_all_candidates_and_preserves_bounds():
    from xfeat_training.raco_objectives import hard_utility, ranking_loss

    matches = torch.tensor([[0, 0], [1, 1]])
    good = torch.tensor([3.0, 2.0, -3.0, -2.0], requires_grad=True)
    bad = -good
    assert hard_utility(good, good, matches, [1, 2]) == 1.0
    assert hard_utility(bad, bad, matches, [1, 2]) == 0.0
    loss = ranking_loss(good, good, matches, [1, 2], 0.2)
    assert 0 <= loss <= 1
    assert loss < ranking_loss(bad, bad, matches, [1, 2], 0.2)
    loss.backward()
    assert good.grad is not None and good.grad.norm() > 0
    # Large budgets use all available candidates but the requested k denominator.
    torch.testing.assert_close(ranking_loss(good, good, matches, [8], 0.2), torch.tensor(0.75))


def test_homography_jacobian_and_bidirectional_mnn():
    from xfeat_training.raco_objectives import geometric_matches, project_with_jacobian

    points = torch.tensor([[5.0, 9.0], [20.0, 8.0], [30.0, 25.0]], dtype=torch.float64)
    h = torch.tensor([[1.1, 0.1, 3.0], [-0.1, 0.9, 2.0], [0.001, -0.002, 1.0]], dtype=torch.float64)
    warped, jac = project_with_jacobian(points, h)
    for i in range(len(points)):
        numerical = torch.autograd.functional.jacobian(lambda p: project_with_jacobian(p[None], h)[0][0], points[i])
        torch.testing.assert_close(jac[i], numerical)
    b = torch.cat((warped, torch.tensor([[100.0, 100.0]], dtype=torch.float64)))
    matches = geometric_matches(points, b, h, threshold=0.01, block_size=1)
    assert torch.equal(matches, torch.tensor([[0, 0], [1, 1], [2, 2]]))
    assert geometric_matches(points[:0], b, h, 3).shape == (0, 2)
    with pytest.raises(ValueError, match="homography"):
        project_with_jacobian(points, torch.zeros(3, 3, dtype=points.dtype))


def test_symmetric_covariance_nll_known_value_and_gradients():
    from xfeat_training.raco_objectives import covariance_nll

    a = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float64)
    b = a + torch.tensor([1.0, 0.0], dtype=torch.float64)
    cov0 = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    cov1 = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    matches = torch.tensor([[0, 0], [1, 1]])
    h = torch.eye(3, dtype=torch.float64)
    loss, stats = covariance_nll(a, b, cov0, cov1, matches, h)
    assert float(loss.detach()) == pytest.approx(math.log(2) + 0.25 + math.log(2 * math.pi))
    assert stats["mahalanobis"] == pytest.approx(0.5)
    assert stats["coverage95"] == 1.0
    loss.backward()
    assert cov0.grad is not None and cov0.grad.norm() > 0
    assert cov1.grad is not None and cov1.grad.norm() > 0
    reverse, _ = covariance_nll(b, a, cov1, cov0, matches, h)
    torch.testing.assert_close(reverse, loss)


def test_covariance_nll_scale_and_rotation_invariance():
    from xfeat_training.raco_objectives import covariance_nll

    a = torch.tensor([[5.0, 9.0], [18.0, 27.0]])
    b = a + 0.5
    cov = torch.tensor([[[2.0, 0.3], [0.3, 1.0]]]).repeat(2, 1, 1)
    matches = torch.tensor([[0, 0], [1, 1]])
    h = torch.eye(3)
    base, _ = covariance_nll(a, b, cov, cov, matches, h)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotated, _ = covariance_nll(
        a @ rotation.T, b @ rotation.T, rotation @ cov @ rotation.T, rotation @ cov @ rotation.T, matches, h
    )
    torch.testing.assert_close(base, rotated)
    scaled, _ = covariance_nll(a * 2, b * 2, cov * 4, cov * 4, matches, h)
    torch.testing.assert_close(scaled, base + 2 * math.log(2))
