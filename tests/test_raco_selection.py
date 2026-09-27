import pytest


def test_explicit_selection_direction_ties_and_legacy():
    from xfeat_training.selection import checkpoint_candidate, selection_rank

    legacy = checkpoint_candidate({"primary_f1": 0.7, "TP": 3}, 4, {})
    assert legacy == {"primary_f1": 0.7, "TP": 3, "step": 4}
    assert selection_rank(legacy) == (0.7, 3, -4)
    config = {"selection_metric": {"name": "covariance_nll", "mode": "min"}}
    a = checkpoint_candidate({"covariance_nll": -1.0}, 3, config)
    b = checkpoint_candidate({"covariance_nll": 2.0}, 2, config)
    assert selection_rank(a) > selection_rank(b)
    c = checkpoint_candidate({"covariance_nll": -1.0}, 4, config)
    assert selection_rank(a) > selection_rank(c)
    with pytest.raises(ValueError, match="finite"):
        checkpoint_candidate({"covariance_nll": float("nan")}, 5, config)
    with pytest.raises(ValueError, match="selection"):
        checkpoint_candidate(
            {"covariance_nll": 1.0}, 5, {"selection_metric": {"name": "covariance_nll", "mode": "magic"}}
        )
