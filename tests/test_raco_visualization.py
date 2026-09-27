import math

import numpy as np
import torch

from modules.raco import XFeatRaCo
from scripts.raco_visualization import (
    advance_tracks,
    cosine_mnn,
    dense_fields,
    ellipse_parameters,
    pose_matrix,
    project_points,
)


def test_pose_and_known_depth_projection():
    pose = pose_matrix([1, 2, 3, 0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    np.testing.assert_allclose(pose @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)
    k = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    depth = np.full((12, 12), 2.0)
    points = np.array([[5.0, 5], [6, 5]])
    transform = np.eye(4)
    result, valid = project_points(points, depth, depth, transform, k)
    np.testing.assert_allclose(result, points)
    assert valid.all()
    transform[0, 3] = 0.2
    result, valid = project_points(points, depth, depth, transform, k)
    np.testing.assert_allclose(result, points + np.array([1, 0]))
    assert valid.all()
    assert not project_points(points, depth, depth / 2, transform, k)[1].any()
    assert not project_points(points, depth * 0, depth, transform, k)[1].any()


def test_ellipse_units_and_scale():
    width, height, angle = ellipse_parameters(np.diag([4.0, 1.0]), scale=4)
    np.testing.assert_allclose([width, height], 8 * np.sqrt(5.991 * np.array([4, 1])))
    assert abs(angle) % 180 == 0


def test_mnn_and_track_termination():
    a = np.eye(3)
    matches = cosine_mnn(a, a[[1, 0, 2]])
    np.testing.assert_array_equal(matches, [[0, 1], [1, 0], [2, 2]])
    ids = advance_tracks(np.array([0, 1, 2]), 3, np.array([[0, 1], [2, 0]]))
    np.testing.assert_array_equal(ids, [2, 0, -1])
    ids = advance_tracks(ids, 3, np.array([[0, 0], [2, 1]]))
    np.testing.assert_array_equal(ids, [2, -1, -1])
    assert cosine_mnn(a[:0], a).shape == (0, 2)


def test_dense_fields_equal_sparse_head_values():
    torch.manual_seed(13)
    model = XFeatRaCo(weights=None, device="cpu", detection_threshold=0)
    image = torch.rand(1, 3, 64, 96)
    fields = dense_fields(model, image)
    candidate = model.candidates(image)[0]
    points = candidate["keypoints"]
    x, y = points.T
    with torch.no_grad():
        correction = 2 * model.heads.ranker(candidate["z"], points).squeeze(-1).tanh()
        covariance = model.heads.covariance(candidate["z"], points)
    np.testing.assert_allclose(fields["rank_correction"][y, x], correction.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(fields["covariance"][y, x], covariance.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(fields["detector"][y, x], candidate["scores"].numpy(), atol=1e-6)
