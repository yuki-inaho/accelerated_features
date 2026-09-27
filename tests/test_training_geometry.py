"""Hand-derived planes and translations, independent of the implementation."""

from __future__ import annotations

import numpy as np
import pytest

from xfeat_training.geometry import matching_labels, overlap, project_points, relative_motion


def test_projection_world_to_camera_and_axis(plane_frame) -> None:
    source, target = plane_frame(), plane_frame(tx=0.1)
    points = np.array([[10, 20], [20, 30]], dtype=float)
    result = project_points(points, source, target)
    np.testing.assert_allclose(result.xy, [[20, 20], [30, 30]], atol=1e-5)
    np.testing.assert_allclose(result.z, [1, 1])
    assert result.valid.all()
    reverse = project_points(result.xy, target, source)
    np.testing.assert_allclose(reverse.xy, points, atol=1e-5)
    rotation, distance = relative_motion(source.w2c, target.w2c)
    assert rotation == pytest.approx(0) and distance == pytest.approx(0.1)


def test_identity_and_stride_overlap(plane_frame) -> None:
    source, target = plane_frame(), plane_frame(tx=0.08)
    assert overlap(source, source) == 1
    # x=0,8,...72: one of ten columns leaves the image in each direction.
    assert overlap(source, target, stride=8) == pytest.approx(0.9)
    source.depth[:] = 0
    assert overlap(source, target) == 0


def test_symmetric_positive_and_zero_index(plane_frame) -> None:
    a, b = plane_frame(), plane_frame(tx=0.1)
    labels = matching_labels(np.array([[10.0, 10], [30, 30]]), np.array([[20.0, 10], [40, 30]]), a, b)
    np.testing.assert_array_equal(labels.matches0, [0, 1])
    np.testing.assert_array_equal(labels.matches1, [0, 1])


@pytest.mark.parametrize("offset,expected", [(2.9, 0), (3.1, -2), (4.99, -2), (5.0, -1)])
def test_pixel_threshold_boundaries(plane_frame, offset: float, expected: int) -> None:
    a, b = plane_frame(), plane_frame()
    labels = matching_labels(np.array([[20.0, 20]]), np.array([[20 + offset, 20.0]]), a, b)
    assert labels.matches0.tolist() == [expected]
    assert labels.matches1.tolist() == [expected]


@pytest.mark.parametrize(
    "case,expected",
    [("occluded", -1), ("inconsistent", -2), ("unknown", -2), ("source_invalid", -2), ("edge", -2), ("outside", -1)],
)
def test_visibility_distinguishes_unknown_and_negative(plane_frame, case: str, expected: int) -> None:
    a, b = plane_frame(), plane_frame()
    p = np.array([[20.0, 20]])
    if case == "occluded":
        b.w2c[2, 3] = 0.1
    elif case == "inconsistent":
        b.w2c[2, 3] = -0.1
    elif case == "unknown":
        b.depth[20, 20] = 0
    elif case == "source_invalid":
        a.depth[20, 20] = 0
    elif case == "edge":
        b.depth[20, 21] = 1.1
    else:
        b.w2c[0, 3] = -1
    result = matching_labels(p, p, a, b)
    assert result.matches0.tolist() == [expected]


def test_mutual_collision_and_invalid_target_is_not_removed(plane_frame) -> None:
    a, b = plane_frame(), plane_frame()
    labels = matching_labels(np.array([[20.0, 20], [20.5, 20]]), np.array([[20.0, 20]]), a, b)
    assert labels.matches0.tolist() == [0, -2]
    assert labels.matches1.tolist() == [0]
    # Projected point at (20,20) is valid; real target token (23,20) has invalid depth.
    b.depth[20, 25] = 0
    labels = matching_labels(np.array([[20.0, 20]]), np.array([[23.0, 20]]), a, b)
    assert labels.matches0.tolist() == [-2]


def test_border_and_empty_points(plane_frame) -> None:
    a = plane_frame()
    labels = matching_labels(np.array([[1.0, 20]]), np.array([[1.0, 20]]), a, a)
    assert labels.matches0.tolist() == [-2]
    labels = matching_labels(np.empty((0, 2)), np.empty((0, 2)), a, a)
    assert labels.matches0.shape == labels.matches1.shape == (0,)


def test_positive_nearest_uses_symmetric_matrix(plane_frame, monkeypatch):
    from xfeat_training import geometry

    points = np.array([[0.0, 0.0], [3.0, 0.0]])
    outputs = iter(
        [
            (
                geometry.Projection(np.array([[1.0, 0.0], [2.0, 0.0]]), np.ones(2), np.ones(2, bool)),
                np.ones(2, bool),
                np.full(2, -2, np.int64),
                np.full(2, "ambiguous", dtype="U32"),
            ),
            (
                geometry.Projection(np.array([[2.0, 0.0], [1.0, 0.0]]), np.ones(2), np.ones(2, bool)),
                np.ones(2, bool),
                np.full(2, -2, np.int64),
                np.full(2, "ambiguous", dtype="U32"),
            ),
        ]
    )
    monkeypatch.setattr(geometry, "_visibility", lambda *args: next(outputs))
    labels = geometry.matching_labels(points, points, plane_frame(), plane_frame())
    # S=max(D01,D10.T) is all 2: index tie break keeps only 0<->0.
    np.testing.assert_array_equal(labels.matches0, [0, -2])
    np.testing.assert_array_equal(labels.matches1, [0, -2])
