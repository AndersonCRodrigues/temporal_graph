"""Testes de rotação e projeção."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.projection import (
    PLANES,
    PROJECTION_MODES,
    project_points,
    rotation_matrix_4d,
)


def test_drop_each_axis():
    pts = np.array([[1.0, 2, 3, 4]])
    expected = {"drop_x": [2, 3, 4], "drop_y": [1, 3, 4], "drop_z": [1, 2, 4], "drop_w": [1, 2, 3]}
    for mode, out in expected.items():
        np.testing.assert_array_equal(project_points(pts, mode=mode), [out])


def test_perspective_magnifies_points_closer_to_camera():
    pts = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 1]])
    out = project_points(pts, mode="perspective", camera_w=3.0)
    assert out.shape == (2, 3)
    assert out[1, 0] > out[0, 0]


def test_perspective_survives_points_beyond_camera():
    out = project_points(np.array([[1.0, 1, 1, 50]]), mode="perspective", camera_w=3.0)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("plane", PLANES)
def test_rotations_are_orthogonal(plane):
    r = rotation_matrix_4d(plane, 0.7)
    np.testing.assert_allclose(r @ r.T, np.eye(4), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_rotation_xw_quarter_turn():
    out = rotation_matrix_4d("xw", np.pi / 2) @ np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(out, [0, 0, 0, 1], atol=1e-10)


def test_rotation_then_project():
    out = project_points(np.array([[1.0, 0, 0, 0]]), "drop_w", rotations=[("xw", np.pi / 2)])
    np.testing.assert_allclose(out, [[0, 0, 0]], atol=1e-10)


def test_rotations_apply_in_order():
    p = np.array([[1.0, 0, 0, 0]])
    a = project_points(p, "drop_y", rotations=[("xw", np.pi / 2), ("zw", np.pi / 2)])
    b = project_points(p, "drop_y", rotations=[("zw", np.pi / 2), ("xw", np.pi / 2)])
    assert not np.allclose(a, b)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        rotation_matrix_4d("ab", 1.0)
    with pytest.raises(ValueError):
        project_points(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        project_points(np.zeros((2, 4)), mode="nope")


def test_all_modes_return_n_by_3():
    pts = np.random.default_rng(0).normal(size=(7, 4))
    for mode in PROJECTION_MODES:
        assert project_points(pts, mode=mode).shape == (7, 3)
