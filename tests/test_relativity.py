"""Testes de boosts de Lorentz, intervalo de Minkowski e tempo-luz."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.model import Graph4D
from temporal_graph.projection import project_points
from temporal_graph.relativity import (
    classify_interval,
    light_time,
    lorentz_boost_matrix,
    minkowski_interval,
    retarded_time,
    to_ct,
)

ETA = np.diag([1.0, 1.0, 1.0, -1.0])


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("beta", [-0.9, -0.3, 0.0, 0.5, 0.99])
def test_boost_preserves_the_minkowski_metric(axis, beta):
    b = lorentz_boost_matrix(axis, beta)
    assert np.allclose(b.T @ ETA @ b, ETA, atol=1e-12)
    assert np.linalg.det(b) == pytest.approx(1.0)


def test_boost_preserves_interval_of_random_events():
    rng = np.random.default_rng(1)
    b = lorentz_boost_matrix("y", 0.7) @ lorentz_boost_matrix("x", -0.4)
    a, c = rng.normal(size=4), rng.normal(size=4)
    assert minkowski_interval(b @ a, b @ c) == pytest.approx(minkowski_interval(a, c))


def test_time_dilation_and_simultaneity():
    beta = 0.6
    gamma = 1.25
    b = lorentz_boost_matrix("x", beta)
    assert (b @ np.array([0, 0, 0, 1.0]))[3] == pytest.approx(gamma)  # w' de um evento em repouso
    moved = b @ np.array([1.0, 0, 0, 0])  # simultâneo em A, mas não em B
    assert moved[3] == pytest.approx(-gamma * beta)


def test_boost_composition_adds_rapidity():
    r = np.arctanh(0.3) + np.arctanh(0.4)
    composed = lorentz_boost_matrix("x", 0.3) @ lorentz_boost_matrix("x", 0.4)
    assert np.allclose(composed, lorentz_boost_matrix("x", np.tanh(r)))


@pytest.mark.parametrize("beta", [1.0, -1.0, 1.5])
def test_boost_rejects_invalid_beta_and_axis(beta):
    with pytest.raises(ValueError):
        lorentz_boost_matrix("x", beta)
    with pytest.raises(ValueError):
        lorentz_boost_matrix("w", 0.1)


def test_classification_is_boost_invariant():
    origin = np.zeros(4)
    cases = {
        "tipo-tempo": [0, 0, 0, 2.0], "tipo-luz": [1, 0, 0, 1.0], "tipo-espaço": [3, 0, 0, 1.0],
    }
    b = lorentz_boost_matrix("x", 0.8)
    for kind, event in cases.items():
        assert classify_interval(origin, event) == kind
        assert classify_interval(origin, b @ np.array(event)) == kind


def test_projection_with_boost_matches_matrix():
    pts = np.random.default_rng(2).normal(size=(5, 4))
    out = project_points(pts, "drop_z", boosts=[("x", 0.5)])
    expected = (pts @ lorentz_boost_matrix("x", 0.5).T)[:, [0, 1, 3]]
    assert np.allclose(out, expected)


def test_light_time_and_retarded_time_in_a_moving_system():
    g = Graph4D(t_max=10.0, light_speed=2.0)
    g.add_node("obs", (0, 0, 0, 0))
    g.add_node("src", (6, 0, 0, 0))  # estático: a luz leva 3 unidades de tempo
    assert light_time(g, 6.0) == pytest.approx(3.0)
    assert to_ct(g, 5.0) == pytest.approx(10.0)
    assert retarded_time(g, "src", "obs", 8.0) == pytest.approx(5.0)
