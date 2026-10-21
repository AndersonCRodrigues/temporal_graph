"""Testes dos geradores."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.generator import (
    generate_knowledge_demo,
    generate_random,
    generate_simplex4,
    generate_tesseract,
)


def assert_edges_after_endpoints(g):
    for e in g.edges:
        assert e.t >= g.nodes[e.src].w and e.t >= g.nodes[e.dst].w


def test_random_sizes_and_ranges():
    g = generate_random(n_nodes=10, n_edges=12, n_steps=4, seed=1)
    assert len(g.nodes) == 10 and len(g.edges) == 12
    for n in g.nodes.values():
        assert n.coords.shape == (4,)
        assert 0 <= n.w <= 4 and n.w == int(n.w)
    assert_edges_after_endpoints(g)


def test_random_is_reproducible():
    a = generate_random(8, 10, 4, seed=99)
    b = generate_random(8, 10, 4, seed=99)
    np.testing.assert_array_equal(a.coords_matrix(), b.coords_matrix())
    assert [(e.src, e.dst, e.t) for e in a.edges] == [(e.src, e.dst, e.t) for e in b.edges]


def test_random_caps_edges_at_complete_graph():
    g = generate_random(n_nodes=4, n_edges=100, n_steps=3)
    assert len(g.edges) == 6


def test_random_label_is_degree_at_least_three():
    g = generate_random(n_nodes=12, n_edges=20, seed=5)
    degree = {n: 0 for n in g.nodes}
    for e in g.edges:
        degree[e.src] += 1
        degree[e.dst] += 1
    assert all(g.nodes[n].label == int(d >= 3) for n, d in degree.items())


@pytest.mark.parametrize("kwargs", [{"n_nodes": 0}, {"n_edges": -1}, {"n_steps": 0}])
def test_random_invalid(kwargs):
    with pytest.raises(ValueError):
        generate_random(**kwargs)


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_tesseract(scale):
    g = generate_tesseract(scale)
    assert len(g.nodes) == 16 and len(g.edges) == 32
    for n in g.nodes.values():
        assert abs(n.x) == scale and abs(n.y) == scale and abs(n.z) == scale
        assert n.w in (0.0, 2.0 * scale)
        assert n.w <= g.t_max
    assert_edges_after_endpoints(g)


def test_simplex4():
    g = generate_simplex4()
    assert len(g.nodes) == 5 and len(g.edges) == 10
    ws = sorted(n.w for n in g.nodes.values())
    assert ws[0] == pytest.approx(0.0) and ws[-1] == pytest.approx(4.0, abs=1e-6)
    assert_edges_after_endpoints(g)


def test_knowledge_demo():
    g = generate_knowledge_demo()
    assert len(g.nodes) == 9 and len(g.edges) == 11
    assert all(n.text for n in g.nodes.values())
    assert {e.relation for e in g.edges} >= {"constrói", "gerencia", "atende"}
    assert sum(e.t_end is not None for e in g.edges) == 2
    assert_edges_after_endpoints(g)
    assert all(abs(c) <= 2.0 + 1e-9 for n in g.nodes.values() for c in n.coords[:3])


def test_knowledge_demo_is_reproducible():
    a, b = generate_knowledge_demo(), generate_knowledge_demo()
    np.testing.assert_array_equal(a.coords_matrix(), b.coords_matrix())
