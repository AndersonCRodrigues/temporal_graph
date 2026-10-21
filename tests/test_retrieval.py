"""Testes da recuperação temporal."""

from __future__ import annotations

import pytest

from temporal_graph.generator import KNOWLEDGE_DEMO_QUERY, generate_knowledge_demo
from temporal_graph.model import Graph4D
from temporal_graph.retrieval import build_context, ensure_embeddings, retrieve


@pytest.fixture
def demo():
    return generate_knowledge_demo()


def ids_at(demo, t, **kw):
    return {h.id: h.hop for h in retrieve(demo, KNOWLEDGE_DEMO_QUERY, t=t, **kw).hits}


def test_seed_is_the_matching_node(demo):
    r = retrieve(demo, KNOWLEDGE_DEMO_QUERY, t=8)
    assert [h.id for h in r.seeds()] == ["Atlas"]
    assert 0 < r.seeds()[0].score <= 1


def test_same_query_changes_with_time(demo):
    # Ana trabalha no Atlas em [1, 5); a Delta é atendida em [4, 7)
    assert "Ana" in ids_at(demo, 3) and "Delta" not in ids_at(demo, 3)
    assert "Ana" in ids_at(demo, 4) and "Delta" in ids_at(demo, 4)
    assert "Ana" not in ids_at(demo, 5) and "Delta" in ids_at(demo, 6)
    assert "Ana" not in ids_at(demo, 8) and "Delta" not in ids_at(demo, 8)


def test_future_nodes_are_not_retrievable(demo):
    assert retrieve(demo, "Atlas", t=0).hits == []


def test_hops_zero_returns_only_seeds(demo):
    assert set(ids_at(demo, 3, hops=0).values()) == {0}


def test_more_hops_reach_further(demo):
    # em t = 8 o vínculo Ana-Atlas já expirou: Ana só é alcançável por caminhos mais longos
    one, two, three = (ids_at(demo, 8, hops=h) for h in (1, 2, 3))
    assert set(one) < set(two) < set(three)
    assert "Ana" not in two
    assert two["Acme"] == 2 and three["Ana"] == 3


def test_k_limits_seeds():
    g = Graph4D(t_max=5)
    for i in range(4):
        g.add_node(f"n{i}", (0, 0, 0, 0), text="dados análise")
    assert len(retrieve(g, "dados", k=2, hops=0).hits) == 2


def test_no_common_words_gives_empty_result(demo):
    r = retrieve(demo, "zebra", t=8)
    assert r.hits == [] and r.edges == []


def test_edges_are_active_edges_between_hits(demo):
    r = retrieve(demo, KNOWLEDGE_DEMO_QUERY, t=3)
    assert ("Ana", "Atlas") in r.edges
    assert all(a in r.ids() and b in r.ids() for a, b in r.edges)
    assert ("Ana", "Atlas") not in retrieve(demo, KNOWLEDGE_DEMO_QUERY, t=6).edges


def test_default_time_is_t_max(demo):
    assert retrieve(demo, KNOWLEDGE_DEMO_QUERY).t == demo.t_max


def test_embeddings_are_cached_and_nodes_without_text_are_skipped():
    g = Graph4D()
    g.add_node("a", (0, 0, 0, 0), text="dados")
    g.add_node("b", (0, 0, 0, 0))
    ensure_embeddings(g)
    first = g.nodes["a"].embedding
    ensure_embeddings(g)
    assert g.nodes["a"].embedding is first and g.nodes["b"].embedding is None


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"hops": -1}])
def test_invalid_parameters(demo, kwargs):
    with pytest.raises(ValueError):
        retrieve(demo, "Atlas", **kwargs)


def test_context_lists_nodes_then_relations_with_validity(demo):
    ctx = build_context(demo, retrieve(demo, KNOWLEDGE_DEMO_QUERY, t=4))
    assert ctx.startswith("Contexto em t = 4")
    assert "- Atlas [semente]:" in ctx
    assert "Ana --constrói--> Atlas  [vale de 1 até 5)" in ctx
    assert "Atlas --atende--> Delta  [vale de 4 até 7)" in ctx


def test_context_empty_when_nothing_found(demo):
    assert build_context(demo, retrieve(demo, "zebra")) == ""
