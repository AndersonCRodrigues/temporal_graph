"""Testes do modelo Graph4D."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.model import Edge4D, Graph4D, Node4D


def make_graph() -> Graph4D:
    g = Graph4D(t_max=10)
    g.add_node("a", coords=(0, 0, 0, 0))
    g.add_node("b", coords=(1, 0, 0, 4))
    return g


class TestNode4D:
    def test_coords_roundtrip(self):
        n = Node4D(id="A", x=1, y=2, z=3, w=4)
        assert n.as_tuple() == (1, 2, 3, 4)
        np.testing.assert_array_equal(n.coords, [1, 2, 3, 4])
        n.coords = (0.5, -0.5, 1.5, -1.5)
        assert n.w == -1.5

    def test_coords_length(self):
        with pytest.raises(ValueError):
            Node4D(id="A").coords = (1, 2, 3)


class TestEdge4D:
    def test_endpoints(self):
        assert Edge4D("a", "b").endpoints() == {"a", "b"}


class TestEditing:
    def test_add_node_rejects_duplicates_and_empty_id(self):
        g = Graph4D()
        assert g.add_node("n1", coords=(0, 0, 0, 0))
        assert not g.add_node("n1", coords=(1, 1, 1, 1))
        assert not g.add_node("", coords=(0, 0, 0, 0))

    def test_add_node_wrong_coords_length(self):
        with pytest.raises(ValueError):
            Graph4D().add_node("a", coords=(0, 0, 0))

    def test_t_birth_overrides_w(self):
        g = Graph4D()
        g.add_node("a", coords=(0, 0, 0, 9), t_birth=2)
        assert g.nodes["a"].w == 2

    def test_add_node_without_coords_is_reproducible(self):
        g1, g2 = Graph4D(t_max=5), Graph4D(t_max=5)
        g1.add_node("a", rng=np.random.default_rng(3))
        g2.add_node("a", rng=np.random.default_rng(3))
        np.testing.assert_array_equal(g1.nodes["a"].coords, g2.nodes["a"].coords)
        assert 0 <= g1.nodes["a"].w <= 5

    def test_add_edge_rules(self):
        g = make_graph()
        assert g.add_edge("a", "b")
        assert not g.add_edge("b", "a")  # não direcionada
        assert not g.add_edge("a", "a")
        assert not g.add_edge("a", "zzz")

    def test_edge_time_never_precedes_endpoints(self):
        g = make_graph()
        g.add_edge("a", "b", t=1)
        assert g.edges[0].t == 4

    def test_moving_node_later_pushes_edge_time(self):
        g = make_graph()
        g.add_edge("a", "b", t=4)
        g.set_coords("a", (0, 0, 0, 7))
        assert g.edges[0].t == 7

    def test_remove_node_drops_its_edges(self):
        g = make_graph()
        g.add_edge("a", "b")
        assert g.remove_node("a")
        assert g.edges == []
        assert not g.remove_node("a")

    def test_remove_edge(self):
        g = make_graph()
        g.add_edge("a", "b")
        assert g.remove_edge("b", "a")
        assert not g.remove_edge("a", "b")

    def test_set_coords_unknown_node(self):
        assert not Graph4D().set_coords("x", (0, 0, 0, 0))


class TestEdgeValidity:
    def test_interval_is_half_open(self):
        e = Edge4D("a", "b", t=2, t_end=5)
        assert [e.is_active(t) for t in (1.9, 2, 4.9, 5, 6)] == [False, True, True, False, False]

    def test_open_ended_edge_never_expires(self):
        assert Edge4D("a", "b", t=2).is_active(1000)

    def test_end_must_be_after_start(self):
        g = make_graph()
        assert not g.add_edge("a", "b", t=1, t_end=4)  # início efetivo é 4 (nascimento de b)
        assert not g.add_edge("a", "b", t=5, t_end=5)
        assert g.add_edge("a", "b", t=5, t_end=6)

    def test_snapshot_drops_expired_edges(self):
        g = make_graph()
        g.add_edge("a", "b", t=4, t_end=7, relation="conhece")
        assert g.snapshot(5).number_of_edges() == 1
        assert g.snapshot(5)["a"]["b"][0]["relation"] == "conhece"
        assert g.snapshot(7).number_of_edges() == 0

    def test_moving_node_keeps_edge_duration(self):
        g = make_graph()
        g.add_edge("a", "b", t=4, t_end=7)
        g.set_coords("a", (0, 0, 0, 6))
        assert (g.edges[0].t, g.edges[0].t_end) == (6, 9)

    def test_to_networkx_has_validity(self):
        g = make_graph()
        g.add_edge("a", "b", t=4, t_end=7, relation="r")
        data = g.to_networkx()["a"]["b"][0]
        assert (data["t"], data["t_end"], data["relation"]) == (4, 7, "r")


class TestNodeText:
    def test_text_and_set_text_clears_embedding(self):
        g = Graph4D()
        g.add_node("a", coords=(0, 0, 0, 0), text="olá", embedding=np.ones(3))
        assert g.nodes["a"].text == "olá"
        assert g.set_text("a", "novo")
        assert g.nodes["a"].text == "novo" and g.nodes["a"].embedding is None
        assert not g.set_text("zzz", "x")

    def test_snapshot_carries_text(self):
        g = Graph4D()
        g.add_node("a", coords=(0, 0, 0, 0), text="olá")
        assert g.snapshot(0).nodes["a"]["text"] == "olá"


class TestRecurringEdges:
    def test_same_relation_needs_disjoint_windows(self):
        g = make_graph()
        assert g.add_edge("a", "b", t=4, t_end=6, relation="evento")
        assert not g.add_edge("a", "b", t=5, t_end=8, relation="evento")  # sobrepõe
        assert g.add_edge("a", "b", t=6, t_end=8, relation="evento")  # [6, 8) não sobrepõe [4, 6)
        assert g.add_edge("a", "b", t=9, relation="evento")

    def test_different_relations_coexist(self):
        g = make_graph()
        assert g.add_edge("a", "b", t=4, relation="orbita")
        assert g.add_edge("a", "b", t=5, t_end=6, relation="eclipse")
        assert len(g.snapshot(5).get_edge_data("a", "b")) == 2

    def test_remove_edge_by_relation(self):
        g = make_graph()
        g.add_edge("a", "b", t=4, relation="x")
        g.add_edge("a", "b", t=4, relation="y")
        assert g.remove_edge("a", "b", relation="x")
        assert [e.relation for e in g.edges] == ["y"]
        assert g.remove_edge("a", "b")
        assert g.edges == []


class TestBodies:
    @staticmethod
    def system():
        class Circle:
            def __init__(self, r, period):
                self.r, self.period = r, period

            def position(self, t):
                t = np.asarray(t, dtype=float)
                ang = 2 * np.pi * t / self.period
                return np.stack([self.r * np.cos(ang), self.r * np.sin(ang), 0 * t], axis=-1)

        g = Graph4D(t_max=100.0)
        g.add_body("sol", kind="estrela")
        g.add_body("terra", parent="sol", orbit=Circle(1.0, 100.0))
        g.add_body("lua", parent="terra", orbit=Circle(0.1, 10.0), kind="lua")
        return g

    def test_one_node_per_body_and_position_follows_the_chain(self):
        g = self.system()
        assert len(g.nodes) == 3 and g.has_bodies
        assert np.allclose(g.position_at("sol", 7.0), 0.0)
        assert np.allclose(g.position_at("terra", 25.0), [0, 1, 0], atol=1e-12)
        assert np.allclose(g.position_at("lua", 0.0), [1.1, 0, 0])
        assert np.allclose(g.local_position_at("lua", 0.0), [0.1, 0, 0])

    def test_position_accepts_time_arrays(self):
        g = self.system()
        assert g.position_at("lua", np.linspace(0, 100, 7)).shape == (7, 3)

    def test_birth_coords_are_the_position_at_birth(self):
        g = self.system()
        assert np.allclose(g.nodes["lua"].coords, [1.1, 0, 0, 0])

    def test_graph_does_not_grow_with_time_samples(self):
        g = self.system()
        before = len(g.nodes)
        g.position_at("lua", np.linspace(0, 100, 5000))
        g.snapshot(50.0)
        assert len(g.nodes) == before

    def test_snapshot_reports_current_position_not_birth(self):
        g = self.system()
        assert np.allclose(g.snapshot(25.0).nodes["terra"]["pos"], [0, 1, 0], atol=1e-12)

    def test_removing_a_body_removes_its_moons(self):
        g = self.system()
        g.add_edge("terra", "lua", relation="orbita")
        assert g.remove_node("terra")
        assert set(g.nodes) == {"sol"} and g.edges == []

    def test_add_body_needs_an_existing_parent_and_unique_id(self):
        g = self.system()
        assert not g.add_body("x", parent="nao-existe")
        assert not g.add_body("terra", parent="sol")

    def test_static_nodes_keep_their_position(self):
        g = self.system()
        g.add_node("ponto", (3, 4, 5, 0))
        assert np.allclose(g.position_at("ponto", 99.0), [3, 4, 5])


class TestTimeLabel:
    def test_plain_number_without_epoch(self):
        assert Graph4D().time_label(2.5) == "2.5"

    def test_date_with_epoch(self):
        from datetime import datetime

        g = Graph4D(epoch=datetime(2000, 1, 1, 12))
        assert g.time_label(0.0) == "2000-01-01"
        assert g.time_label(366.0) == "2001-01-01"


class TestSplitAndStats:
    def test_split_follows_birth_time(self):
        g = Graph4D(t_max=10)
        for nid, w in (("a", 6), ("b", 7), ("c", 8), ("d", 9)):
            g.add_node(nid, coords=(0, 0, 0, w))
        assert [g.nodes[i].split for i in "abcd"] == ["train", "val", "val", "test"]

    def test_split_updates_when_t_max_changes(self):
        g = Graph4D(t_max=10)
        g.add_node("a", coords=(0, 0, 0, 9))
        assert g.nodes["a"].split == "test"
        g.set_t_max(20)
        assert g.nodes["a"].split == "train"

    def test_stats_keeps_fractional_t_max(self):
        g = Graph4D(t_max=2.5)
        g.add_node("a", coords=(0, 0, 0, 0))
        s = g.stats()
        assert s["t_max"] == 2.5
        assert s["n_nodes"] == 1 and s["n_train"] == 1


class TestQueries:
    def test_snapshot(self):
        g = Graph4D(t_max=5)
        g.add_node("a", coords=(0, 0, 0, 0))
        g.add_node("b", coords=(1, 0, 0, 2))
        g.add_edge("a", "b", t=3)
        assert set(g.snapshot(0).nodes) == {"a"}
        s2 = g.snapshot(2)
        assert set(s2.nodes) == {"a", "b"} and s2.number_of_edges() == 0
        assert g.snapshot(3).number_of_edges() == 1

    def test_to_networkx_keeps_attributes(self):
        g = make_graph()
        g.add_edge("a", "b", t=5)
        nxg = g.to_networkx()
        assert nxg.nodes["b"]["t_birth"] == 4
        assert nxg["a"]["b"][0]["t"] == 5

    def test_coords_matrix_sorted_by_id(self):
        g = Graph4D()
        g.add_node("b", coords=(0, 0, 0, 1))
        g.add_node("a", coords=(1, 0, 0, 0))
        m = g.coords_matrix()
        assert m.shape == (2, 4)
        assert m[0, 0] == 1  # "a" primeiro

    def test_coords_matrix_empty(self):
        assert Graph4D().coords_matrix().shape == (0, 4)
