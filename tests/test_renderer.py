"""Testes de fumaça do renderer (off-screen)."""

from __future__ import annotations

import pytest
import pyvista as pv

from temporal_graph.generator import (
    KNOWLEDGE_DEMO_QUERY,
    generate_knowledge_demo,
    generate_random,
    generate_tesseract,
)
from temporal_graph.model import Graph4D
from temporal_graph.renderer import SPLIT_COLORS, SceneRenderer
from temporal_graph.retrieval import retrieve

pv.OFF_SCREEN = True


@pytest.fixture
def db():
    return generate_random(n_nodes=6, n_edges=8, n_steps=4, seed=11)


@pytest.fixture
def renderer():
    return SceneRenderer()


@pytest.mark.parametrize("mode", ["drop_w", "drop_x", "drop_y", "drop_z", "perspective"])
def test_every_projection_mode_renders_all_nodes(db, renderer, mode):
    renderer.render(db, proj_mode=mode)
    assert len(renderer.actor_to_node) == len(db.nodes)


def test_rotations(db, renderer):
    renderer.render(db, rot_xw=0.5, rot_yw=-0.3, rot_zw=0.1)
    assert len(renderer.actor_to_node) == len(db.nodes)


def test_tesseract(renderer):
    renderer.render(generate_tesseract(), proj_mode="perspective", camera_w=3.0)
    assert len(renderer.actor_to_node) == 16


def test_time_filter_hides_later_nodes(db, renderer):
    renderer.render(db, t_filter=1)
    visible = {nid for nid in renderer.actor_to_node.values()}
    assert visible == {n.id for n in db.nodes.values() if n.w <= 1}


def test_filter_with_nothing_visible(renderer):
    g = Graph4D()
    g.add_node("a", coords=(0, 0, 0, 5))
    renderer.render(g, t_filter=1)
    assert renderer.actor_to_node == {}


def test_empty_graph(renderer):
    renderer.render(Graph4D())
    assert renderer.actor_to_node == {}


def test_selection_and_first_edge_node(db, renderer):
    a, b = list(db.nodes)[:2]
    renderer.render(db, selected_node=a, edge_first=b)
    assert {a, b} <= set(renderer.actor_to_node.values())


def test_pick_callback(db, renderer):
    picked = []
    renderer.set_pick_callback(picked.append)
    renderer.render(db)
    actor, nid = next(iter(renderer.actor_to_node.items()))
    renderer._handle_pick(actor)
    assert picked == [nid]


def test_graph_without_edges(renderer):
    g = Graph4D()
    g.add_node("a", coords=(0, 0, 0, 0))
    renderer.render(g)
    assert len(renderer.actor_to_node) == 1


def test_split_colors():
    assert set(SPLIT_COLORS) == {"train", "val", "test"}


def test_retrieval_and_expired_edges_render(renderer):
    g = generate_knowledge_demo()
    result = retrieve(g, KNOWLEDGE_DEMO_QUERY, t=6)
    renderer.render(g, t_filter=6, retrieval=result)
    assert len(renderer.actor_to_node) == len([n for n in g.nodes.values() if n.w <= 6])


def test_retrieval_without_filter_and_empty_result(renderer):
    g = generate_knowledge_demo()
    renderer.render(g, retrieval=retrieve(g, "zebra"))
    assert len(renderer.actor_to_node) == 9


# -- sistema solar ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def solar():
    from temporal_graph.celestial.catalog import generate_solar_system

    return generate_solar_system(events=False)


def solar_frame(solar, **kw):
    from temporal_graph.celestial.scene import SceneOptions, build_frame

    return build_frame(solar, solar.t_default, SceneOptions(**kw))


def test_solar_scene_draws_one_actor_per_body(solar, renderer):
    renderer.render(solar, t_filter=solar.t_default, frame=solar_frame(solar), show_edges=False)
    assert len(renderer.actor_to_node) == len(solar.nodes)


@pytest.mark.parametrize("mode", ["log", "sqrt", "real"])
def test_solar_scene_in_every_scale(solar, renderer, mode):
    from temporal_graph.celestial.scale import DisplayScale

    frame = solar_frame(solar, scale=DisplayScale(mode=mode), trails=True)
    renderer.render(solar, t_filter=solar.t_default, frame=frame, proj_mode="drop_z")
    assert len(renderer.actor_to_node) == len(solar.nodes)


def test_minkowski_scene_only_draws_bodies_inside_the_cone(solar, renderer):
    frame = solar_frame(solar, minkowski=True)
    renderer.render(solar, t_filter=solar.t_default, frame=frame, proj_mode="drop_z",
                    boosts=[("x", 0.6)], show_edges=False)
    assert set(renderer.actor_to_node.values()) == frame.visible
    assert 0 < len(frame.visible) < len(solar.nodes)


def test_labels_and_selection_in_the_solar_scene(solar, renderer):
    renderer.render(solar, t_filter=solar.t_default, frame=solar_frame(solar),
                    label_ids={"Sol", "Terra"}, selected_node="Terra", show_edges=False)
    renderer.render(solar, t_filter=solar.t_default, frame=solar_frame(solar), label_ids=set())
    assert len(renderer.actor_to_node) == len(solar.nodes)


def test_retrieval_highlights_the_moons_of_jupiter(solar, renderer):
    result = retrieve(solar, "satélites de Júpiter", t=solar.t_default, k=6, hops=1)
    assert {"Io", "Europa", "Ganimedes", "Calisto", "Amalteia", "Himalia"} <= {
        h.id for h in result.seeds()
    }
    renderer.render(solar, t_filter=solar.t_default, frame=solar_frame(solar), retrieval=result)
    assert len(renderer.actor_to_node) == len(solar.nodes)


def test_generic_graph_in_minkowski_mode(renderer):
    from temporal_graph.frame import static_frame

    g = generate_random(n_nodes=6, n_edges=8, n_steps=4, seed=11)
    ids = sorted(g.nodes)
    frame = static_frame(g, ids, observer=ids[0], cone=True)
    renderer.render(g, frame=frame, proj_mode="drop_z", boosts=[("y", -0.4)])
    assert len(renderer.actor_to_node) == len(g.nodes)
