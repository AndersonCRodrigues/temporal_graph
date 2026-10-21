"""Teste de fumaça da aplicação Trame, sem abrir navegador."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("trame")
from trame.app import get_server  # noqa: E402

from temporal_graph.app import create_app  # noqa: E402


@pytest.fixture
def server():
    # get_server() sem nome devolve sempre o mesmo servidor, e os handlers de testes
    # anteriores continuariam registrados: cada teste ganha um servidor próprio.
    s = create_app(get_server(name=f"test-{uuid.uuid4().hex}", client_type="vue3"))
    s.state.ready()  # sem isso o trame não dispara os @state.change fora de um cliente
    return s


def test_initial_state(server):
    assert server.state.db_nodes == 15
    assert server.state.t_filter == server.state.db_t_max


@pytest.mark.parametrize(
    ("preset", "nodes", "edges"), [("tesseract", 16, 32), ("simplex4", 5, 10), ("random", 15, 25)]
)
def test_presets(server, preset, nodes, edges):
    server.state.preset = preset
    server.controller.regenerate()
    assert (server.state.db_nodes, server.state.db_edges) == (nodes, edges)


def test_add_then_remove_node(server):
    state, ctrl = server.state, server.controller
    state.new_node_id = "Z1"
    ctrl.add_node()
    assert state.db_nodes == 16
    state.new_node_id = "Z1"
    ctrl.add_node()
    assert state.status_type == "warning"
    state.selected_node = "Z1"
    ctrl.remove_selected()
    assert state.db_nodes == 15


def load_demo(server):
    server.state.preset = "knowledge"
    server.controller.regenerate()
    assert server.state.retrieval_query


def test_demo_search_follows_the_time_slider(server):
    state = server.state
    load_demo(server)
    server.controller.run_retrieval()
    assert state.retrieval_active and state.retrieval_rows[0].startswith("Atlas")

    state.t_filter = 3.0
    state.flush()
    assert any(row.startswith("Ana") for row in state.retrieval_rows)
    assert state.retrieval_t == 3.0

    state.t_filter = 6.0
    state.flush()
    assert not any(row.startswith("Ana") for row in state.retrieval_rows)
    assert any(row.startswith("Delta") for row in state.retrieval_rows)
    assert "Contexto em t = 6" in state.retrieval_context


def test_search_without_filter_uses_t_max(server):
    state = server.state
    load_demo(server)
    state.use_t_filter = False
    server.controller.run_retrieval()
    assert state.retrieval_t == state.db_t_max


def test_empty_query_and_no_match(server):
    state, ctrl = server.state, server.controller
    state.retrieval_query = "  "
    ctrl.run_retrieval()
    assert not state.retrieval_active and state.status_type == "warning"
    load_demo(server)
    state.retrieval_query = "zebra"
    ctrl.run_retrieval()
    assert state.retrieval_rows == [] and state.status_type == "warning"


def test_clear_retrieval_and_graph_change_reset_search(server):
    state, ctrl = server.state, server.controller
    load_demo(server)
    ctrl.run_retrieval()
    ctrl.clear_retrieval()
    assert not state.retrieval_active and state.retrieval_rows == []
    ctrl.run_retrieval()
    state.preset = "tesseract"
    ctrl.regenerate()
    assert not state.retrieval_active


def test_new_node_text_enters_the_search(server):
    state, ctrl = server.state, server.controller
    load_demo(server)
    state.new_node_id, state.new_node_w = "Eva", 1.0
    state.new_node_text = "Eva, especialista em zebras"
    ctrl.add_node()
    state.retrieval_query = "zebras"
    ctrl.run_retrieval()
    assert state.retrieval_rows[0].startswith("Eva")


def test_moving_selected_node_updates_split(server):
    state = server.state
    state.selected_node = "S0"
    state.sel_x = state.sel_y = state.sel_z = 0.0
    state.sel_w = 0.0
    state.flush()
    state.sel_w = float(state.db_t_max)
    state.flush()
    assert state.sel_split == "test"


# -- modo sistema solar ---------------------------------------------------------------------


@pytest.fixture
def solar_server(server):
    server.state.app_mode = "solar"
    server.state.flush()
    return server


def test_switching_to_solar_loads_the_solar_system(solar_server):
    state = solar_server.state
    assert state.graph_has_bodies and state.db_nodes == 52
    assert state.time_text == "2026-10-19" and state.t_step == 1.0
    assert state.retrieval_query == "satélites de Júpiter" and state.show_edges is False
    assert "Terra" in state.body_items and len(state.body_items) == 52


def test_each_mode_keeps_its_own_graph(server):
    state, ctrl = server.state, server.controller
    state.new_node_id = "Meu"
    ctrl.add_node()
    assert state.db_nodes == 16
    state.app_mode = "solar"
    state.flush()
    assert state.db_nodes == 52
    state.app_mode = "grafos"
    state.flush()
    assert state.db_nodes == 16 and not state.graph_has_bodies and state.t_step == 0.5
    assert state.show_edges is True and state.retrieval_k == 3


def test_solar_search_finds_the_moons_of_jupiter(solar_server):
    state = solar_server.state
    solar_server.controller.run_retrieval()
    assert state.retrieval_active and state.retrieval_k == 6
    assert sum("semente" in row for row in state.retrieval_rows) == 6
    assert any(row.startswith("Júpiter") for row in state.retrieval_rows)
    assert "ressonância" in state.retrieval_context  # Io-Europa e Europa-Ganimedes


def test_solar_date_slider_moves_the_search_time(solar_server):
    state = solar_server.state
    solar_server.controller.run_retrieval()
    state.t_filter = 100.0
    state.flush()
    assert state.time_text == "2000-04-10" and state.retrieval_t == 100.0


@pytest.mark.parametrize("scale", ["log", "sqrt", "real"])
def test_solar_options_render_without_errors(solar_server, scale):
    state = solar_server.state
    state.scale_mode, state.show_trails, state.trail_span = scale, True, 730.0
    state.proj_mode = "drop_z"
    state.flush()
    state.minkowski, state.transform_kind, state.boost_x = True, "lorentz", 0.5
    state.flush()
    assert state.status_type != "warning"


def test_solar_dynamics_can_be_switched_to_nbody(solar_server):
    state = solar_server.state
    state.dynamics = "nbody"
    solar_server.controller.reload_solar()
    assert "nbody" in state.status_msg and state.db_nodes == 52


def test_generic_minkowski_cone_uses_the_selected_node(server):
    state = server.state
    state.selected_node = "S0"
    state.minkowski = True
    state.transform_kind, state.boost_y = "lorentz", -0.3
    state.flush()
    assert state.minkowski and state.status_type != "warning"


def test_moving_a_body_with_sliders_is_ignored(solar_server):
    state = solar_server.state
    state.selected_node = "Terra"
    state.sel_x = 99.0
    state.flush()
    assert state.db_nodes == 52 and state.selected_node == "Terra"
