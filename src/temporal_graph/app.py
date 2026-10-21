"""Aplicação Trame: estado, callbacks e ponto de entrada.

Dois modos na mesma aplicação, cada um com o seu grafo:
  grafos  grafos 4D genéricos (o grafo do usuário é preservado ao trocar de modo)
  solar   sistema solar, criado na primeira vez que o modo é aberto
"""

from __future__ import annotations

import math

import pyvista as pv
from trame.app import get_server

from .celestial.catalog import describe_body, generate_solar_system
from .celestial.scale import DisplayScale
from .celestial.scene import SceneOptions, build_frame
from .frame import static_frame
from .generator import (
    KNOWLEDGE_DEMO_QUERY,
    generate_knowledge_demo,
    generate_random,
    generate_simplex4,
    generate_tesseract,
)
from .projection import DEFAULT_CAMERA_W
from .renderer import SceneRenderer
from .retrieval import build_context, retrieve

INITIAL_STATUS = "(x, y, z) = espaço, w = tempo. Use o slider de t para ver snapshots."
SOLAR_STATUS = "Sistema solar. Mexa na data, ligue as trilhas ou a vista Minkowski."
SOLAR_QUERY = "satélites de Júpiter"


def _init_state(state, db) -> None:
    state.update({
        "app_mode": "grafos",
        "proj_mode": "drop_w",
        "camera_w": DEFAULT_CAMERA_W,
        "transform_kind": "rotation",
        "rot_xw_deg": 0.0, "rot_yw_deg": 0.0, "rot_zw_deg": 0.0,
        "boost_x": 0.0, "boost_y": 0.0, "boost_z": 0.0,
        "minkowski": False, "observer_id": "Terra", "minkowski_span": 0.25,
        "use_t_filter": True,
        "t_filter": float(db.t_max), "t_step": db.t_step, "time_text": "",
        "selected_node": None,
        "edge_first": None,
        "edit_mode": "select",
        "pending_edge_t": 0.0,
        "edge_has_end": False,
        "pending_edge_t_end": float(db.t_max),
        "pending_edge_relation": "",
        "new_node_id": "",
        "new_node_text": "",
        "new_node_x": 0.0, "new_node_y": 0.0, "new_node_z": 0.0, "new_node_w": 0.0,
        "sel_x": 0.0, "sel_y": 0.0, "sel_z": 0.0, "sel_w": 0.0,
        "sel_label": 0, "sel_split": "train", "sel_text": "", "sel_info": "",
        "sel_is_body": False,
        "retrieval_query": "", "retrieval_k": 3, "retrieval_hops": 1,
        "retrieval_active": False, "retrieval_t": 0.0,
        "retrieval_rows": [], "retrieval_context": "",
        "status_msg": INITIAL_STATUS, "status_type": "info",
        "gen_n_nodes": 15, "gen_n_edges": 25, "gen_n_steps": 6, "gen_seed": 42,
        "preset": "random",
        "dynamics": "kepler", "scale_mode": "log", "moon_extent": 0.08, "size_boost": 1.0,
        "show_trails": False, "trail_span": 365.0, "show_edges": True,
        "show_moon_labels": False,
        "graph_has_bodies": False, "body_items": [],
    })


def create_app(server=None):
    pv.OFF_SCREEN = True
    server = server or get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    graphs = {"grafos": generate_random(), "solar": None}
    current = {"db": graphs["grafos"], "retrieval": None}
    _init_state(state, current["db"])

    plotter = pv.Plotter()
    renderer = SceneRenderer(plotter)

    def refresh_stats() -> None:
        s = current["db"].stats()
        state.db_nodes, state.db_edges, state.db_t_max = s["n_nodes"], s["n_edges"], s["t_max"]
        state.db_n_train, state.db_n_val, state.db_n_test = s["n_train"], s["n_val"], s["n_test"]

    def set_status(msg: str, kind: str = "info") -> None:
        state.status_msg, state.status_type = msg, kind

    def current_time() -> float:
        db = current["db"]
        return float(state.t_filter) if state.use_t_filter else float(db.t_max)

    def update_retrieval() -> None:
        """Refaz a busca no instante atual; roda a cada render para acompanhar o slider de t."""
        if not state.retrieval_active:
            current["retrieval"] = None
            return
        db = current["db"]
        result = retrieve(db, state.retrieval_query, t=current_time(),
                          k=int(state.retrieval_k), hops=int(state.retrieval_hops))
        current["retrieval"] = result
        state.retrieval_t = result.t
        state.retrieval_rows = [
            f"{h.id}: {'semente' if h.hop == 0 else f'vizinho ({h.hop})'}"
            + (f", similaridade {h.score:.2f}" if h.hop == 0 else "")
            for h in result.hits
        ]
        state.retrieval_context = build_context(db, result)

    def transform_args() -> dict:
        if state.transform_kind == "lorentz":
            boosts = [(a, float(getattr(state, f"boost_{a}"))) for a in "xyz"]
            return {"boosts": [b for b in boosts if b[1]]}
        return {
            "rot_xw": math.radians(float(state.rot_xw_deg)),
            "rot_yw": math.radians(float(state.rot_yw_deg)),
            "rot_zw": math.radians(float(state.rot_zw_deg)),
        }

    def make_frame(t_view: float):
        db = current["db"]
        if db.has_bodies:
            opts = SceneOptions(
                scale=DisplayScale(mode=state.scale_mode, moon_extent=float(state.moon_extent),
                                   size_boost=float(state.size_boost)),
                trails=bool(state.show_trails), trail_span=float(state.trail_span),
                minkowski=bool(state.minkowski), observer=state.observer_id,
                minkowski_span=float(state.minkowski_span),
            )
            return build_frame(db, t_view, opts)
        if state.minkowski:
            ids = [n.id for n in db.nodes.values()
                   if not state.use_t_filter or n.w <= float(state.t_filter)]
            return static_frame(db, ids, observer=state.selected_node, cone=True)
        return None

    def do_render() -> None:
        db = current["db"]
        t_view = current_time()
        update_retrieval()
        state.time_text = db.time_label(t_view)
        label_ids = None
        if db.has_bodies and not state.show_moon_labels:
            label_ids = {n.id for n in db.nodes.values() if not n.body or n.body.kind != "lua"}
        renderer.render(
            db,
            proj_mode=state.proj_mode,
            camera_w=float(state.camera_w),
            t_filter=float(state.t_filter) if state.use_t_filter else None,
            selected_node=state.selected_node,
            edge_first=state.edge_first,
            retrieval=current["retrieval"],
            frame=make_frame(t_view),
            label_ids=label_ids,
            show_edges=bool(state.show_edges),
            **transform_args(),
        )
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()

    def reset_view() -> None:
        plotter.reset_camera()
        if current["db"].has_bodies and not state.minkowski:
            plotter.view_xy()  # de cima, como num mapa do sistema solar
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()

    def apply_graph(db, *, status: str) -> None:
        """Troca o grafo ativo e leva o estado da interface para valores coerentes."""
        current["db"] = db
        state.selected_node = state.edge_first = None
        state.sel_is_body = False
        state.retrieval_active = False
        state.retrieval_rows, state.retrieval_context = [], ""
        state.t_step = db.t_step
        state.use_t_filter = True
        state.t_filter = float(db.t_default if db.t_default is not None else db.t_max)
        state.pending_edge_t_end = float(db.t_max)
        state.graph_has_bodies = db.has_bodies
        state.body_items = sorted(n.id for n in db.nodes.values() if n.body)
        refresh_stats()
        set_status(status, "success" if db.has_bodies else "info")
        do_render()
        reset_view()

    refresh_stats()

    # -- seleção e conexão por clique --------------------------------------

    def connect(nid: str) -> None:
        first = state.edge_first
        if first is None:
            state.edge_first = nid
            set_status(f"1º nó: {nid}. Clique no segundo.")
        elif first == nid:
            state.edge_first = None
            set_status("Seleção cancelada.")
        else:
            t_end = float(state.pending_edge_t_end) if state.edge_has_end else None
            ok = current["db"].add_edge(
                first, nid, t=float(state.pending_edge_t), t_end=t_end,
                relation=(state.pending_edge_relation or "").strip(),
            )
            state.edge_first = None
            if ok:
                edge = current["db"].edges[-1]
                refresh_stats()
                end = "sem fim" if edge.t_end is None else f"fim em {edge.t_end:g}"
                set_status(f"Aresta {first} - {nid} criada: início em {edge.t:g}, {end}.",
                           "success")
            else:
                set_status("Aresta inválida, duplicada ou com fim antes do início.", "warning")
        state.flush()
        do_render()

    def select(nid: str) -> None:
        db = current["db"]
        n = db.nodes[nid]
        state.selected_node = nid
        state.sel_x, state.sel_y, state.sel_z, state.sel_w = n.as_tuple()
        state.sel_label, state.sel_split, state.sel_text = n.label, n.split, n.text
        state.sel_is_body = n.body is not None
        if n.body is not None:
            observer = state.observer_id if state.observer_id in db.nodes else None
            state.sel_info = describe_body(db, nid, current_time(), observer)
            set_status(f"{nid} ({n.body.kind})")
        else:
            state.sel_info = ""
            set_status(
                f"Nó {nid}: espaço = ({n.x:.2f}, {n.y:.2f}, {n.z:.2f}), tempo w = {n.w:.2f}"
            )
        state.flush()
        do_render()

    def on_pick(nid: str) -> None:
        (connect if state.edit_mode == "connect" else select)(nid)

    renderer.set_pick_callback(on_pick)

    # -- reações a mudanças de estado --------------------------------------

    @state.change("app_mode")
    def _on_mode(app_mode, **_):
        if app_mode == "solar":
            if graphs["solar"] is None:
                graphs["solar"] = generate_solar_system(dynamics=state.dynamics)
            state.show_edges = False
            state.retrieval_query = SOLAR_QUERY
            state.retrieval_k = 6  # as seis luas de Júpiter do catálogo
            apply_graph(graphs["solar"], status=SOLAR_STATUS)
        else:
            state.show_edges = True
            state.minkowski = False
            state.retrieval_k = 3
            apply_graph(graphs["grafos"], status=INITIAL_STATUS)

    @state.change("proj_mode", "camera_w", "transform_kind", "rot_xw_deg", "rot_yw_deg",
                  "rot_zw_deg", "boost_x", "boost_y", "boost_z", "t_filter", "use_t_filter",
                  "retrieval_k", "retrieval_hops", "scale_mode", "moon_extent", "size_boost",
                  "show_trails", "trail_span", "show_edges", "show_moon_labels", "minkowski",
                  "observer_id", "minkowski_span")
    def _on_view(**_):
        do_render()

    @state.change("scale_mode", "minkowski")
    def _on_camera(**_):
        reset_view()

    @state.change("edit_mode")
    def _on_edit_mode(**_):
        state.edge_first = None
        do_render()

    @state.change("sel_x", "sel_y", "sel_z", "sel_w")
    def _on_selected_coords(**_):
        node = current["db"].nodes.get(state.selected_node)
        if node is None or node.body is not None:
            return  # corpos andam pela órbita, não por sliders
        new = tuple(float(v) for v in (state.sel_x, state.sel_y, state.sel_z, state.sel_w))
        if new == node.as_tuple():
            return  # mudança vinda da própria seleção, nada a refazer
        current["db"].set_coords(node.id, new)
        state.sel_split = node.split
        refresh_stats()
        do_render()

    # -- ações -------------------------------------------------------------

    @ctrl.set("add_node")
    def add_node():
        nid = (state.new_node_id or "").strip()
        if not nid:
            set_status("Informe um ID.", "warning")
            return
        coords = (state.new_node_x, state.new_node_y, state.new_node_z, state.new_node_w)
        text = (state.new_node_text or "").strip()
        if not current["db"].add_node(nid, coords=coords, label=0, text=text):
            set_status(f"ID '{nid}' já existe.", "warning")
            return
        state.new_node_id = ""
        state.new_node_text = ""
        refresh_stats()
        set_status(f"Nó '{nid}' criado com w = {state.new_node_w}.", "success")
        do_render()

    @ctrl.set("remove_selected")
    def remove_selected():
        nid = state.selected_node
        if not nid:
            set_status("Nenhum nó selecionado.", "warning")
            return
        current["db"].remove_node(nid)
        state.selected_node = None
        state.edge_first = None
        refresh_stats()
        set_status(f"Nó '{nid}' removido.", "success")
        do_render()

    @ctrl.set("regenerate")
    def regenerate():
        if state.app_mode == "solar":
            reload_solar()
            return
        state.retrieval_active = False
        state.retrieval_rows, state.retrieval_context = [], ""
        if state.preset == "knowledge":
            graphs["grafos"] = generate_knowledge_demo()
            msg = "Demo carregada. Use Buscar e mexa em t."
            state.retrieval_query = KNOWLEDGE_DEMO_QUERY
        elif state.preset == "tesseract":
            graphs["grafos"], msg = generate_tesseract(), "Tesseract carregado."
        elif state.preset == "simplex4":
            graphs["grafos"], msg = generate_simplex4(), "4-simplex carregado."
        else:
            graphs["grafos"] = generate_random(
                n_nodes=int(state.gen_n_nodes), n_edges=int(state.gen_n_edges),
                n_steps=int(state.gen_n_steps), seed=int(state.gen_seed),
            )
            msg = "Grafo aleatório gerado."
        current["db"] = graphs["grafos"]
        state.selected_node = None
        state.edge_first = None
        state.t_filter = float(current["db"].t_max)
        refresh_stats()
        set_status(msg, "success")
        do_render()

    @ctrl.set("reload_solar")
    def reload_solar():
        graphs["solar"] = generate_solar_system(dynamics=state.dynamics)
        if state.app_mode == "solar":
            apply_graph(graphs["solar"], status=f"Sistema solar recalculado ({state.dynamics}).")

    @ctrl.set("run_retrieval")
    def run_retrieval():
        if not (state.retrieval_query or "").strip():
            set_status("Escreva uma consulta.", "warning")
            return
        state.retrieval_active = True
        do_render()
        result = current["retrieval"]
        if result.hits:
            set_status(f"Busca em t = {result.t:g}: {len(result.seeds())} semente(s), "
                       f"{len(result.hits)} nó(s) no contexto.", "success")
        else:
            set_status("Nenhum nó visível em t tem palavras em comum com a consulta.", "warning")

    @ctrl.set("clear_retrieval")
    def clear_retrieval():
        state.retrieval_active = False
        state.retrieval_rows, state.retrieval_context = [], ""
        set_status(SOLAR_STATUS if current["db"].has_bodies else INITIAL_STATUS)
        do_render()

    @ctrl.set("clear_selection")
    def clear_selection():
        state.selected_node = None
        state.edge_first = None
        state.sel_is_body = False
        do_render()

    @ctrl.set("reset_camera")
    def reset_camera():
        reset_view()

    @ctrl.set("reset_rotations")
    def reset_rotations():
        state.rot_xw_deg = state.rot_yw_deg = state.rot_zw_deg = 0.0
        state.boost_x = state.boost_y = state.boost_z = 0.0
        do_render()

    from .ui import build_layout  # noqa: PLC0415 - evita importar o Trame nos testes de lógica

    build_layout(server, plotter)
    do_render()
    return server


def main() -> None:
    create_app().start()


if __name__ == "__main__":
    main()
