"""Layout Trame/Vuetify. Só descreve a interface; a lógica fica em app.py.

Dois modos na mesma aplicação, escolhidos na barra superior:
  grafos  grafos 4D genéricos: edição, presets, snapshot, projeções
  solar   sistema solar: corpos com trajetória, escala, trilhas e relatividade
"""

from __future__ import annotations

from trame.ui.vuetify3 import VAppLayout
from trame.widgets import html
from trame.widgets import vuetify3 as vuetify

try:
    from trame_pyvista.ui import plotter_ui
except ImportError:  # versões antigas do pyvista traziam isto embutido
    from pyvista.trame.ui import plotter_ui

from .celestial.catalog import KIND_COLORS
from .renderer import (
    EXPIRED_EDGE_COLOR,
    NEIGHBOR_COLOR,
    SEED_COLOR,
    SPLIT_COLORS,
)

DRAWER_WIDTH = 360
GRAFOS = "app_mode === 'grafos'"
SOLAR = "app_mode === 'solar'"

PROJECTION_ITEMS = [
    {"title": "Só espaço (descarta w)", "value": "drop_w"},
    {"title": "Vê o tempo (descarta x)", "value": "drop_x"},
    {"title": "Vê o tempo (descarta y)", "value": "drop_y"},
    {"title": "Vê o tempo (descarta z)", "value": "drop_z"},
    {"title": "Perspectiva ao longo do tempo", "value": "perspective"},
]
PRESET_ITEMS = [
    {"title": "Base de conhecimento (demo de busca)", "value": "knowledge"},
    {"title": "Aleatório", "value": "random"},
    {"title": "Tesseract", "value": "tesseract"},
    {"title": "4-simplex", "value": "simplex4"},
]
SCALE_ITEMS = [
    {"title": "Comprimida (logarítmica)", "value": "log"},
    {"title": "Comprimida (raiz quadrada)", "value": "sqrt"},
    {"title": "Real (linear)", "value": "real"},
]
DYNAMICS_ITEMS = [
    {"title": "Kepler (analítico)", "value": "kepler"},
    {"title": "N-corpos (integrado, planetas)", "value": "nbody"},
]
TRANSFORM_ITEMS = [
    {"title": "Rotação euclidiana", "value": "rotation"},
    {"title": "Boost de Lorentz", "value": "lorentz"},
]


def _section(title: str) -> None:
    vuetify.VListSubheader(title, classes="px-0 text-subtitle-2")


def _note(text: str, **attrs) -> None:
    html.Div(text, classes="text-caption text-medium-emphasis mb-2", **attrs)


def _slider(label, key, default, lo, hi, step, **opts):
    opts.setdefault("thumb_label", True)
    return vuetify.VSlider(
        label=label, v_model=(key, default), min=lo, max=hi, step=step,
        density="compact", hide_details=True, **opts,
    )


def _select(label, key, default, items_key, items, **attrs):
    return vuetify.VSelect(
        label=label, v_model=(key, default), items=(items_key, items),
        item_title="title", item_value="value",
        density="compact", variant="outlined", hide_details=True, classes="mb-2", **attrs,
    )


def _check(label, key, default, **attrs):
    return vuetify.VCheckbox(label=label, v_model=(key, default), density="compact",
                             hide_details=True, **attrs)


def _axis_sliders(prefix: str, step: float) -> None:
    for axis in "xyz":
        _slider(axis, f"{prefix}_{axis}", 0.0, -3, 3, step)
    _slider("w (tempo)", f"{prefix}_w", 0.0, 0, ("db_t_max", 6), step)


def _time_section() -> None:
    _section("Tempo (eixo w)")
    _check("Filtrar por instante t (snapshot)", "use_t_filter", True, v_if=GRAFOS)
    with vuetify.VCard(v_if="use_t_filter", variant="outlined", classes="mb-2 pa-2"):
        _slider("Instante t", "t_filter", 6, 0, ("db_t_max", 6), ("t_step", 0.5),
                thumb_label="always", color="primary")
        html.Div("{{ time_text }}", v_if="time_text", classes="text-caption font-weight-bold")
        _note("Grafos: só nós com w <= t e arestas ativas em t.", v_if=GRAFOS)
        _note("Sistema solar: o instante em que os corpos são mostrados (dias desde J2000).",
              v_if=SOLAR)


def _solar_section(ctrl) -> None:
    _section("Sistema solar")
    _note("Planetas e Lua: elementos médios, validados contra a efeméride do astropy. "
          "Demais corpos: forma real, fase ilustrativa.")
    _select("Dinâmica", "dynamics", "kepler", "dynamics_items", DYNAMICS_ITEMS)
    vuetify.VBtn("Recalcular", click=ctrl.reload_solar, size="small", variant="tonal",
                 block=True, classes="mb-3", prepend_icon="mdi-orbit")

    _select("Escala de exibição", "scale_mode", "log", "scale_items", SCALE_ITEMS)
    _slider("Afastamento das luas", "moon_extent", 0.08, 0.02, 0.3, 0.01,
            v_if="scale_mode !== 'real'")
    _slider("Tamanho dos corpos", "size_boost", 1.0, 0.5, 3.0, 0.1)
    _check("Trilhas (linhas de mundo)", "show_trails", False)
    _slider("Duração das trilhas (dias)", "trail_span", 365, 1, 3650, 1, v_if="show_trails")
    _check("Rótulos das luas", "show_moon_labels", False)
    _check("Mostrar arestas", "show_edges", False)


def _projection_section() -> None:
    _section("Projeção R⁴ para R³")
    _select("Modo", "proj_mode", "drop_w", "proj_items", PROJECTION_ITEMS)
    with vuetify.VCard(v_if="proj_mode === 'perspective'", variant="outlined", classes="mb-2 pa-2"):
        _slider("Câmera em w", "camera_w", 8.0, 2, 20, 0.5)


def _transform_section(ctrl) -> None:
    _section("Transformação que mistura espaço e tempo")
    _select("Tipo", "transform_kind", "rotation", "transform_items", TRANSFORM_ITEMS)
    with html.Div(v_if="transform_kind === 'rotation'"):
        for plane, label in (("xw", "x-tempo"), ("yw", "y-tempo"), ("zw", "z-tempo")):
            _slider(f"Plano {label} (graus)", f"rot_{plane}_deg", 0, -180, 180, 1)
    with html.Div(v_if="transform_kind === 'lorentz'"):
        _note("Boost com velocidade v = beta·c ao longo do eixo (w = c·t).")
        for axis in "xyz":
            _slider(f"beta em {axis}", f"boost_{axis}", 0.0, -0.95, 0.95, 0.05)
    vuetify.VBtn("Zerar", click=ctrl.reset_rotations, size="small", variant="tonal",
                 block=True, classes="my-2")


def _relativity_section() -> None:
    _section("Relatividade restrita")
    _check("Vista Minkowski (cone de luz)", "minkowski", False)
    _note("Mostra o cone de luz do evento do observador. No sistema solar, w = c·t em UA e "
          "os pontos brancos marcam onde cada corpo estava quando saiu a luz que chega agora.",
          v_if="minkowski")
    with html.Div(v_if="minkowski && graph_has_bodies"):
        _select("Observador", "observer_id", "Terra", "body_items", [])
        _slider("Janela de tempo (dias)", "minkowski_span", 0.25, 0.05, 2.0, 0.05)


def _legend_row(items: dict[str, str]) -> None:
    with vuetify.VRow(dense=True, no_gutters=True):
        for name, color in items.items():
            with vuetify.VCol(cols=6), html.Div(classes="d-flex align-center"):
                html.Div(style=f"width:12px;height:12px;border-radius:50%;"
                               f"background:{color};margin-right:6px;")
                html.Span(name, classes="text-caption")


def _legend() -> None:
    with vuetify.VCard(variant="tonal", classes="mb-3 pa-2"):
        with html.Div(v_if=GRAFOS):
            html.Div("Cor do nó = divisão pelo tempo de nascimento",
                     classes="text-caption font-weight-bold mb-1")
            _legend_row(SPLIT_COLORS)
        with html.Div(v_if=SOLAR):
            html.Div("Cor do nó = tipo de corpo", classes="text-caption font-weight-bold mb-1")
            _legend_row(KIND_COLORS)
        html.Div("Busca e arestas", classes="text-caption font-weight-bold mt-2 mb-1")
        _legend_row({
            "semente da busca": SEED_COLOR,
            "vizinho da busca": NEIGHBOR_COLOR,
            "aresta expirada": EXPIRED_EDGE_COLOR,
            "aresta no contexto": "#FFFFFF",
        })


def _retrieval_section(ctrl) -> None:
    _section("Recuperação (GraphRAG didático)")
    _note("Busca no grafo como ele era em t (o slider de tempo vale aqui).")
    vuetify.VTextField(
        label="Consulta", v_model=("retrieval_query", ""), density="compact",
        variant="outlined", hide_details=True, classes="mb-2",
        __events=[("enter", "keyup.enter")], enter=ctrl.run_retrieval,
    )
    _slider("Sementes (k)", "retrieval_k", 3, 1, 8, 1)
    _slider("Saltos (hops)", "retrieval_hops", 1, 0, 3, 1)
    with vuetify.VRow(dense=True, classes="my-1"):
        with vuetify.VCol(cols=6):
            vuetify.VBtn("Buscar", click=ctrl.run_retrieval, color="primary", block=True,
                         size="small", prepend_icon="mdi-magnify")
        with vuetify.VCol(cols=6):
            vuetify.VBtn("Limpar", click=ctrl.clear_retrieval, variant="tonal", block=True,
                         size="small")
    with vuetify.VCard(v_if="retrieval_active", variant="outlined", classes="pa-2 mb-2"):
        html.Div("Resultado em t = {{ retrieval_t }}", classes="text-caption font-weight-bold")
        html.Div("{{ row }}", v_for="(row, i) in retrieval_rows", key="i", classes="text-caption")
        html.Div("Nenhum nó em comum com a consulta neste instante.",
                 v_if="!retrieval_rows.length", classes="text-caption text-medium-emphasis")
        html.Pre("{{ retrieval_context }}", v_if="retrieval_context", classes="text-caption mt-2",
                 style="white-space: pre-wrap; font-family: inherit;")


def _selection_info(ctrl) -> None:
    with vuetify.VCard(v_if="selected_node && sel_is_body", variant="outlined", classes="mb-3"):
        vuetify.VCardTitle("{{ selected_node }}", classes="text-subtitle-2 py-2")
        with vuetify.VCardText(classes="pt-0"):
            html.Div("{{ sel_text }}", v_if="sel_text", classes="text-caption mb-1")
            html.Div("{{ sel_info }}", classes="text-caption")
            vuetify.VBtn("Limpar seleção", click=ctrl.clear_selection, size="x-small",
                         variant="text")


def _edit_section(ctrl) -> None:
    _section("Edição")
    with vuetify.VBtnToggle(v_model=("edit_mode", "select"), mandatory=True,
                            density="compact", color="primary", classes="mb-2", divided=True):
        vuetify.VBtn("Selecionar", value="select", size="small")
        vuetify.VBtn("Conectar", value="connect", size="small")
    with vuetify.VCard(v_if="edit_mode === 'connect'", variant="outlined", classes="mb-2 pa-2"):
        _slider("Início da aresta", "pending_edge_t", 0, 0, ("db_t_max", 6), 0.5)
        _check("Com fim (valid_to)", "edge_has_end", False)
        _slider("Fim da aresta", "pending_edge_t_end", 6, 0, ("db_t_max", 6), 0.5,
                v_if="edge_has_end")
        vuetify.VTextField(label="Relação (opcional)", v_model=("pending_edge_relation", ""),
                           density="compact", variant="outlined", hide_details=True,
                           classes="mt-2")
    vuetify.VChip(
        "1º nó: {{ edge_first }}", v_if="edge_first", color="warning", variant="tonal",
        size="small", closable=True, click_close=ctrl.clear_selection, classes="mb-2",
    )

    vuetify.VDivider(classes="my-2")
    _section("Novo nó")
    vuetify.VTextField(label="ID", v_model=("new_node_id", ""), density="compact",
                       variant="outlined", hide_details=True, classes="mb-1")
    vuetify.VTextField(label="Texto (usado na busca)", v_model=("new_node_text", ""),
                       density="compact", variant="outlined", hide_details=True, classes="mb-1")
    _axis_sliders("new_node", 0.1)
    vuetify.VBtn("Criar nó", click=ctrl.add_node, color="primary", block=True,
                 prepend_icon="mdi-plus", classes="my-2")

    with vuetify.VCard(v_if="selected_node && !sel_is_body", variant="outlined", classes="mb-2"):
        vuetify.VCardTitle("Editando: {{ selected_node }}", classes="text-subtitle-2 py-2")
        with vuetify.VCardText(classes="pt-0"):
            html.Div("divisão: {{ sel_split }}  |  rótulo: {{ sel_label }}",
                     classes="text-caption mb-2")
            html.Div("{{ sel_text }}", v_if="sel_text", classes="text-caption mb-2")
            _axis_sliders("sel", 0.05)
            vuetify.VBtn("Remover nó", click=ctrl.remove_selected, color="error", variant="tonal",
                         block=True, prepend_icon="mdi-delete", classes="mt-2")


def _presets(ctrl) -> None:
    with vuetify.VExpansionPanels(variant="accordion"), vuetify.VExpansionPanel():
        vuetify.VExpansionPanelTitle("Presets e gerador")
        with vuetify.VExpansionPanelText():
            _select("Preset", "preset", "random", "preset_items", PRESET_ITEMS)
            with vuetify.VCard(v_if="preset === 'random'", variant="outlined",
                               classes="pa-2 mb-2"):
                _slider("Nós", "gen_n_nodes", 15, 3, 40, 1)
                _slider("Arestas", "gen_n_edges", 25, 0, 80, 1)
                _slider("Horizonte t_max", "gen_n_steps", 6, 2, 20, 1)
                vuetify.VNumberInput(label="Seed", v_model=("gen_seed", 42), density="compact",
                                     variant="outlined", hide_details=True)
            vuetify.VBtn("Carregar", click=ctrl.regenerate, color="secondary", block=True,
                         prepend_icon="mdi-dice-multiple")


def build_layout(server, plotter):
    """Monta a página e devolve a view do PyVista."""
    ctrl = server.controller
    with VAppLayout(server) as layout:
        with vuetify.VAppBar(density="compact"):
            vuetify.VToolbarTitle("Grafo espaço-tempo (R⁴)")
            vuetify.VSpacer()
            with vuetify.VBtnToggle(v_model=("app_mode", "grafos"), mandatory=True,
                                    density="compact", color="primary", variant="outlined",
                                    divided=True, classes="mr-3"):
                vuetify.VBtn("Grafos 4D", value="grafos", size="small")
                vuetify.VBtn("Sistema solar", value="solar", size="small")
            vuetify.VChip("{{ db_nodes }} nós, {{ db_edges }} arestas", color="primary",
                          variant="tonal", size="small", classes="ma-1")
            vuetify.VBtn(icon="mdi-fit-to-screen", click=ctrl.reset_camera, density="comfortable")

        with (
            vuetify.VNavigationDrawer(permanent=True, width=DRAWER_WIDTH),
            vuetify.VContainer(classes="pa-3"),
        ):
            vuetify.VAlert("{{ status_msg }}", density="compact", variant="tonal",
                           type=("status_type", "info"), classes="mb-3 text-caption")
            _time_section()
            with html.Div(v_if=SOLAR):
                _solar_section(ctrl)
                vuetify.VDivider(classes="my-2")
            _projection_section()
            _transform_section(ctrl)
            _relativity_section()
            _legend()
            _selection_info(ctrl)
            vuetify.VDivider(classes="my-2")
            _retrieval_section(ctrl)
            with html.Div(v_if=GRAFOS):
                vuetify.VDivider(classes="my-2")
                _edit_section(ctrl)
                vuetify.VDivider(classes="my-2")
                _presets(ctrl)

        with vuetify.VMain(), vuetify.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
            view = plotter_ui(plotter, mode="trame")
            ctrl.view_update = view.update
    return layout
