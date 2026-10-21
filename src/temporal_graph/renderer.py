"""Cena PyVista do grafo espaço-tempo: (x, y, z) espaço, w tempo."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyvista as pv

from .frame import Frame, cone_surfaces
from .model import Graph4D
from .projection import DEFAULT_CAMERA_W, project_points
from .retrieval import RetrievalResult

SPLIT_COLORS = {"train": "#4C9BE8", "val": "#E8B84C", "test": "#E85C4C"}
EDGE_COLOR = "#888888"
EXPIRED_EDGE_COLOR = "#8a4b4b"
RETRIEVED_EDGE_COLOR = "#FFFFFF"
SEED_COLOR = "#4CE88B"
NEIGHBOR_COLOR = "#4CD8E8"
BACKGROUND = "#1a1a2e"
FIRST_NODE_COLOR = "#FFD700"
HIGHLIGHT_COLOR = "#FFFFFF"
TRAIL_COLOR = "#6c8ebf"
MARKER_COLOR = "#FFFFFF"
CONE_PAST_COLOR = "#4C9BE8"
CONE_FUTURE_COLOR = "#E8B84C"

NODE_RADIUS = 0.14
SEED_RADIUS = 0.2
FIRST_NODE_RADIUS = 0.18
HIGHLIGHT_RADIUS = 0.28
DIMMED_OPACITY = 0.3


def mode_description(proj_mode: str, camera_w: float) -> str:
    return {
        "drop_w": "Espaço (x, y, z), tempo w descartado",
        "drop_x": "Descarta x: vê y, z e o tempo",
        "drop_y": "Descarta y: vê x, z e o tempo",
        "drop_z": "Descarta z: vê x, y e o tempo",
        "perspective": f"Perspectiva ao longo do tempo (w_cam = {camera_w:.1f})",
    }.get(proj_mode, proj_mode)


class SceneRenderer:
    def __init__(self, plotter: pv.Plotter | None = None):
        self.plotter = plotter or pv.Plotter()
        self.plotter.background_color = BACKGROUND
        self.plotter.add_axes()
        self.actor_to_node: dict[object, str] = {}
        self._on_pick: Callable[[str], None] | None = None

    def set_pick_callback(self, cb: Callable[[str], None] | None) -> None:
        self._on_pick = cb

    def clear(self) -> None:
        self.plotter.clear()
        self.actor_to_node.clear()

    def _text(self, msg: str, position: str, size: int = 12, color: str = "white") -> None:
        self.plotter.add_text(msg, position=position, font_size=size, color=color)

    def render(
        self,
        db: Graph4D,
        *,
        proj_mode: str = "drop_w",
        camera_w: float = DEFAULT_CAMERA_W,
        rot_xw: float = 0.0,
        rot_yw: float = 0.0,
        rot_zw: float = 0.0,
        t_filter: float | None = None,
        selected_node: str | None = None,
        edge_first: str | None = None,
        show_labels: bool = True,
        retrieval: RetrievalResult | None = None,
        boosts: list[tuple[str, float]] | None = None,
        frame: Frame | None = None,
        label_ids: set[str] | None = None,
        show_edges: bool = True,
    ) -> None:
        """Redesenha a cena.

        t_filter: se definido, mostra só nós com w <= t_filter e arestas já
        começadas. Arestas ativas aparecem em cinza e as que já terminaram em
        vermelho escuro. O filtro usa o w original, antes das rotações.

        retrieval: se definido, destaca sementes e vizinhos da busca e esmaece
        o resto.

        frame: posições, raios, cores, trilhas, marcadores e cone já em
        coordenadas de cena. Sem ele, valem as coordenadas dos nós.
        boosts: boosts de Lorentz (eixo, beta), aplicados depois das rotações.
        label_ids: se definido, só esses nós ganham rótulo (mais os da busca).
        show_edges: desenha ou não as arestas.
        """
        frame = frame or Frame()
        self.clear()
        if not db.nodes:
            self._text("Grafo vazio", "upper_left", 14)
            return

        ids = sorted(
            n.id for n in db.nodes.values()
            if (t_filter is None or n.w <= t_filter)
            and (frame.visible is None or n.id in frame.visible)
        )
        if not ids:
            self._text(f"Nada visível em t <= {t_filter:g}", "upper_left", 14)
            return
        visible = set(ids)

        active, expired = [], []
        for e in db.edges:
            if e.src not in visible or e.dst not in visible:
                continue
            if t_filter is None or e.is_active(t_filter):
                active.append(e)
            elif e.t <= t_filter:
                expired.append(e)

        rotations = [(p, a) for p, a in (("xw", rot_xw), ("yw", rot_yw), ("zw", rot_zw)) if a]
        def project(points4: np.ndarray) -> np.ndarray:
            flat = points4.reshape(-1, 4)
            out = project_points(flat, mode=proj_mode, camera_w=camera_w,
                                 rotations=rotations, boosts=boosts)
            return out.reshape(*points4.shape[:-1], 3)

        pts4 = np.array([frame.pos.get(i, db.nodes[i].coords) for i in ids])
        pts3 = project(pts4)
        index = {nid: k for k, nid in enumerate(ids)}

        hits = {h.id: h for h in retrieval.hits} if retrieval else {}
        picked_edges = {frozenset(p) for p in retrieval.edges} if retrieval else set()
        highlighted = [e for e in active if frozenset((e.src, e.dst)) in picked_edges]
        normal = [e for e in active if frozenset((e.src, e.dst)) not in picked_edges]

        if show_edges:
            self._draw_edges(pts3, index, expired, EXPIRED_EDGE_COLOR, width=2.0, opacity=0.6)
            self._draw_edges(pts3, index, normal, EDGE_COLOR, width=2.5,
                             opacity=0.35 if retrieval else 0.75)
            self._draw_edges(pts3, index, highlighted, RETRIEVED_EDGE_COLOR, width=5.0,
                             opacity=1.0)
        self._draw_nodes(db, ids, pts3, edge_first, hits, frame, dim=retrieval is not None)
        self._draw_extras(frame, ids, project)

        if show_labels:
            keep = [
                nid for nid in ids
                if label_ids is None or nid in label_ids or nid in hits or nid == selected_node
            ]
            if keep:
                labels = [
                    f"{nid} ({hits[nid].score:.2f})" if nid in hits and hits[nid].hop == 0 else nid
                    for nid in keep
                ]
                self.plotter.add_point_labels(
                    pts3[[index[nid] for nid in keep]], labels, font_size=12, text_color="white",
                    shape=None, always_visible=True, point_size=0,
                )
        if selected_node in index:
            base = frame.radius.get(selected_node, HIGHLIGHT_RADIUS / 2)
            self.plotter.add_mesh(
                pv.Sphere(radius=max(HIGHLIGHT_RADIUS, 1.8 * base),
                          center=pts3[index[selected_node]]),
                color=HIGHLIGHT_COLOR, opacity=0.25, pickable=False, name="highlight",
            )

        title = f"Espaço-tempo R⁴  |  {mode_description(proj_mode, camera_w)}"
        if boosts:
            title += "  |  boost de Lorentz"
        if frame.title:
            title += f"  |  {frame.title}"
        elif t_filter is not None:
            title += f"  |  snapshot t <= {t_filter:g}"
        self._text(title, "upper_left")
        total = db.stats()
        edge_info = f"arestas ativas: {len(active)}" if show_edges else "arestas ocultas"
        if show_edges and t_filter is not None:
            edge_info += f"  expiradas: {len(expired)}"
        self._text(
            f"nós: {len(ids)}/{total['n_nodes']}  |  {edge_info}", "lower_left", 11, "#aaaaaa"
        )
        if frame.footer:
            self._text(frame.footer, "lower_right", 11, "#aaaaaa")
        if retrieval is not None:
            self._text(
                f"busca: {len(retrieval.seeds())} semente(s), {len(hits)} nó(s) no contexto",
                "upper_right", 11, SEED_COLOR,
            )

        self.plotter.disable_picking()
        self.plotter.enable_mesh_picking(
            callback=self._handle_pick, use_actor=True, show=False, show_message=False
        )

    def _draw_edges(self, pts3, index, edges, color, *, width, opacity) -> None:
        if not edges:
            return
        mesh = pv.PolyData(pts3)
        mesh.lines = np.hstack([[2, index[e.src], index[e.dst]] for e in edges])
        self.plotter.add_mesh(mesh, color=color, line_width=width, opacity=opacity, pickable=False)

    def _draw_nodes(self, db, ids, pts3, edge_first, hits, frame, *, dim) -> None:
        for nid, pos in zip(ids, pts3, strict=True):
            hit = hits.get(nid)
            base = frame.radius.get(nid, NODE_RADIUS)
            if nid == edge_first:
                color, radius, opacity = FIRST_NODE_COLOR, max(base, FIRST_NODE_RADIUS), 1.0
            elif hit is not None:
                color = SEED_COLOR if hit.hop == 0 else NEIGHBOR_COLOR
                radius = max(base * 1.3, SEED_RADIUS) if hit.hop == 0 else base
                opacity = 1.0
            else:
                color = frame.color.get(nid) or SPLIT_COLORS.get(db.nodes[nid].split, EDGE_COLOR)
                radius, opacity = base, DIMMED_OPACITY if dim else 1.0
            actor = self.plotter.add_mesh(
                pv.Sphere(radius=radius, center=pos, theta_resolution=20, phi_resolution=20),
                color=color, opacity=opacity, pickable=True, name=f"node_{nid}",
                smooth_shading=True,
            )
            self.actor_to_node[actor] = nid

    def _draw_extras(self, frame: Frame, ids, project) -> None:
        """Trilhas, marcadores de eventos e cone de luz."""
        visible = set(ids)
        trails = [project(arr) for nid, arr in frame.trails.items() if nid in visible]
        if trails:
            lines, offset = [], 0
            for arr in trails:
                lines.extend([len(arr), *range(offset, offset + len(arr))])
                offset += len(arr)
            mesh = pv.PolyData(np.vstack(trails))
            mesh.lines = np.array(lines)
            self.plotter.add_mesh(mesh, color=TRAIL_COLOR, line_width=1.5, opacity=0.7,
                                  pickable=False)
        marks = [m for nid, m in frame.markers.items() if nid in visible]
        if marks:
            self.plotter.add_mesh(
                pv.PolyData(project(np.array(marks))), color=MARKER_COLOR, point_size=8,
                render_points_as_spheres=True, pickable=False,
            )
        if frame.cone is not None:
            for surface, color in zip(cone_surfaces(frame.cone),
                                      (CONE_PAST_COLOR, CONE_FUTURE_COLOR), strict=True):
                pts = project(surface)
                n_r, n_t = pts.shape[:2]
                idx = np.arange(n_r * n_t).reshape(n_r, n_t)
                quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1)
                faces = np.hstack([np.full((quads.size // 4, 1), 4), quads.reshape(-1, 4)])
                self.plotter.add_mesh(pv.PolyData(pts.reshape(-1, 3), faces.ravel()),
                                      color=color, opacity=0.13, pickable=False)

    def _handle_pick(self, actor) -> None:
        nid = self.actor_to_node.get(actor)
        if nid and self._on_pick:
            self._on_pick(nid)
