"""Quadro de cena do sistema solar: posição atual, trilhas e vista Minkowski.

Dois regimes:

  comprimido (padrão)  posições hierárquicas e comprimidas (ver scale.py); o
                       tempo w é relativo ao instante atual: w = 0 para o
                       "agora" e negativo nas trilhas, em `w_extent` unidades
                       para `trail_span` dias.
  Minkowski            posições físicas em UA relativas ao observador, w = c*t
                       em UA relativo ao evento "agora" do observador. Mostra o
                       cone de luz passado e os eventos retardados: onde cada
                       corpo está sobre a linha de mundo no instante em que
                       saiu a luz que chega ao observador agora.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frame import ConeSpec, Frame
from ..model import Graph4D
from ..relativity import retarded_time
from .scale import DisplayScale, body_radius, display_position

W_EXTENT = 4.0
MINKOWSKI_SPHERE = 0.012  # fração do raio do cone


@dataclass
class SceneOptions:
    scale: DisplayScale = field(default_factory=DisplayScale)
    trails: bool = False
    trail_span: float = 365.0  # dias
    trail_samples: int = 96
    minkowski: bool = False
    observer: str | None = "Terra"
    minkowski_span: float = 0.25  # dias; altura do cone e das trilhas


def _time_samples(node, t: float, span: float, n: int) -> np.ndarray | None:
    ts = np.linspace(t - span, t, n)
    ts = ts[ts >= node.w]
    return ts if len(ts) >= 2 else None


def build_frame(graph: Graph4D, t: float, opts: SceneOptions) -> Frame:
    """Quadro do grafo com corpos no instante t."""
    if opts.minkowski:
        return _minkowski_frame(graph, t, opts)
    frame = Frame(title=f"{graph.time_label(t)}  |  escala {opts.scale.mode}")
    for nid, node in graph.nodes.items():
        body = node.body
        if body is None:
            frame.pos[nid] = node.coords
            continue
        frame.pos[nid] = np.array([*display_position(graph, nid, t, opts.scale), 0.0])
        frame.radius[nid] = body_radius(graph, nid, opts.scale)
        if body.color:
            frame.color[nid] = body.color
        if not opts.trails or body.parent is None:
            continue
        period = getattr(body.orbit, "period_days", None)
        span = min(opts.trail_span, period) if period else opts.trail_span
        ts = _time_samples(node, t, span, opts.trail_samples)
        if ts is None:
            continue
        xyz = display_position(graph, nid, ts, opts.scale)
        w = (ts - t) / opts.trail_span * W_EXTENT
        frame.trails[nid] = np.column_stack([xyz, w])
    return frame


def _minkowski_frame(graph: Graph4D, t: float, opts: SceneOptions) -> Frame:
    bodies = [n for n, node in graph.nodes.items() if node.body is not None]
    observer = opts.observer if opts.observer in bodies else (bodies[0] if bodies else None)
    c = graph.light_speed
    origin = graph.position_at(observer, t) if observer else np.zeros(3)
    span = opts.minkowski_span
    cone_radius = c * span
    frame = Frame(
        title=f"Minkowski, observador {observer}  |  {graph.time_label(t)}  |  w = c·t (UA)",
        cone=ConeSpec(np.zeros(4), cone_radius),
    )
    sphere = cone_radius * MINKOWSKI_SPHERE
    frame.visible = set()
    for nid, node in graph.nodes.items():
        if node.body is None:
            frame.pos[nid] = node.coords
            frame.visible.add(nid)
            continue
        if np.linalg.norm(graph.position_at(nid, t) - origin) > 1.3 * cone_radius:
            continue  # fora do alcance do cone: não entra na vista
        frame.visible.add(nid)
        frame.pos[nid] = np.array([*(graph.position_at(nid, t) - origin), 0.0])
        frame.radius[nid] = sphere * (1.8 if node.body.parent is None else 1.0)
        if node.body.color:
            frame.color[nid] = node.body.color
        ts = _time_samples(node, t, span, opts.trail_samples)
        if ts is not None:
            xyz = graph.position_at(nid, ts) - origin
            frame.trails[nid] = np.column_stack([xyz, c * (ts - t)])
        if nid != observer:
            t_ret = retarded_time(graph, nid, observer, t)
            if t_ret >= node.w:
                frame.markers[nid] = np.array(
                    [*(graph.position_at(nid, t_ret) - origin), c * (t_ret - t)]
                )
    if observer and "Sol" in graph.nodes and observer != "Sol":
        minutes = np.linalg.norm(graph.position_at("Sol", t) - origin) / c * 24 * 60
        frame.footer = f"luz do Sol até {observer}: {minutes:.1f} min"
    return frame
