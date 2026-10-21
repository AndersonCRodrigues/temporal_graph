"""Quadro de cena: o que o renderer desenha, já em coordenadas de cena.

É um contêiner neutro. O núcleo preenche um quadro simples para grafos
estáticos (`static_frame`); o subpacote `celestial` preenche um para corpos em
movimento. O renderer só consome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Graph4D


@dataclass
class ConeSpec:
    """Cone de luz de um evento: |dx| = |dw|, fatia z = z0."""

    center: np.ndarray  # (4,)
    radius: float  # raio (e meia altura) das superfícies desenhadas


@dataclass
class Frame:
    pos: dict[str, np.ndarray] = field(default_factory=dict)  # id -> (4,)
    radius: dict[str, float] = field(default_factory=dict)  # id -> raio da esfera
    color: dict[str, str] = field(default_factory=dict)  # id -> cor
    trails: dict[str, np.ndarray] = field(default_factory=dict)  # id -> (M, 4)
    markers: dict[str, np.ndarray] = field(default_factory=dict)  # id -> (4,)
    cone: ConeSpec | None = None
    visible: set[str] | None = None  # se definido, só estes nós são desenhados
    title: str = ""
    footer: str = ""


def static_frame(
    graph: Graph4D, ids: list[str], *, observer: str | None = None, cone: bool = False,
) -> Frame:
    """Quadro para nós estáticos. Com `cone`, centra tudo no evento do observador.

    As coordenadas passam a ser relativas ao evento do observador (ou à origem)
    e o cone de luz desse evento é desenhado. Pressupõe w = c*t com c = 1.
    """
    frame = Frame()
    origin = graph.nodes[observer].coords if observer in graph.nodes else np.zeros(4)
    for nid in ids:
        frame.pos[nid] = graph.nodes[nid].coords - (origin if cone else 0.0)
    if cone:
        extent = max((float(np.max(np.abs(p))) for p in frame.pos.values()), default=1.0)
        frame.cone = ConeSpec(np.zeros(4), max(1.0, 1.1 * extent))
        frame.title = "Minkowski: cone de luz do evento selecionado (c = 1)"
    return frame


def cone_surfaces(cone: ConeSpec, n_r: int = 10, n_theta: int = 48):
    """Superfícies do cone (passado, futuro), cada uma (n_r, n_theta, 4) em 4D.

    Fatia z = z0 do cone: pontos (x0 + r cos a, y0 + r sen a, z0, w0 -/+ r).
    """
    r = np.linspace(0.0, cone.radius, n_r)[:, None]
    a = np.linspace(0.0, 2 * np.pi, n_theta + 1)[None, :]
    base = np.zeros((n_r, n_theta + 1, 4))
    base[..., 0] = cone.center[0] + r * np.cos(a)
    base[..., 1] = cone.center[1] + r * np.sin(a)
    base[..., 2] = cone.center[2]
    past, future = base.copy(), base.copy()
    past[..., 3] = cone.center[3] - r
    future[..., 3] = cone.center[3] + r
    return past, future
