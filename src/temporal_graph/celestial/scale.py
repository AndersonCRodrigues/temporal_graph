"""Escala de exibição: comprime distâncias e tamanhos para caber numa cena.

Distâncias reais não cabem numa tela: Netuno está a 30 UA e a Lua a 0,0026 UA
da Terra. A posição na cena é hierárquica: cada corpo fica ao redor da posição
(já comprimida) do pai, na mesma direção da posição física relativa ao pai,
com distância comprimida em duas escalas.

  heliocêntrico (pai é o Sol)  log, raiz ou linear sobre a distância em UA
  luas (pai é um planeta)      afastamento mínimo (raios do pai e do corpo)
                               mais uma escala logarítmica em km
  tamanhos                     logarítmicos; no modo "real", proporcionais

As direções e as ordens de grandeza se preservam. Razões de distância, não.
O modo "real" é linear em tudo e mostra o quanto o sistema solar é vazio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import Graph4D
from .orbits import AU_KM

SCALE_MODES = ("log", "sqrt", "real")
HELIO_D0_AU = 0.05
MOON_D0_KM = 1000.0
MIN_RADIUS = 0.025


@dataclass
class DisplayScale:
    mode: str = "log"  # "log" | "sqrt" | "real"
    helio_extent: float = 3.0  # unidades de cena por década de distância heliocêntrica
    moon_extent: float = 0.08  # idem, por década de distância (km) das luas
    size_boost: float = 1.0
    real_factor: float = 0.25  # unidades de cena por UA no modo real

    def __post_init__(self) -> None:
        if self.mode not in SCALE_MODES:
            raise ValueError(f"modo de escala desconhecido: {self.mode!r}")


def visual_radius(radius_km: float, scale: DisplayScale) -> float:
    """Raio da esfera na cena."""
    if scale.mode == "real":
        r = radius_km / AU_KM * scale.real_factor
    else:
        r = 0.03 + 0.07 * max(0.0, np.log10(max(radius_km, 1.0) / 100.0))
    return max(MIN_RADIUS, r * scale.size_boost)


def helio_distance(d_au, scale: DisplayScale):
    """Distância heliocêntrica na cena a partir da distância em UA."""
    d = np.asarray(d_au, dtype=float)
    if scale.mode == "real":
        return scale.real_factor * d
    if scale.mode == "sqrt":
        return scale.helio_extent * np.sqrt(d) / 2.0
    return scale.helio_extent * np.log10(1.0 + d / HELIO_D0_AU)


def moon_distance(d_au, parent_radius: float, own_radius: float, scale: DisplayScale):
    """Distância de uma lua ao planeta na cena."""
    d = np.asarray(d_au, dtype=float)
    if scale.mode == "real":
        return scale.real_factor * d
    return parent_radius + own_radius + scale.moon_extent * np.log10(1.0 + d * AU_KM / MOON_D0_KM)


def body_radius(graph: Graph4D, nid: str, scale: DisplayScale) -> float:
    body = graph.nodes[nid].body
    return visual_radius(body.radius_km, scale) if body else MIN_RADIUS


def display_position(graph: Graph4D, nid: str, t, scale: DisplayScale) -> np.ndarray:
    """Posição (..., 3) do corpo na cena em t (escalar ou array)."""
    t = np.asarray(t, dtype=float)
    node = graph.nodes[nid]
    body = node.body
    if body is None:
        return np.broadcast_to(node.coords[:3], (*t.shape, 3)).copy()
    if body.parent is None:
        return np.zeros((*t.shape, 3))
    local = graph.local_position_at(nid, t)
    dist = np.linalg.norm(local, axis=-1, keepdims=True)
    unit = np.divide(local, dist, out=np.zeros_like(local), where=dist > 0)
    parent = graph.nodes[body.parent]
    parent_is_center = parent.body is not None and parent.body.parent is None
    if parent_is_center:
        shown = helio_distance(dist, scale)
    else:
        shown = moon_distance(
            dist, body_radius(graph, body.parent, scale), body_radius(graph, nid, scale), scale
        )
    return display_position(graph, body.parent, t, scale) + unit * shown
