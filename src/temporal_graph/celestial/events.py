"""Eventos que se repetem: alinhamentos, eclipses candidatos e ressonâncias.

Cada ocorrência vira uma aresta com a sua própria janela [t - w, t + w). Como
o grafo aceita várias arestas por par, desde que as janelas da mesma relação
não se sobreponham, a relação "reaparece" a cada evento.

Os eclipses são CANDIDATOS do modelo kepleriano: sizígia (lua nova ou cheia)
com latitude eclíptica da Lua abaixo de um limite. As datas podem diferir das
reais em algumas horas (o modelo erra a longitude da Lua em até ~4 graus) e os
casos limítrofes podem entrar ou sair.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

from ..model import Graph4D

BISECTION_STEPS = 50


def _lon(v: np.ndarray) -> np.ndarray:
    return np.arctan2(v[..., 1], v[..., 0])


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _crossings(f, t0: float, t1: float, step: float) -> np.ndarray:
    """Instantes em que f (com saltos de ±pi por enrolamento) cruza o zero."""
    ts = np.arange(t0, t1 + step, step)
    v = f(ts)
    hit = np.where((v[:-1] * v[1:] < 0) & (np.abs(v[1:] - v[:-1]) < 1.0))[0]
    lo, hi = ts[hit], ts[hit + 1]
    flo = v[hit]
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi)


def _period(graph: Graph4D, nid: str) -> float:
    p = getattr(graph.nodes[nid].body.orbit, "period_days", None) if graph.nodes[nid].body else None
    return float(p) if p else 365.25


def find_alignments(
    graph: Graph4D, a: str, b: str, t0: float, t1: float, *,
    kind: str = "aligned", center: str | None = None, step: float | None = None,
) -> np.ndarray:
    """Instantes em que a e b têm a mesma longitude (aligned) ou opostas (opposed).

    As longitudes são medidas a partir de `center` (por padrão, o pai de a).
    Entre a Terra e um planeta externo, "aligned" é a oposição vista da Terra.
    """
    if kind not in ("aligned", "opposed"):
        raise ValueError("kind deve ser 'aligned' ou 'opposed'")
    body_a = graph.nodes[a].body
    center = center or (body_a.parent if body_a else None)
    shift = np.pi if kind == "opposed" else 0.0

    def rel(nid, t):
        p = graph.position_at(nid, t)
        return p - graph.position_at(center, t) if center else p

    def f(t):
        return _wrap(_lon(rel(b, t)) - _lon(rel(a, t)) - shift)

    step = step or max(min(_period(graph, a), _period(graph, b)) / 40.0, 0.25)
    return _crossings(f, t0, t1, step)


def add_alignment_edges(
    graph: Graph4D, a: str, b: str, t0: float, t1: float, *, kind: str = "aligned",
    center: str | None = None, relation: str = "alinhamento", window_days: float = 15.0,
) -> int:
    """Uma aresta a-b por alinhamento, vigente por `window_days` de cada lado."""
    n = 0
    for t in find_alignments(graph, a, b, t0, t1, kind=kind, center=center):
        n += graph.add_edge(a, b, t=max(0.0, t - window_days), t_end=t + window_days,
                            relation=relation)
    return n


@dataclass(frozen=True)
class Eclipse:
    t: float
    kind: str  # "solar" ou "lunar"
    latitude_deg: float  # latitude eclíptica da Lua na sizígia


def find_eclipses(
    graph: Graph4D, earth: str, moon: str, sun: str, t0: float, t1: float, *,
    solar_limit_deg: float = 1.5, lunar_limit_deg: float = 1.0,
) -> list[Eclipse]:
    """Sizígias com a Lua perto da eclíptica (candidatos a eclipse)."""

    def geo(nid, t):
        return graph.position_at(nid, t) - graph.position_at(earth, t)

    def elongation(t, shift):
        return _wrap(_lon(geo(moon, t)) - _lon(geo(sun, t)) - shift)

    found: list[Eclipse] = []
    for kind, shift, limit in (("solar", 0.0, solar_limit_deg), ("lunar", np.pi, lunar_limit_deg)):
        for t in _crossings(lambda s, sh=shift: elongation(s, sh), t0, t1, 0.5):
            m = geo(moon, t)
            lat = float(np.degrees(np.arcsin(m[2] / np.linalg.norm(m))))
            if abs(lat) < limit:
                found.append(Eclipse(float(t), kind, lat))
    return sorted(found, key=lambda e: e.t)


def add_eclipse_edges(
    graph: Graph4D, earth: str, moon: str, sun: str, t0: float, t1: float, *,
    window_days: float = 0.25, **limits,
) -> int:
    """Aresta Lua-Sol por eclipse solar e Lua-Terra por eclipse lunar (candidatos)."""
    n = 0
    for e in find_eclipses(graph, earth, moon, sun, t0, t1, **limits):
        other = sun if e.kind == "solar" else earth
        n += graph.add_edge(moon, other, t=max(0.0, e.t - window_days), t_end=e.t + window_days,
                            relation=f"eclipse {e.kind} (candidato)")
    return n


@dataclass(frozen=True)
class Resonance:
    a: str  # órbita mais curta
    b: str  # órbita mais longa
    p: int  # P_b : P_a = p : q
    q: int
    deviation: float  # desvio relativo da razão exata


def find_resonances(
    graph: Graph4D, *, tol: float = 0.009, max_int: int = 4,
    kinds: tuple[str, ...] = ("planeta", "planeta-anão", "lua"),
) -> list[Resonance]:
    """Pares de corpos de mesmo pai com razão de períodos próxima de p:q.

    São quase-ressonâncias de movimento médio: o ângulo de ressonância não é
    verificado. Com tol pequeno, p e q pequenos, a chance de coincidência é
    baixa, mas não nula.
    """
    groups: dict[str, list[str]] = {}
    for n in graph.nodes.values():
        has_period = n.body and getattr(n.body.orbit, "period_days", None)
        if has_period and n.body.parent and n.body.kind in kinds:
            groups.setdefault(n.body.parent, []).append(n.id)
    found: list[Resonance] = []
    for members in groups.values():
        for x, y in combinations(members, 2):
            a, b = sorted((x, y), key=lambda i: _period(graph, i))
            ratio = _period(graph, b) / _period(graph, a)
            best = min(
                (Fraction(p, q) for q in range(1, max_int) for p in range(q + 1, max_int + 1)
                 if np.gcd(p, q) == 1),
                key=lambda fr: abs(ratio - float(fr)) / float(fr),
            )
            dev = abs(ratio - float(best)) / float(best)
            if dev <= tol:
                found.append(Resonance(a, b, best.numerator, best.denominator, dev))
    return found


def add_resonance_edges(graph: Graph4D, **kwargs) -> int:
    n = 0
    for r in find_resonances(graph, **kwargs):
        n += graph.add_edge(r.a, r.b, t=0.0, relation=f"ressonância {r.p}:{r.q}")
    return n
