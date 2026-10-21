"""Relatividade restrita: boosts de Lorentz, intervalo de Minkowski e tempo-luz.

Eventos são (x, y, z, w) com w = c*t, ou seja, na mesma unidade das distâncias,
e métrica (+, +, +, -). Para um grafo com `light_speed` diferente de 1 (por
exemplo UA/dia no sistema solar), use `to_ct` e `from_ct` para converter.

Um boost com velocidade beta = v/c ao longo de um eixo leva um evento do
referencial A para o referencial B que se move com v em relação a A:

    x' = gamma (x - beta w)        w' = gamma (w - beta x)
"""

from __future__ import annotations

import numpy as np

from .model import Graph4D

AXES = {"x": 0, "y": 1, "z": 2}
LIGHT_TOLERANCE = 1e-9


def lorentz_boost_matrix(axis: str, beta: float) -> np.ndarray:
    """Matriz 4x4 de um boost ao longo de `axis` ("x", "y" ou "z")."""
    if axis not in AXES:
        raise ValueError(f"Eixo desconhecido: {axis!r}")
    if not -1.0 < beta < 1.0:
        raise ValueError("beta deve estar em (-1, 1)")
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    m = np.eye(4)
    i = AXES[axis]
    m[i, i] = m[3, 3] = gamma
    m[i, 3] = m[3, i] = -gamma * beta
    return m


def minkowski_interval(a, b) -> float:
    """Intervalo ao quadrado s2 = |dx|^2 - dw^2. Negativo = tipo-tempo."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(np.dot(d[:3], d[:3]) - d[3] ** 2)


def classify_interval(a, b, tol: float = LIGHT_TOLERANCE) -> str:
    """"tipo-tempo", "tipo-luz" ou "tipo-espaço" para o par de eventos."""
    s2 = minkowski_interval(a, b)
    scale = max(np.dot(np.subtract(b, a)[:3], np.subtract(b, a)[:3]), 1.0)
    if abs(s2) <= tol * scale:
        return "tipo-luz"
    return "tipo-tempo" if s2 < 0 else "tipo-espaço"


def to_ct(graph: Graph4D, t):
    """Instante t -> w = c*t, na unidade de distância do grafo."""
    return np.asarray(t, dtype=float) * graph.light_speed


def light_time(graph: Graph4D, distance) -> np.ndarray:
    """Tempo (unidade de t) que a luz leva para percorrer `distance`."""
    return np.asarray(distance, dtype=float) / graph.light_speed


def retarded_time(graph: Graph4D, nid: str, observer: str, t: float, iterations: int = 8) -> float:
    """Instante em que a luz que chega ao observador em t saiu do corpo `nid`.

    Resolve t_ret = t - |x_nid(t_ret) - x_obs(t)| / c por iteração de ponto fixo.
    """
    obs = graph.position_at(observer, t)
    tau = 0.0
    for _ in range(iterations):
        tau = float(light_time(graph, np.linalg.norm(graph.position_at(nid, t - tau) - obs)))
    return t - tau
