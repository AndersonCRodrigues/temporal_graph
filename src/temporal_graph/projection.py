"""Rotações e projeções R^4 -> R^3.

Convenção: colunas (x, y, z, w), com w = tempo.

As rotações são euclidianas (ortogonais): misturam espaço e tempo como se w
fosse mais uma dimensão espacial. Para a transformação de Lorentz (relatividade
restrita, com w = c*t), use `boosts`: cada (eixo, beta) aplica um boost.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .relativity import lorentz_boost_matrix

DEFAULT_CAMERA_W = 8.0
MIN_DEPTH = 0.15

PLANES = {
    "xy": (0, 1), "xz": (0, 2), "xw": (0, 3),
    "yz": (1, 2), "yw": (1, 3), "zw": (2, 3),
}
DROP_AXIS = {"drop_x": 0, "drop_y": 1, "drop_z": 2, "drop_w": 3}
PROJECTION_MODES = (*DROP_AXIS, "perspective")


def rotation_matrix_4d(plane: str, angle: float) -> np.ndarray:
    """Matriz 4x4 de rotação no plano `plane` (ex.: "xw") por `angle` radianos."""
    if plane not in PLANES:
        raise ValueError(f"Plano desconhecido: {plane!r}")
    i, j = PLANES[plane]
    c, s = np.cos(angle), np.sin(angle)
    rot = np.eye(4)
    rot[i, i], rot[i, j] = c, -s
    rot[j, i], rot[j, j] = s, c
    return rot


def project_drop_axis(points4: np.ndarray, drop: int = 3) -> np.ndarray:
    """Descarta uma coluna. drop=3 mantém só o espaço (x, y, z)."""
    keep = [i for i in range(4) if i != drop]
    return points4[:, keep]


def project_perspective_w(points4: np.ndarray, camera_w: float = DEFAULT_CAMERA_W) -> np.ndarray:
    """Perspectiva com câmera em w = camera_w olhando para w decrescente.

    Pontos com w próximo ou acima de camera_w têm a profundidade limitada a
    MIN_DEPTH, para evitar divisão por zero.
    """
    depth = np.clip(camera_w - points4[:, 3], MIN_DEPTH, None)
    return points4[:, :3] * (camera_w / depth)[:, None]


def project_points(
    points4: np.ndarray,
    mode: str = "drop_w",
    *,
    camera_w: float = DEFAULT_CAMERA_W,
    rotations: Iterable[tuple[str, float]] | None = None,
    boosts: Iterable[tuple[str, float]] | None = None,
) -> np.ndarray:
    """Aplica rotações, depois boosts (na ordem dada), e projeta (N, 4) em (N, 3)."""
    pts = np.asarray(points4, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError("points4 deve ter formato (N, 4)")
    for plane, angle in rotations or ():
        pts = pts @ rotation_matrix_4d(plane, angle).T
    for axis, beta in boosts or ():
        pts = pts @ lorentz_boost_matrix(axis, beta).T
    if mode == "perspective":
        return project_perspective_w(pts, camera_w=camera_w)
    if mode in DROP_AXIS:
        return project_drop_axis(pts, drop=DROP_AXIS[mode])
    raise ValueError(f"Modo de projeção desconhecido: {mode!r}")
