"""Órbitas: posição de um corpo em função do tempo, relativa ao corpo pai.

Convenções
  tempo      dias desde J2000 (2000-01-01 12:00 TT); escalar ou array
  posição    UA, eixos da eclíptica J2000, relativa ao pai (Sol, planeta...)

Duas implementações:
  KeplerOrbit  elipse kepleriana com taxas seculares lineares (dois corpos)
  TableOrbit   tabela amostrada, p.ex. de uma efeméride externa ou de uma
               integração numérica, com interpolação de Hermite
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

AU_KM = 149_597_870.7
J2000_JD = 2_451_545.0
DAYS_PER_CENTURY = 36_525.0
GAUSS_K = 0.01720209895  # constante gravitacional de Gauss, UA^1.5 / dia (Sol)
OBLIQUITY_J2000 = np.radians(23.4392911)
_RAD = np.pi / 180.0


class Orbit(Protocol):
    def position(self, t) -> np.ndarray: ...

    @property
    def period_days(self) -> float | None: ...


def solve_kepler(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    """Resolve E - e sen E = M por Newton, vetorizado. M em radianos."""
    m = np.mod(np.asarray(mean_anomaly, dtype=float) + np.pi, 2 * np.pi) - np.pi
    ecc = np.copysign(np.pi, m) if e > 0.8 else m + e * np.sin(m)
    ecc = np.broadcast_to(ecc, m.shape).astype(float).copy()
    for _ in range(50):
        step = (ecc - e * np.sin(ecc) - m) / (1.0 - e * np.cos(ecc))
        ecc = ecc - step
        if np.max(np.abs(step)) < 1e-13:
            break
    return ecc


def pole_frame(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Matriz 3x3 (colunas = eixos) do referencial equatorial de um corpo.

    z = polo do corpo; x = nó ascendente do equador dele sobre a eclíptica;
    y completa a tríade. Tudo expresso em coordenadas da eclíptica J2000.
    """
    ra, dec = ra_deg * _RAD, dec_deg * _RAD
    p_eq = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    c, s = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    z = np.array([p_eq[0], p_eq[1] * c + p_eq[2] * s, -p_eq[1] * s + p_eq[2] * c])
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


@dataclass(eq=False)
class KeplerOrbit:
    """Elipse kepleriana. Elementos em J2000 e taxas por século juliano.

    a (UA), e, i, L (longitude média), varpi (longitude do periastro) e node
    (longitude do nó ascendente), os três últimos em graus. `rates` traz a
    variação por século de (a, e, i, L, varpi, node). `frame` leva do
    referencial dos elementos para a eclíptica (identidade por padrão).
    """

    a: float
    e: float
    i: float
    L: float
    varpi: float
    node: float
    rates: tuple[float, float, float, float, float, float] = (0.0,) * 6
    frame: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_period(
        cls, a_au, e, i, period_days, *, node=0.0, argp=0.0, mean_anomaly=0.0,
        node_rate=0.0, varpi_rate=0.0, frame=None,
    ) -> KeplerOrbit:
        """Órbita a partir do período e de (Omega, omega, M0) em t = 0.

        `node_rate` e `varpi_rate` são taxas em graus por século; o movimento
        médio sai de `period_days`, que é o período sideral.
        """
        varpi = node + argp
        n = 360.0 / period_days * DAYS_PER_CENTURY
        return cls(a_au, e, i, varpi + mean_anomaly, varpi, node,
                   (0.0, 0.0, 0.0, n, varpi_rate, node_rate), frame)

    @classmethod
    def from_perihelion(cls, a_au, e, i, *, node, argp, t_peri_days, frame=None) -> KeplerOrbit:
        """Órbita heliocêntrica a partir da data de passagem pelo periélio."""
        n_day = GAUSS_K / a_au**1.5  # rad/dia
        m0 = -np.degrees(n_day) * t_peri_days
        return cls(a_au, e, i, node + argp + m0, node + argp, node,
                   (0.0, 0.0, 0.0, np.degrees(n_day) * DAYS_PER_CENTURY, 0.0, 0.0), frame)

    @property
    def period_days(self) -> float:
        return 360.0 / self.rates[3] * DAYS_PER_CENTURY

    def position(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        cy = t / DAYS_PER_CENTURY
        da, de, di, dl, dw, dn = self.rates
        a, e = self.a + da * cy, self.e + de * cy
        inc = (self.i + di * cy) * _RAD
        varpi = (self.varpi + dw * cy) * _RAD
        node = (self.node + dn * cy) * _RAD
        mean = (self.L + dl * cy) * _RAD - varpi
        ecc = solve_kepler(mean, float(np.mean(e)) if np.ndim(e) else float(e))
        x_o = a * (np.cos(ecc) - e)
        y_o = a * np.sqrt(1.0 - e**2) * np.sin(ecc)
        w = varpi - node
        cw, sw, cn, sn = np.cos(w), np.sin(w), np.cos(node), np.sin(node)
        ci, si = np.cos(inc), np.sin(inc)
        pos = np.stack([
            (cw * cn - sw * sn * ci) * x_o + (-sw * cn - cw * sn * ci) * y_o,
            (cw * sn + sw * cn * ci) * x_o + (-sw * sn + cw * cn * ci) * y_o,
            (sw * si) * x_o + (cw * si) * y_o,
        ], axis=-1)
        return pos if self.frame is None else pos @ self.frame.T


class TableOrbit:
    """Órbita tabelada (t em dias, posição em UA relativa ao pai).

    Com velocidades (UA/dia) a interpolação é cúbica de Hermite; sem elas,
    linear, e a tabela precisa ser densa. Fora do intervalo, vale o extremo.
    """

    def __init__(self, t, pos, vel=None, period_days: float | None = None):
        self.t = np.asarray(t, dtype=float)
        self.pos = np.asarray(pos, dtype=float)
        self.vel = None if vel is None else np.asarray(vel, dtype=float)
        if self.t.ndim != 1 or len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError("t deve ser 1D, estritamente crescente, com ao menos 2 pontos")
        if self.pos.shape != (len(self.t), 3):
            raise ValueError("pos deve ter formato (N, 3)")
        self._period = period_days

    @property
    def period_days(self) -> float | None:
        return self._period

    def position(self, t) -> np.ndarray:
        tq = np.clip(np.asarray(t, dtype=float), self.t[0], self.t[-1])
        idx = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, len(self.t) - 2)
        h = (self.t[idx + 1] - self.t[idx])[..., None]
        s = ((tq - self.t[idx])[..., None]) / h
        p0, p1 = self.pos[idx], self.pos[idx + 1]
        if self.vel is None:
            return p0 + s * (p1 - p0)
        v0, v1 = self.vel[idx], self.vel[idx + 1]
        s2, s3 = s * s, s**3
        return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * v0
                + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * v1)

    @classmethod
    def from_csv(cls, path, *, time="jd", unit="au", period_days=None) -> TableOrbit:
        """Lê um CSV com cabeçalho `t,x,y,z` (ou `jd,x,y,z` com time="jd").

        Posições relativas ao pai, em eixos da eclíptica J2000. unit: "au" ou
        "km". time: "jd" (dia juliano) ou "days" (dias desde J2000).
        """
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        key = "jd" if time == "jd" else "t"
        t = np.array([float(r[key]) for r in rows])
        if time == "jd":
            t = t - J2000_JD
        pos = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        if unit == "km":
            pos = pos / AU_KM
        elif unit != "au":
            raise ValueError("unit deve ser 'au' ou 'km'")
        return cls(t, pos, period_days=period_days)
