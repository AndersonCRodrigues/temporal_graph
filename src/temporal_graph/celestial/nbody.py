"""Integração numérica N-corpos (Sol e planetas) por leapfrog.

Integra a atração gravitacional mútua com um integrador simplético de 4ª
ordem (Yoshida) e passo fixo, em um referencial baricêntrico. Devolve órbitas
tabeladas heliocêntricas, com velocidades, que o grafo usa no lugar das elipses.

Estados iniciais: os vetores de J2000 em `J2000_STATE` (gerados com a
efeméride embutida do astropy) ou, na falta deles, os das órbitas keplerianas.
Elementos MÉDIOS dão uma condição inicial pior do que um estado real: o
movimento médio osculante difere do médio e a fase deriva com o tempo.

Limites: passo fixo (dt = 1 dia serve a planetas, não a cometas de periélio
próximo), sem relatividade geral, sem as luas (cada planeta leva a massa do seu
sistema) e sem corpos menores. Não substitui uma efeméride do JPL.
"""

from __future__ import annotations

import numpy as np

from .orbits import AU_KM, KeplerOrbit, TableOrbit

SECONDS_PER_DAY = 86_400.0
GM_SUN = 132_712_440_018.0  # km^3/s^2
# massas dos sistemas planeta+luas (km^3/s^2)
SYSTEM_GM = {
    "Mercúrio": 22031.86, "Vênus": 324858.59, "Terra": 403503.24, "Marte": 42828.37,
    "Júpiter": 126712767.9, "Saturno": 37940584.8, "Urano": 5794556.4,
    "Netuno": 6836527.1, "Plutão": 975.5,
}
# Estado heliocêntrico em J2000 (UA e UA/dia), eclíptica J2000; Terra = baricentro
# Terra-Lua. Posições da efeméride embutida do astropy; velocidades por diferença
# finita de 4ª ordem (h = 0,5 d) dessas posições, porque as velocidades devolvidas
# pelo astropy para os planetas externos divergem até 0,3% da evolução das posições.
J2000_STATE = {
    "Mercúrio": (-0.130091772797, -0.447286717761, -0.024597982446,
                 0.0213663472833, -0.00644800820567, -0.00248786007668),
    "Vênus": (-0.718301785174, -0.0326561905344, 0.0410153174315,
              0.000798187816252, -0.0202949938582, -0.000323429926206),
    "Terra": (-0.177160633351, 0.967213978946, 1.87567517074e-10,
              -0.0172031446845, -0.00316404590387, -5.91136591529e-09),
    "Marte": (1.39070519983, -0.0133738169973, -0.0344617453076,
              0.000671348702153, 0.0151870856958, 0.000301662273468),
    "Júpiter": (4.0015600833, 2.93811129883, -0.10166254386,
                -0.00456635674814, 0.0064428918998, 7.55358543175e-05),
    "Saturno": (6.40460226671, 6.57042038016, -0.369610483189,
                -0.00429543972114, 0.00389075768601, 0.0001031311636),
    "Urano": (14.4320596937, -13.7351145926, -0.238305361691,
              0.0026772156946, 0.00267156528977, -2.47791433837e-05),
    "Netuno": (16.8120250656, -24.9916782969, 0.127205368467,
               0.00257988303201, 0.00177671673237, -9.60309399067e-05),
}
_YOSHIDA_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_W0 = -(2.0 ** (1.0 / 3.0)) * _YOSHIDA_W1
_CACHE: dict[tuple, dict[str, TableOrbit]] = {}


def to_au3_day2(gm_km3_s2: float) -> float:
    return gm_km3_s2 * SECONDS_PER_DAY**2 / AU_KM**3


def accelerations(gm: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """a_i = soma_j GM_j (r_j - r_i) / |r_j - r_i|^3."""
    d = pos[None, :, :] - pos[:, None, :]
    r3 = np.linalg.norm(d, axis=-1) ** 3
    np.fill_diagonal(r3, np.inf)
    return np.einsum("ij,ijk->ik", gm[None, :] / r3, d)


def leapfrog(gm, pos, vel, steps: int, dt: float):
    """Kick-drift-kick. Devolve (pos, vel) em cada passo, formato (steps+1, N, 3)."""
    p, v = pos.copy(), vel.copy()
    out_p, out_v = np.empty((steps + 1, *p.shape)), np.empty((steps + 1, *p.shape))
    out_p[0], out_v[0] = p, v
    acc = accelerations(gm, p)
    for k in range(1, steps + 1):
        v += 0.5 * dt * acc
        p += dt * v
        acc = accelerations(gm, p)
        v += 0.5 * dt * acc
        out_p[k], out_v[k] = p, v
    return out_p, out_v


def yoshida4(gm, pos, vel, steps: int, dt: float):
    """Integrador simplético de 4ª ordem (três kick-drift-kick por passo)."""
    p, v = pos.copy(), vel.copy()
    out_p, out_v = np.empty((steps + 1, *p.shape)), np.empty((steps + 1, *p.shape))
    out_p[0], out_v[0] = p, v
    acc = accelerations(gm, p)
    sub = (_YOSHIDA_W1 * dt, _YOSHIDA_W0 * dt, _YOSHIDA_W1 * dt)
    for k in range(1, steps + 1):
        for h in sub:
            v += 0.5 * h * acc
            p += h * v
            acc = accelerations(gm, p)
            v += 0.5 * h * acc
        out_p[k], out_v[k] = p, v
    return out_p, out_v


def total_energy(gm, pos, vel) -> float:
    """Energia por unidade de G (massas = GM). Útil para checar a integração."""
    kinetic = 0.5 * np.sum(gm * np.sum(vel**2, axis=-1))
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    iu = np.triu_indices(len(gm), 1)
    return float(kinetic - np.sum(gm[iu[0]] * gm[iu[1]] / d[iu]))


def nbody_orbits(
    kepler: dict[str, KeplerOrbit], t_end: float, dt: float = 1.0,
    initial_state: dict[str, tuple] | None = J2000_STATE,
):
    """Integra de t = 0 a t_end e devolve {nome: TableOrbit} heliocêntrico.

    `initial_state` dá (x, y, z, vx, vy, vz) heliocêntricos por corpo; quem não
    estiver nele (Plutão) parte da órbita kepleriana.
    """
    state_key = tuple(sorted(initial_state)) if initial_state else ()
    key = (round(t_end, 6), dt, state_key,
           tuple((n, o.a, o.e, o.i, o.L, o.varpi, o.node, o.rates) for n, o in kepler.items()))
    if key in _CACHE:
        return _CACHE[key]
    names = list(kepler)
    gm = np.array([to_au3_day2(GM_SUN)] + [to_au3_day2(SYSTEM_GM[n]) for n in names])
    h = 0.01
    given = initial_state or {}
    x = np.array([given[n][:3] if n in given else kepler[n].position(0.0) for n in names])
    v = np.array([
        given[n][3:] if n in given
        else (kepler[n].position(h) - kepler[n].position(-h)) / (2 * h)
        for n in names
    ])
    # referencial do centro de massa: o Sol está em -sum(m x)/M
    sun_x = -np.sum(gm[1:, None] * x, axis=0) / gm.sum()
    sun_v = -np.sum(gm[1:, None] * v, axis=0) / gm.sum()
    pos0 = np.vstack([sun_x, x + sun_x])
    vel0 = np.vstack([sun_v, v + sun_v])
    steps = int(np.ceil(t_end / dt))
    pos, vel = yoshida4(gm, pos0, vel0, steps, dt)
    times = np.arange(steps + 1) * dt
    result = {
        n: TableOrbit(times, pos[:, i + 1] - pos[:, 0], vel[:, i + 1] - vel[:, 0],
                      period_days=kepler[n].period_days)
        for i, n in enumerate(names)
    }
    _CACHE[key] = result
    return result
