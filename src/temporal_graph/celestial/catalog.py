"""Catálogo do sistema solar e construção do grafo.

Origem e confiança dos dados (detalhes em docs/celestial.md):

  Sol, planetas      elementos médios e taxas seculares do JPL (aproximação para
                     1800-2050); validados contra a efeméride do astropy.
  Lua                elementos médios de Meeus, com regressão dos nós e precessão
                     do apogeu; erro de até ~4 graus em longitude.
  outras luas        semieixo, excentricidade, inclinação e período reais;
                     orientação e fase ILUSTRATIVAS (real_phase = False).
  planetas-anões,    forma da órbita real; fase ILUSTRATIVA, exceto os cometas
  asteroides         (data do periélio) e Plutão (tabela do JPL).

Não é um catálogo completo: são os corpos principais (cerca de 50). Para usar
efemérides de verdade, carregue um CSV com `TableOrbit.from_csv` e troque o
`orbit` do corpo.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from ..model import Graph4D
from .events import add_alignment_edges, add_eclipse_edges, add_resonance_edges
from .orbits import AU_KM, J2000_JD, KeplerOrbit, pole_frame

EPOCH = datetime(2000, 1, 1, 12)
T_MAX_DAYS = 36_525.0  # 2000 a 2100
LIGHT_SPEED_AU_DAY = 299_792.458 * 86_400.0 / AU_KM

KIND_COLORS = {
    "estrela": "#FFD166",
    "planeta": "#4C9BE8",
    "planeta-anão": "#C77DFF",
    "lua": "#B8C0CC",
    "asteroide": "#C08552",
    "cometa": "#7AE7C7",
    "transnetuniano": "#F28482",
}


def days_since_j2000(year: int, month: int, day: float) -> float:
    whole = int(day)
    d = date(year, month, whole)
    return (d - EPOCH.date()).days - 0.5 + (day - whole)


# nome: (elementos J2000 [a, e, i, L, varpi, node], taxas por século, raio km, GM km3/s2, texto)
_PLANETS = {
    "Mercúrio": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                 (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081),
                 2439.7, 22031.86, "Mercúrio, planeta rochoso mais próximo do Sol, sem luas."),
    "Vênus": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418),
              6051.8, 324858.59, "Vênus, planeta rochoso com atmosfera densa, sem luas."),
    "Terra": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0),
              6371.0, 398600.44, "Terra, planeta rochoso com água líquida e uma lua."),
    "Marte": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
              (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343),
              3389.5, 42828.37, "Marte, planeta rochoso com duas luas pequenas."),
    "Júpiter": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106),
                69911.0, 126686534.0, "Júpiter, o maior planeta, gigante gasoso com muitas luas."),
    "Saturno": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
                (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794),
                58232.0, 37931187.0, "Saturno, gigante gasoso com anéis e muitas luas."),
    "Urano": ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
              (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589),
              25362.0, 5793939.0, "Urano, gigante de gelo com o eixo inclinado de lado."),
    "Netuno": ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
               (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664),
               24622.0, 6836529.0, "Netuno, gigante de gelo, o planeta mais distante do Sol."),
    "Plutão": ((39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684),
               (-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482),
               1188.3, 869.6, "Plutão, planeta-anão do cinturão de Kuiper com cinco luas."),
}

# nome: (a UA, e, i graus, Omega, omega, raio km, tipo, texto). Fase ilustrativa.
_SMALL_BODIES = {
    "Ceres": (2.7692, 0.0760, 10.594, 80.306, 73.598, 469.7, "planeta-anão",
              "Ceres, planeta-anão no cinturão de asteroides."),
    "Haumea": (43.12, 0.195, 28.2, 122.2, 239.0, 780.0, "planeta-anão",
               "Haumea, planeta-anão alongado do cinturão de Kuiper."),
    "Makemake": (45.5, 0.161, 29.0, 79.3, 297.0, 715.0, "planeta-anão",
                 "Makemake, planeta-anão do cinturão de Kuiper."),
    "Éris": (67.7, 0.436, 44.0, 35.9, 151.4, 1163.0, "planeta-anão",
             "Éris, planeta-anão do disco disperso, mais massivo que Plutão."),
    "Vesta": (2.3615, 0.0887, 7.14, 103.8, 150.7, 262.7, "asteroide",
              "Vesta, asteroide do cinturão principal, segundo mais massivo."),
    "Palas": (2.7722, 0.2299, 34.93, 172.9, 310.4, 256.0, "asteroide",
              "Palas, grande asteroide do cinturão principal, de órbita inclinada."),
    "Hígia": (3.1415, 0.1146, 3.83, 283.2, 312.3, 217.0, "asteroide",
              "Hígia, asteroide do cinturão principal."),
    "Eros": (1.4579, 0.2229, 10.83, 304.3, 178.9, 8.4, "asteroide",
             "Eros, asteroide próximo da Terra."),
    "Apófis": (0.9224, 0.1914, 3.34, 204.0, 126.7, 0.17, "asteroide",
               "Apófis, asteroide próximo da Terra."),
    "Sedna": (506.0, 0.8496, 11.93, 144.5, 311.5, 500.0, "transnetuniano",
              "Sedna, objeto transnetuniano de órbita muito alongada."),
}

# nome: (a UA, e, i, Omega, omega, (ano, mês, dia do periélio), raio km, texto). Fase real.
_COMETS = {
    "Halley": (17.834, 0.96714, 162.26, 58.42, 111.33, (1986, 2, 9.46), 5.5,
               "Halley, cometa periódico de período próximo de 76 anos."),
    "Encke": (2.2150, 0.8483, 11.78, 334.57, 186.54, (2023, 10, 22.7), 2.4,
              "Encke, cometa de período curto, cerca de 3,3 anos."),
    "Hale-Bopp": (186.0, 0.99506, 89.43, 282.47, 130.59, (1997, 4, 1.1), 30.0,
                  "Hale-Bopp, cometa de período muito longo, visível em 1997."),
}

# pai: (RA, Dec do polo em graus)
_POLES = {
    "Marte": (317.681, 52.886), "Júpiter": (268.057, 64.495), "Saturno": (40.589, 83.537),
    "Urano": (257.311, -15.175), "Netuno": (299.36, 43.46), "Plutão": (132.993, -6.163),
}

# (pai, nome, a km, e, i graus ao equador do pai, período dias, raio km, GM km3/s2, texto)
_MOONS = [
    ("Marte", "Fobos", 9376, 0.0151, 1.08, 0.31891, 11.3, 0, "Fobos, satélite natural de Marte, a lua mais interna."),
    ("Marte", "Deimos", 23463, 0.0005, 1.79, 1.26244, 6.2, 0, "Deimos, satélite natural de Marte, pequeno e distante."),
    ("Júpiter", "Amalteia", 181366, 0.0032, 0.374, 0.49818, 83.5, 0, "Amalteia, satélite natural de Júpiter, pequeno e interno."),
    ("Júpiter", "Io", 421800, 0.0041, 0.036, 1.769138, 1821.6, 5959.9, "Io, satélite natural de Júpiter, o corpo mais vulcânico."),
    ("Júpiter", "Europa", 671100, 0.0094, 0.466, 3.551181, 1560.8, 3202.7, "Europa, satélite natural de Júpiter, com oceano sob o gelo."),
    ("Júpiter", "Ganimedes", 1070400, 0.0013, 0.177, 7.154553, 2634.1, 9887.8, "Ganimedes, satélite natural de Júpiter, a maior lua do sistema."),
    ("Júpiter", "Calisto", 1882700, 0.0074, 0.192, 16.689018, 2410.3, 7179.3, "Calisto, satélite natural de Júpiter, muito craterizado."),
    ("Júpiter", "Himalia", 11461000, 0.1513, 27.5, 250.56, 69.8, 0, "Himalia, satélite natural irregular de Júpiter."),
    ("Saturno", "Mimas", 185520, 0.0196, 1.574, 0.942422, 198.2, 0, "Mimas, satélite natural de Saturno, com uma cratera gigante."),
    ("Saturno", "Encélado", 238020, 0.0047, 0.009, 1.370218, 252.1, 0, "Encélado, satélite natural de Saturno, com gêiseres de gelo."),
    ("Saturno", "Tétis", 294660, 0.0001, 1.091, 1.887802, 531.1, 0, "Tétis, satélite natural de Saturno, de gelo."),
    ("Saturno", "Dione", 377400, 0.0022, 0.028, 2.736915, 561.4, 0, "Dione, satélite natural de Saturno, de gelo."),
    ("Saturno", "Reia", 527040, 0.0013, 0.331, 4.5175, 763.8, 0, "Reia, satélite natural de Saturno, de gelo."),
    ("Saturno", "Titã", 1221870, 0.0288, 0.348, 15.945421, 2574.7, 8978.1, "Titã, satélite natural de Saturno, com atmosfera densa."),
    ("Saturno", "Hipérion", 1481010, 0.1230, 0.43, 21.276609, 135.0, 0, "Hipérion, satélite natural irregular de Saturno."),
    ("Saturno", "Jápeto", 3560820, 0.0276, 7.57, 79.3215, 734.5, 0, "Jápeto, satélite natural de Saturno, com dois tons de cor."),
    ("Saturno", "Febe", 12952000, 0.1562, 173.0, 550.31, 106.5, 0, "Febe, satélite natural de Saturno, de órbita retrógrada."),
    ("Urano", "Miranda", 129390, 0.0013, 4.338, 1.413479, 235.8, 0, "Miranda, satélite natural de Urano, de relevo acidentado."),
    ("Urano", "Ariel", 191020, 0.0012, 0.041, 2.520379, 578.9, 0, "Ariel, satélite natural de Urano."),
    ("Urano", "Umbriel", 266300, 0.0039, 0.128, 4.144176, 584.7, 0, "Umbriel, satélite natural de Urano, escuro."),
    ("Urano", "Titânia", 435910, 0.0011, 0.079, 8.705872, 788.4, 0, "Titânia, satélite natural de Urano, a maior lua dele."),
    ("Urano", "Oberon", 583520, 0.0014, 0.068, 13.463239, 761.4, 0, "Oberon, satélite natural de Urano."),
    ("Netuno", "Proteu", 117647, 0.0005, 0.075, 1.122315, 210.0, 0, "Proteu, satélite natural de Netuno, irregular."),
    ("Netuno", "Tritão", 354759, 0.000016, 156.865, 5.876854, 1353.4, 1427.6, "Tritão, satélite natural de Netuno, de órbita retrógrada."),
    ("Netuno", "Nereida", 5513818, 0.7507, 7.09, 360.1362, 170.0, 0, "Nereida, satélite natural de Netuno, de órbita muito excêntrica."),
    ("Plutão", "Caronte", 19591, 0.0002, 0.0, 6.38723, 606.0, 105.9, "Caronte, satélite natural de Plutão, quase do tamanho dele."),
    ("Plutão", "Nix", 48694, 0.0020, 0.0, 24.8562, 25.0, 0, "Nix, satélite natural pequeno de Plutão."),
    ("Plutão", "Hidra", 64738, 0.0059, 0.0, 38.2018, 30.0, 0, "Hidra, satélite natural pequeno de Plutão."),
]

# Lua: elementos médios de Meeus (referência: eclíptica), taxas por século
_MOON_L0, _MOON_LRATE = 218.3164477, 481267.88123421
_MOON_M0, _MOON_MRATE = 134.9633964, 477198.8675055
GOLDEN = 137.50776405


def _phase(k: int) -> tuple[float, float, float]:
    """(nó, argumento do periastro, anomalia média) ilustrativos e reprodutíveis."""
    return (k * GOLDEN) % 360.0, (k * 222.49223595) % 360.0, (k * 97.3) % 360.0


def _kepler_helio(a, e, i, node, argp, k: int) -> KeplerOrbit:
    period = 365.25636 * a**1.5
    _, _, m0 = _phase(k)
    return KeplerOrbit.from_period(a, e, i, period, node=node, argp=argp, mean_anomaly=m0)


def moon_orbit() -> KeplerOrbit:
    return KeplerOrbit(
        384_399 / AU_KM, 0.0549, 5.145, _MOON_L0, _MOON_L0 - _MOON_M0, 125.0445479,
        rates=(0, 0, 0, _MOON_LRATE, _MOON_LRATE - _MOON_MRATE, -1934.1362891),
    )


def generate_solar_system(*, dynamics: str = "kepler", events: bool = True) -> Graph4D:
    """Sistema solar como um grafo com um nó por corpo.

    dynamics: "kepler" (analítico) ou "nbody" (Sol e planetas integrados
    numericamente a partir das condições iniciais keplerianas; ver nbody.py).
    events: se verdadeiro, acrescenta arestas recorrentes (oposições,
    conjunções, eclipses candidatos) e de ressonância.
    """
    if dynamics not in ("kepler", "nbody"):
        raise ValueError("dynamics deve ser 'kepler' ou 'nbody'")
    g = Graph4D(
        t_max=T_MAX_DAYS, epoch=EPOCH, light_speed=LIGHT_SPEED_AU_DAY,
        t_default=days_since_j2000(2026, 10, 19), t_step=1.0,
    )
    g.add_body("Sol", kind="estrela", radius_km=695_700.0, gm_km3_s2=132_712_440_018.0,
               color=KIND_COLORS["estrela"],
               text="Sol, estrela no centro do sistema solar.")

    planet_orbits = {name: KeplerOrbit(*el[:3], el[3], el[4], el[5], rates=rt)
                     for name, (el, rt, *_rest) in _PLANETS.items()}
    if dynamics == "nbody":
        from .nbody import nbody_orbits  # noqa: PLC0415 - integração só quando pedida

        planet_orbits = nbody_orbits(planet_orbits, T_MAX_DAYS)
    for name, (_el, _rt, radius, gm, text) in _PLANETS.items():
        kind = "planeta-anão" if name == "Plutão" else "planeta"
        g.add_body(name, parent="Sol", orbit=planet_orbits[name], kind=kind, radius_km=radius,
                   gm_km3_s2=gm, color=KIND_COLORS[kind], text=text)

    k = 0
    for name, (a, e, i, node, argp, radius, kind, text) in _SMALL_BODIES.items():
        k += 1
        g.add_body(name, parent="Sol", orbit=_kepler_helio(a, e, i, node, argp, k), kind=kind,
                   radius_km=radius, color=KIND_COLORS[kind], text=text, real_phase=False)
    for name, (a, e, i, node, argp, (y, m, d), radius, text) in _COMETS.items():
        orbit = KeplerOrbit.from_perihelion(a, e, i, node=node, argp=argp,
                                            t_peri_days=days_since_j2000(y, m, d))
        g.add_body(name, parent="Sol", orbit=orbit, kind="cometa", radius_km=radius,
                   color=KIND_COLORS["cometa"], text=text)

    g.add_body("Lua", parent="Terra", orbit=moon_orbit(), kind="lua", radius_km=1737.4,
               gm_km3_s2=4902.80, color=KIND_COLORS["lua"],
               text="Lua, satélite natural da Terra.")
    for parent, name, a_km, e, i, period, radius, gm, text in _MOONS:
        k += 1
        node, argp, m0 = _phase(k)
        orbit = KeplerOrbit.from_period(
            a_km / AU_KM, e, i, period, node=node, argp=argp, mean_anomaly=m0,
            frame=pole_frame(*_POLES[parent]),
        )
        g.add_body(name, parent=parent, orbit=orbit, kind="lua", radius_km=radius, gm_km3_s2=gm,
                   color=KIND_COLORS["lua"], text=text, real_phase=False)

    for node in list(g.nodes.values()):
        if node.body and node.body.parent:
            g.add_edge(node.body.parent, node.id, t=0.0, relation="orbita")
    if events:
        add_resonance_edges(g)
        add_alignment_edges(g, "Terra", "Marte", 0.0, T_MAX_DAYS, relation="oposição")
        add_alignment_edges(g, "Júpiter", "Saturno", 0.0, T_MAX_DAYS, center="Terra",
                            relation="conjunção (vista da Terra)")
        add_eclipse_edges(g, "Terra", "Lua", "Sol", 0.0, T_MAX_DAYS)
    return g


def describe_body(graph: Graph4D, nid: str, t: float, observer: str | None = None) -> str:
    """Resumo físico em t: posição, distância ao pai, período e atraso da luz."""
    node = graph.nodes[nid]
    body = node.body
    if body is None:
        return ""
    pos = graph.position_at(nid, t)
    parts = [f"posição ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) UA"]
    if body.parent is not None:
        d = float(np.linalg.norm(graph.local_position_at(nid, t)))
        parts.append(f"a {d:.4g} UA de {body.parent}" if d >= 0.01 else
                     f"a {d * AU_KM:,.0f} km de {body.parent}".replace(",", "."))
    period = getattr(body.orbit, "period_days", None)
    if period:
        parts.append(f"período {period / 365.25636:.3g} anos" if period > 800
                     else f"período {period:.4g} dias")
    if observer and observer != nid and observer in graph.nodes:
        dist = float(np.linalg.norm(pos - graph.position_at(observer, t)))
        minutes = dist / graph.light_speed * 24 * 60
        parts.append(f"a luz leva {minutes:.4g} min até {observer}")
    if not body.real_phase:
        parts.append("fase da órbita ilustrativa")
    return "; ".join(parts)


__all__ = [
    "EPOCH", "J2000_JD", "KIND_COLORS", "LIGHT_SPEED_AU_DAY", "T_MAX_DAYS",
    "days_since_j2000", "describe_body", "generate_solar_system", "moon_orbit",
]
