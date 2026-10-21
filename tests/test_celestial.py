"""Testes do catálogo do sistema solar, dos eventos recorrentes e do N-corpos."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from temporal_graph.celestial.catalog import (
    EPOCH,
    T_MAX_DAYS,
    days_since_j2000,
    describe_body,
    generate_solar_system,
)
from temporal_graph.celestial.events import (
    add_alignment_edges,
    find_alignments,
    find_eclipses,
    find_resonances,
)
from temporal_graph.celestial.nbody import (
    J2000_STATE,
    SYSTEM_GM,
    accelerations,
    leapfrog,
    nbody_orbits,
    to_au3_day2,
    total_energy,
    yoshida4,
)
from temporal_graph.celestial.orbits import AU_KM, J2000_JD, KeplerOrbit, TableOrbit
from temporal_graph.model import Graph4D

D = days_since_j2000


@pytest.fixture(scope="module")
def solar():
    return generate_solar_system(events=False)


def date(t):
    return (EPOCH + timedelta(days=float(t))).strftime("%Y-%m-%d")


# -- catálogo ---------------------------------------------------------------


def test_catalog_has_one_node_per_body_with_text_and_valid_parents(solar):
    kinds = {}
    for n in solar.nodes.values():
        kinds[n.body.kind] = kinds.get(n.body.kind, 0) + 1
        assert n.text and (n.body.parent is None or n.body.parent in solar.nodes)
    assert kinds == {"estrela": 1, "planeta": 8, "planeta-anão": 5, "asteroide": 5,
                     "transnetuniano": 1, "cometa": 3, "lua": 29}
    assert solar.epoch == EPOCH and solar.t_max == T_MAX_DAYS
    assert solar.light_speed == pytest.approx(173.1446, rel=1e-6)


def test_kepler_third_law_holds_for_every_orbit(solar):
    gm_system = {n.id: n.body.gm_km3_s2 for n in solar.nodes.values()}
    gm_system["Plutão"] = SYSTEM_GM["Plutão"]  # já inclui Caronte; Nix e Hidra orbitam o baricentro
    for n in solar.nodes.values():
        b = n.body
        if b.parent is None or not getattr(b.orbit, "period_days", None):
            continue
        mu = gm_system[b.parent] + (0.0 if b.parent == "Plutão" else b.gm_km3_s2)
        p_calc = 2 * np.pi * np.sqrt((b.orbit.a * AU_KM) ** 3 / mu)
        assert abs(p_calc / 86400 / b.orbit.period_days - 1) < 0.01, n.id


def test_phase_flags(solar):
    real = {n.id for n in solar.nodes.values() if n.body.real_phase}
    assert {"Sol", "Terra", "Júpiter", "Lua", "Halley", "Encke", "Hale-Bopp", "Plutão"} <= real
    assert not ({"Fobos", "Titã", "Ceres", "Éris", "Sedna", "Eros"} & real)


def test_halley_passes_perihelion_on_the_given_date(solar):
    t = np.arange(D(1985, 12, 1), D(1986, 4, 1), 0.1)
    d = np.linalg.norm(solar.position_at("Halley", t), axis=-1)
    assert date(t[np.argmin(d)]) == "1986-02-09" and d.min() == pytest.approx(0.5859, abs=2e-3)


def test_earth_perihelion_is_in_early_january(solar):
    t = np.arange(D(2026, 1, 1), D(2027, 1, 1), 0.25)
    d = np.linalg.norm(solar.position_at("Terra", t), axis=-1)
    assert abs(t[np.argmin(d)] - D(2026, 1, 3)) < 5 and 0.98 < d.min() < 0.985 and d.max() > 1.015


def test_moon_period_and_distance(solar):
    assert solar.nodes["Lua"].body.orbit.period_days == pytest.approx(27.3216, abs=1e-3)
    d = np.linalg.norm(solar.local_position_at("Lua", np.linspace(0, 400, 800)), axis=-1) * AU_KM
    assert 356_000 < d.min() < 365_000 and 400_000 < d.max() < 408_000


def test_retrograde_moon_goes_the_other_way(solar):
    tritao = solar.local_position_at("Tritão", np.array([0.0, 0.2]))
    io = solar.local_position_at("Io", np.array([0.0, 0.2]))
    # sentido de giro em torno do polo do pai: sinal do produto vetorial com o polo
    from temporal_graph.celestial.catalog import _POLES
    from temporal_graph.celestial.orbits import pole_frame

    def spin(p, parent):
        return np.sign(np.dot(np.cross(p[0], p[1]), pole_frame(*_POLES[parent])[:, 2]))

    assert spin(tritao, "Netuno") == -1 and spin(io, "Júpiter") == 1


def test_describe_body_reports_light_time_and_phase_honesty(solar):
    text = describe_body(solar, "Júpiter", D(2026, 10, 19), "Terra")
    assert "min até Terra" in text and "a 5." in text and "anos" in text
    assert "fase da órbita ilustrativa" in describe_body(solar, "Fobos", 0.0, "Terra")
    assert describe_body(solar, "Sol", 0.0) == "posição (0.000, 0.000, 0.000) UA"


def test_generate_validates_dynamics_and_events_flag():
    with pytest.raises(ValueError):
        generate_solar_system(dynamics="relativistico")
    g = generate_solar_system()
    relations = {e.relation.split(" (")[0] for e in g.edges}
    assert {"orbita", "oposição", "eclipse solar", "eclipse lunar", "ressonância 2:1"} <= relations
    assert sum(e.relation == "orbita" for e in g.edges) == 51


# -- planetas contra uma referência independente ---------------------------------


def _ecliptic(v):
    from temporal_graph.celestial.orbits import OBLIQUITY_J2000

    c, s = np.cos(OBLIQUITY_J2000), np.sin(OBLIQUITY_J2000)
    return np.array([v[0], v[1] * c + v[2] * s, -v[1] * s + v[2] * c])


def _astropy_helio(body, t):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    when = Time(J2000_JD + t, format="jd", scale="tdb")
    return _ecliptic((get_body_barycentric(body, when) - get_body_barycentric("sun", when))
                     .xyz.to("au").value)


PLANETS = {"Mercúrio": "mercury", "Vênus": "venus", "Terra": "earth-moon-barycenter",
           "Marte": "mars", "Júpiter": "jupiter", "Saturno": "saturn", "Urano": "uranus",
           "Netuno": "neptune"}


def test_planets_match_the_astropy_ephemeris(solar):
    pytest.importorskip("astropy")
    for t in (0.0, D(2026, 10, 19), 18262.5, 32872.5):
        for name, ref in PLANETS.items():
            err = np.linalg.norm(solar.position_at(name, t) - _astropy_helio(ref, t))
            assert err < 0.04, (name, t, err)  # pior caso: Saturno, ~0,03 UA a 9,5 UA


def test_moon_matches_the_astropy_ephemeris_within_a_few_degrees(solar):
    pytest.importorskip("astropy")
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time

    worst = 0.0
    for t in np.linspace(0, T_MAX_DAYS, 60):
        when = Time(J2000_JD + t, format="jd", scale="tdb")
        ref = _ecliptic((get_body_barycentric("moon", when) - get_body_barycentric("earth", when))
                        .xyz.to("au").value)
        mine = solar.local_position_at("Lua", t)
        lon = np.degrees(np.arctan2(mine[1], mine[0]) - np.arctan2(ref[1], ref[0]))
        worst = max(worst, abs((lon + 180) % 360 - 180))
    assert worst < 5.0


# -- eventos recorrentes ------------------------------------------------------------


def test_mars_oppositions_match_real_dates(solar):
    found = find_alignments(solar, "Terra", "Marte", D(2020, 1, 1), D(2030, 1, 1))
    real = [D(2020, 10, 13), D(2022, 12, 8), D(2025, 1, 16), D(2027, 2, 19), D(2029, 3, 25)]
    assert len(found) == 5
    for t in real:
        assert np.min(np.abs(found - t)) < 2.0


def test_jupiter_saturn_conjunction_seen_from_earth(solar):
    found = find_alignments(solar, "Júpiter", "Saturno", D(2019, 1, 1), D(2022, 1, 1),
                            center="Terra")
    assert len(found) == 1 and abs(found[0] - D(2020, 12, 21)) < 3.0


def test_opposed_and_aligned_alternate(solar):
    aligned = find_alignments(solar, "Terra", "Marte", 0, 6000)
    opposed = find_alignments(solar, "Terra", "Marte", 0, 6000, kind="opposed")
    merged = np.sort(np.concatenate([aligned, opposed]))
    assert set(np.diff(np.searchsorted(merged, aligned)) % 2) == {0}  # nunca duas seguidas iguais
    with pytest.raises(ValueError):
        find_alignments(solar, "Terra", "Marte", 0, 10, kind="outro")


def test_known_eclipses_2025_2027_are_found_without_false_candidates(solar):
    found = find_eclipses(solar, "Terra", "Lua", "Sol", D(2025, 1, 1), D(2028, 1, 1))
    real = [("solar", (2025, 3, 29)), ("solar", (2025, 9, 21)), ("solar", (2026, 2, 17)),
            ("solar", (2026, 8, 12)), ("solar", (2027, 2, 6)), ("solar", (2027, 8, 2)),
            ("lunar", (2025, 3, 14)), ("lunar", (2025, 9, 7)), ("lunar", (2026, 3, 3)),
            ("lunar", (2026, 8, 28))]
    for kind, ymd in real:
        assert any(e.kind == kind and abs(e.t - D(*ymd)) < 1.5 for e in found), (kind, ymd)
    assert len(found) == len(real)  # os penumbrais de 2027 ficam de fora, por projeto


def test_solar_eclipse_count_in_the_21st_century_is_realistic(solar):
    found = find_eclipses(solar, "Terra", "Lua", "Sol", 0, T_MAX_DAYS)
    solar_n = sum(e.kind == "solar" for e in found)
    assert 210 <= solar_n <= 240  # os catálogos citam 224 para o século


def test_alignment_edges_recur_with_disjoint_windows():
    g = generate_solar_system(events=False)
    n = add_alignment_edges(g, "Terra", "Marte", 0.0, T_MAX_DAYS, relation="oposição")
    edges = sorted((e for e in g.edges if e.relation == "oposição"), key=lambda e: e.t)
    assert 46 <= n <= 48 and len(edges) == n
    assert all(a.t_end <= b.t for a, b in zip(edges, edges[1:], strict=False))
    t = (edges[3].t + edges[3].t_end) / 2
    active = [e for e in g.edges if e.relation == "oposição" and e.is_active(t)]
    assert len(active) == 1 and len(g.snapshot(t).get_edge_data("Terra", "Marte")) == 1


def test_resonances_are_the_known_ones(solar):
    pairs = {(r.a, r.b, f"{r.p}:{r.q}") for r in find_resonances(solar)}
    assert pairs == {
        ("Io", "Europa", "2:1"), ("Europa", "Ganimedes", "2:1"), ("Mimas", "Tétis", "2:1"),
        ("Encélado", "Dione", "2:1"), ("Titã", "Hipérion", "4:3"), ("Netuno", "Plutão", "3:2"),
    }


# -- N-corpos ----------------------------------------------------------------------------


def _initial_system():
    names = list(J2000_STATE)
    gm = np.array([to_au3_day2(1.32712440018e11)] + [to_au3_day2(SYSTEM_GM[n]) for n in names])
    x = np.array([J2000_STATE[n][:3] for n in names])
    v = np.array([J2000_STATE[n][3:] for n in names])
    sx = -np.sum(gm[1:, None] * x, 0) / gm.sum()
    sv = -np.sum(gm[1:, None] * v, 0) / gm.sum()
    return gm, np.vstack([sx, x + sx]), np.vstack([sv, v + sv])


def test_two_body_acceleration_is_newtonian():
    gm = np.array([4.0, 1.0])
    pos = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    a = accelerations(gm, pos)
    assert np.allclose(a[0], [1.0 / 4.0, 0, 0]) and np.allclose(a[1], [-4.0 / 4.0, 0, 0])


def test_yoshida4_is_more_accurate_than_leapfrog_on_a_kepler_orbit():
    gm = np.array([1.0, 1e-12])
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    vel = np.array([[0.0, 0, 0], [0.0, 1.0, 0]])  # órbita circular de período 2π
    steps = 100
    p2, _ = leapfrog(gm, pos, vel, steps, 2 * np.pi / steps)
    p4, _ = yoshida4(gm, pos, vel, steps, 2 * np.pi / steps)
    err2, err4 = (np.linalg.norm(p[-1, 1] - pos[1]) for p in (p2, p4))
    assert err4 < err2 / 50


def test_solar_system_energy_is_conserved():
    gm, pos, vel = _initial_system()
    p, v = yoshida4(gm, pos, vel, 1826, 1.0)  # 5 anos
    e0, e1 = total_energy(gm, p[0], v[0]), total_energy(gm, p[-1], v[-1])
    assert abs(e1 / e0 - 1) < 1e-6


def test_nbody_orbits_are_tables_with_velocity_and_are_cached():
    kep = {"Terra": KeplerOrbit(1.00000261, 0.01671123, 0.0, 100.46457166, 102.93768193, 0.0,
                                rates=(0, 0, 0, 35999.37244981, 0.32327364, 0))}
    first = nbody_orbits(kep, 400.0)
    assert nbody_orbits(kep, 400.0) is first
    orbit = first["Terra"]
    assert isinstance(orbit, TableOrbit) and orbit.vel is not None
    assert orbit.period_days == pytest.approx(365.2564, abs=0.01)
    assert np.allclose(orbit.position(0.0), J2000_STATE["Terra"][:3])


def test_nbody_solar_system_tracks_the_ephemeris_within_a_tenth_of_an_au():
    pytest.importorskip("astropy")
    g = generate_solar_system(dynamics="nbody", events=False)
    assert isinstance(g.nodes["Marte"].body.orbit, TableOrbit)
    assert isinstance(g.nodes["Ceres"].body.orbit, KeplerOrbit)  # corpos menores seguem Kepler
    for name, ref in PLANETS.items():
        t = D(2026, 10, 19)
        err = np.linalg.norm(g.position_at(name, t) - _astropy_helio(ref, t))
        assert err < 0.12, (name, err)


def test_graph_without_bodies_is_unaffected_by_the_celestial_package():
    g = Graph4D(t_max=3.0)
    g.add_node("a", (0, 0, 0, 0))
    assert not g.has_bodies and np.allclose(g.position_at("a", 99.0), 0.0)
