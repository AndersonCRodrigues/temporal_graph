"""Testes das órbitas keplerianas e tabeladas."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.celestial.orbits import (
    AU_KM,
    J2000_JD,
    KeplerOrbit,
    TableOrbit,
    pole_frame,
    solve_kepler,
)


@pytest.mark.parametrize("e", [0.0, 0.0167, 0.5, 0.8, 0.967, 0.995, 0.999])
def test_kepler_equation_residual(e):
    m = np.linspace(-np.pi, np.pi, 2001)
    ecc = solve_kepler(m, e)
    resid = ecc - e * np.sin(ecc) - m
    assert np.max(np.abs((resid + np.pi) % (2 * np.pi) - np.pi)) < 1e-11


def test_circular_orbit_radius_period_and_quarter_turn():
    orbit = KeplerOrbit.from_period(2.0, 0.0, 0.0, 100.0)
    p = orbit.position(np.array([0.0, 25.0, 100.0]))
    assert np.allclose(np.linalg.norm(p, axis=1), 2.0)
    assert np.allclose(p[0], p[2], atol=1e-9)
    assert abs(np.dot(p[0], p[1])) < 1e-9  # um quarto de volta: perpendicular
    assert orbit.period_days == pytest.approx(100.0)


def test_inclination_limits_height():
    orbit = KeplerOrbit.from_period(1.0, 0.0, 30.0, 50.0)
    z = orbit.position(np.linspace(0, 50, 400))[:, 2]
    assert z.max() == pytest.approx(np.sin(np.radians(30)), abs=1e-3)


def test_perihelion_and_aphelion_distances():
    e, a, t_peri = 0.9, 5.0, 123.0
    orbit = KeplerOrbit.from_perihelion(a, e, 10.0, node=40.0, argp=70.0, t_peri_days=t_peri)
    half = orbit.period_days / 2
    assert np.linalg.norm(orbit.position(t_peri)) == pytest.approx(a * (1 - e), rel=1e-9)
    assert np.linalg.norm(orbit.position(t_peri + half)) == pytest.approx(a * (1 + e), rel=1e-6)


def test_orbit_accepts_scalar_and_arrays():
    orbit = KeplerOrbit.from_period(1.0, 0.1, 5.0, 365.0)
    assert orbit.position(3.0).shape == (3,)
    assert orbit.position(np.zeros((2, 4))).shape == (2, 4, 3)


def test_pole_frame_is_orthonormal_and_saturn_is_tilted():
    f = pole_frame(40.589, 83.537)
    assert np.allclose(f.T @ f, np.eye(3), atol=1e-12)
    tilt = np.degrees(np.arccos(f[2, 2]))  # ângulo entre o polo e o norte da eclíptica
    assert 27.0 < tilt < 29.0


def test_frame_rotates_the_orbit_plane():
    flat = KeplerOrbit.from_period(1.0, 0.0, 0.0, 10.0)
    tilted = KeplerOrbit.from_period(1.0, 0.0, 0.0, 10.0, frame=pole_frame(40.589, 83.537))
    assert np.max(np.abs(flat.position(np.linspace(0, 10, 50))[:, 2])) < 1e-12
    assert np.max(np.abs(tilted.position(np.linspace(0, 10, 50))[:, 2])) > 0.3


def _circle_table(n, with_velocity):
    t = np.linspace(0, 10, n)
    w = 2 * np.pi / 10
    pos = np.column_stack([np.cos(w * t), np.sin(w * t), np.zeros(n)])
    vel = np.column_stack([-w * np.sin(w * t), w * np.cos(w * t), np.zeros(n)])
    return TableOrbit(t, pos, vel if with_velocity else None, period_days=10.0)


def test_table_hermite_beats_linear():
    q = np.linspace(0.3, 9.7, 200)
    truth = np.column_stack([np.cos(2 * np.pi / 10 * q), np.sin(2 * np.pi / 10 * q), 0 * q])
    err_h = np.max(np.abs(_circle_table(21, True).position(q) - truth))
    err_l = np.max(np.abs(_circle_table(21, False).position(q) - truth))
    assert err_h < 1e-4 < err_l


def test_table_clamps_outside_range_and_validates():
    orbit = _circle_table(11, True)
    assert np.allclose(orbit.position(-5.0), orbit.position(0.0))
    assert np.allclose(orbit.position(99.0), orbit.position(10.0))
    with pytest.raises(ValueError):
        TableOrbit([0, 0], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        TableOrbit([0, 1, 2], np.zeros((2, 3)))


def test_table_from_csv_units_and_time(tmp_path):
    path = tmp_path / "orbit.csv"
    path.write_text(f"jd,x,y,z\n{J2000_JD},{AU_KM},0,0\n{J2000_JD + 1},0,{AU_KM},0\n")
    orbit = TableOrbit.from_csv(path, time="jd", unit="km")
    assert np.allclose(orbit.position(0.0), [1, 0, 0])
    assert np.allclose(orbit.position(1.0), [0, 1, 0])
    assert np.allclose(orbit.position(0.5), [0.5, 0.5, 0])
    with pytest.raises(ValueError):
        TableOrbit.from_csv(path, unit="parsec")
