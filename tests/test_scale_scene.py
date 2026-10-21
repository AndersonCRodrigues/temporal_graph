"""Testes da escala de exibição, do quadro de cena e do cone de luz."""

from __future__ import annotations

import numpy as np
import pytest

from temporal_graph.celestial.catalog import generate_solar_system
from temporal_graph.celestial.scale import (
    DisplayScale,
    display_position,
    helio_distance,
    moon_distance,
    visual_radius,
)
from temporal_graph.celestial.scene import W_EXTENT, SceneOptions, build_frame
from temporal_graph.frame import ConeSpec, cone_surfaces, static_frame
from temporal_graph.model import Graph4D
from temporal_graph.relativity import classify_interval

T = 9787.5  # 2026-10-19


@pytest.fixture(scope="module")
def solar():
    return generate_solar_system(events=False)


def test_scale_mode_is_validated():
    with pytest.raises(ValueError):
        DisplayScale(mode="cubica")


@pytest.mark.parametrize("mode", ["log", "sqrt"])
def test_helio_compression_is_monotonic_and_compresses(mode):
    s = DisplayScale(mode=mode)
    d = np.array([0.39, 1.0, 5.2, 30.0, 500.0])
    shown = helio_distance(d, s)
    assert np.all(np.diff(shown) > 0)
    assert shown[-1] / shown[0] < d[-1] / d[0] / 10  # a razão real é 1300; a exibida, bem menor


def test_real_mode_is_linear():
    s = DisplayScale(mode="real")
    assert helio_distance(10.0, s) == pytest.approx(10 * helio_distance(1.0, s))
    assert moon_distance(2.0, 0.1, 0.1, s) == pytest.approx(2 * moon_distance(1.0, 0.1, 0.1, s))


def test_moons_are_pushed_outside_both_spheres():
    s = DisplayScale()
    assert moon_distance(0.0026, 0.15, 0.1, s) > 0.15 + 0.1


def test_visual_radius_orders_the_bodies(solar):
    s = DisplayScale()
    sun, jup, earth, ceres, comet = (visual_radius(r, s) for r in (695700, 69911, 6371, 470, 5))
    assert sun > jup > earth > ceres >= comet >= 0.025
    assert visual_radius(6371, DisplayScale(size_boost=2.0)) > visual_radius(6371, s)


def test_sun_is_at_the_origin_and_directions_are_preserved(solar):
    s = DisplayScale()
    assert np.allclose(display_position(solar, "Sol", T, s), 0.0)
    shown = display_position(solar, "Marte", T, s)
    real = solar.position_at("Marte", T)
    assert np.allclose(shown / np.linalg.norm(shown), real / np.linalg.norm(real))


def test_moon_stays_next_to_its_planet_at_any_time(solar):
    s = DisplayScale()
    t = np.linspace(T, T + 400, 20)
    gap = display_position(solar, "Lua", t, s) - display_position(solar, "Terra", t, s)
    assert gap.shape == (20, 3) and np.all(np.linalg.norm(gap, axis=1) < 1.0)


def test_current_position_changes_with_time_but_graph_size_does_not(solar):
    s = DisplayScale()
    n = len(solar.nodes)
    a, b = display_position(solar, "Mercúrio", T, s), display_position(solar, "Mercúrio", T + 30, s)
    assert np.linalg.norm(a - b) > 0.1 and len(solar.nodes) == n


def test_compressed_frame_has_current_positions_and_optional_trails(solar):
    plain = build_frame(solar, T, SceneOptions())
    assert plain.trails == {} and len(plain.pos) == len(solar.nodes)
    assert all(p[3] == 0.0 for p in plain.pos.values())
    assert set(plain.radius) == set(solar.nodes) and plain.color["Sol"]
    trails = build_frame(solar, T, SceneOptions(trails=True, trail_span=3650.0))
    assert "Sol" not in trails.trails and "Terra" in trails.trails
    w = trails.trails["Terra"][:, 3]
    assert w.max() == pytest.approx(0.0) and -W_EXTENT <= w.min() < 0


def test_trail_never_goes_before_the_body_existed(solar):
    frame = build_frame(solar, 5.0, SceneOptions(trails=True, trail_span=3650.0))
    assert frame.trails["Terra"][:, 3].min() >= -5.0 / 3650.0 * W_EXTENT - 1e-9


def test_short_period_bodies_trail_only_one_orbit(solar):
    frame = build_frame(solar, T, SceneOptions(trails=True, trail_span=3650.0))
    phobos = frame.trails["Fobos"]
    assert phobos[:, 3].min() >= -W_EXTENT * 0.32 / 3650.0 * 1.01


def test_minkowski_frame_puts_the_observer_at_the_origin_event(solar):
    frame = build_frame(solar, T, SceneOptions(minkowski=True, observer="Terra"))
    assert np.allclose(frame.pos["Terra"], 0.0)
    assert frame.cone is not None and frame.cone.radius == pytest.approx(solar.light_speed * 0.25)
    assert "Sedna" not in frame.visible and "Júpiter" in frame.visible


def test_retarded_events_lie_on_the_past_light_cone(solar):
    frame = build_frame(solar, T, SceneOptions(minkowski=True, observer="Terra"))
    origin = np.zeros(4)
    assert "Sol" in frame.markers and "Terra" not in frame.markers
    for nid, event in frame.markers.items():
        assert event[3] < 0, nid
        assert classify_interval(origin, event, tol=1e-6) == "tipo-luz", nid
    sun_distance = np.linalg.norm(frame.pos["Sol"][:3])
    assert abs(frame.markers["Sol"][3]) == pytest.approx(sun_distance, rel=1e-3)
    assert "luz do Sol" in frame.footer and "8." in frame.footer


def test_minkowski_unknown_observer_falls_back_to_a_body(solar):
    frame = build_frame(solar, T, SceneOptions(minkowski=True, observer="Nenhum"))
    assert any(np.allclose(p, 0.0) for p in frame.pos.values())


def test_cone_surfaces_are_lightlike():
    cone = ConeSpec(np.array([1.0, 2.0, 3.0, 4.0]), 5.0)
    for surface in cone_surfaces(cone):
        rel = surface.reshape(-1, 4) - cone.center
        assert np.allclose(np.hypot(rel[:, 0], rel[:, 1]), np.abs(rel[:, 3]), atol=1e-12)
        assert np.all(surface[..., 2] == 3.0)
    past, future = cone_surfaces(cone)
    assert past[..., 3].max() <= 4.0 <= future[..., 3].min()


def test_static_frame_recenters_on_the_observer_and_adds_a_cone():
    g = Graph4D(t_max=5.0)
    g.add_node("a", (1, 1, 1, 1))
    g.add_node("b", (3, 1, 1, 2))
    frame = static_frame(g, ["a", "b"], observer="a", cone=True)
    assert np.allclose(frame.pos["a"], 0.0) and np.allclose(frame.pos["b"], [2, 0, 0, 1])
    assert frame.cone is not None and frame.cone.radius >= 2.0
    assert static_frame(g, ["a", "b"]).cone is None
