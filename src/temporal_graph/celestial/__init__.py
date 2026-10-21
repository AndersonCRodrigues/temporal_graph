"""Corpos celestes sobre o grafo 4D: órbitas, catálogo do sistema solar, escala.

Este subpacote usa o núcleo (`temporal_graph.model`), mas o núcleo não o
importa: um grafo 4D funciona sem nada daqui.
"""

from .catalog import describe_body, generate_solar_system
from .events import add_alignment_edges, add_eclipse_edges, add_resonance_edges
from .orbits import AU_KM, J2000_JD, KeplerOrbit, TableOrbit, pole_frame
from .scale import DisplayScale
from .scene import SceneOptions, build_frame

__all__ = [
    "AU_KM",
    "J2000_JD",
    "DisplayScale",
    "KeplerOrbit",
    "SceneOptions",
    "TableOrbit",
    "add_alignment_edges",
    "add_eclipse_edges",
    "add_resonance_edges",
    "build_frame",
    "describe_body",
    "generate_solar_system",
    "pole_frame",
]
