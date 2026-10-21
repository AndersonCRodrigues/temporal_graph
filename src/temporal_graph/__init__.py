"""Grafo espaço-tempo em R^4 (simulação didática, w = tempo)."""

from .generator import (
    generate_knowledge_demo,
    generate_random,
    generate_simplex4,
    generate_tesseract,
)
from .model import Body, Edge4D, Graph4D, Node4D
from .projection import project_points, rotation_matrix_4d
from .relativity import classify_interval, lorentz_boost_matrix, minkowski_interval
from .retrieval import RetrievalResult, build_context, retrieve

__version__ = "3.0.0"

__all__ = [
    "Body",
    "Edge4D",
    "Graph4D",
    "Node4D",
    "RetrievalResult",
    "__version__",
    "build_context",
    "classify_interval",
    "generate_knowledge_demo",
    "generate_random",
    "generate_simplex4",
    "generate_tesseract",
    "lorentz_boost_matrix",
    "minkowski_interval",
    "project_points",
    "retrieve",
    "rotation_matrix_4d",
]
