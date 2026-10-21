"""Testes do embedding lexical e do PCA 3D."""

from __future__ import annotations

import numpy as np

from temporal_graph.embedding import DIM, cosine, embed_text, project_3d, tokenize


def test_tokenize_drops_accents_stopwords_and_short_words():
    assert tokenize("A engenheira de Dados é ÓTIMA!") == ["engenheira", "dados", "otima"]


def test_embedding_is_unit_and_deterministic():
    a, b = embed_text("plataforma de dados"), embed_text("plataforma de dados")
    np.testing.assert_array_equal(a, b)
    assert a.shape == (DIM,)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_empty_text_gives_zero_vector_and_zero_similarity():
    z = embed_text("de a o")
    assert not z.any()
    assert cosine(z, embed_text("dados")) == 0.0


def test_shared_words_raise_similarity():
    q = embed_text("análise de dados")
    assert cosine(q, embed_text("plataforma de análise de dados")) > cosine(
        q, embed_text("servidor de nuvem")
    )


def test_no_semantics_is_the_documented_limit():
    assert cosine(embed_text("carro"), embed_text("automóvel")) == 0.0


def test_project_3d_shape_range_and_determinism():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(9, 40))
    out = project_3d(vecs, extent=2.0)
    assert out.shape == (9, 3)
    assert np.isclose(np.abs(out).max(), 2.0)
    np.testing.assert_array_equal(out, project_3d(vecs, extent=2.0))


def test_project_3d_edge_cases():
    assert project_3d(np.zeros((0, 5))).shape == (0, 3)
    assert project_3d(np.zeros((4, 5))).shape == (4, 3)
    assert project_3d(np.ones((2, 5))).shape == (2, 3)
