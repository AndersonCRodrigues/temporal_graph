"""Embedding lexical por hashing, sem dependências externas.

Cada palavra (sem acento, sem stopwords) cai num de DIM baldes pelo hash dela
e o vetor é normalizado. Dois textos são parecidos quando compartilham
palavras. Não há semântica: "carro" e "automóvel" não se aproximam.

Serve para o recuperador funcionar offline e de forma determinística. Palavras
diferentes podem cair no mesmo balde (colisão) e gerar uma similaridade
espúria pequena; com DIM = 1024 isso é raro, mas acontece. Num sistema real,
troque `embed_text` por um modelo de embeddings de verdade.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

DIM = 1024
STOPWORDS = frozenset({
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "e", "ou", "que", "se",
    "ao", "aos", "ser", "foi", "sao", "quem", "qual", "quais", "como", "onde", "quando",
})


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", _strip_accents(text))
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def embed_text(text: str, dim: int = DIM) -> np.ndarray:
    """Vetor unitário de dimensão `dim`. Texto sem palavras úteis dá vetor nulo."""
    vec = np.zeros(dim)
    for word in tokenize(text):
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        vec[int.from_bytes(digest, "big") % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade de cosseno. 0.0 se algum dos vetores for nulo."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def project_3d(vectors: np.ndarray, extent: float = 2.0) -> np.ndarray:
    """PCA para 3 componentes, reescalado para caber em [-extent, extent].

    Só para posicionar nós na cena. A projeção perde informação: distância no
    gráfico não é similaridade. O sinal de cada componente é fixado para o
    resultado não oscilar entre execuções.
    """
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        return np.zeros((0, 3))
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = np.zeros((3, x.shape[1]))
    for i, row in enumerate(vt[:3]):
        comps[i] = -row if row[np.argmax(np.abs(row))] < 0 else row
    out = centered @ comps.T
    peak = np.abs(out).max()
    return out * (extent / peak) if peak else out
