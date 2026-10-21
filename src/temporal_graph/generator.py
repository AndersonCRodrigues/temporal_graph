"""Geradores de grafos espaço-tempo: aleatório, tesseract e 4-simplex."""

from __future__ import annotations

import itertools

import numpy as np

from .embedding import embed_text, project_3d
from .model import Graph4D


def generate_random(
    n_nodes: int = 15,
    n_edges: int = 25,
    n_steps: int = 6,
    seed: int = 42,
) -> Graph4D:
    """Grafo aleatório reprodutível.

    (x, y, z) em [-2, 2]; w inteiro em [0, n_steps]. O rótulo de cada nó é 1
    se o grau final for >= 3, senão 0.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes deve ser >= 1")
    if n_edges < 0:
        raise ValueError("n_edges deve ser >= 0")
    if n_steps < 1:
        raise ValueError("n_steps deve ser >= 1")

    rng = np.random.default_rng(seed)
    g = Graph4D(t_max=float(n_steps))
    ids = [f"S{i}" for i in range(n_nodes)]

    for nid in ids:
        xyz = rng.uniform(-2, 2, size=3)
        w = float(rng.integers(0, n_steps + 1))
        g.add_node(nid, coords=(*xyz, w))

    max_pairs = n_nodes * (n_nodes - 1) // 2
    target = min(n_edges, max_pairs)
    tried: set[tuple[int, int]] = set()
    max_tries = max(target * 30, 100)
    for _ in range(max_tries):
        if len(g.edges) >= target:
            break
        a, b = sorted(int(i) for i in rng.choice(n_nodes, 2, replace=False))
        if (a, b) in tried:
            continue
        tried.add((a, b))
        g.add_edge(ids[a], ids[b], t=float(rng.integers(0, n_steps + 1)))

    degree = dict.fromkeys(ids, 0)
    for e in g.edges:
        degree[e.src] += 1
        degree[e.dst] += 1
    for nid in ids:
        g.nodes[nid].label = int(degree[nid] >= 3)
    return g


def generate_tesseract(scale: float = 1.0) -> Graph4D:
    """Tesseract: 16 vértices, 32 arestas. w vai de 0 a 2*scale."""
    g = Graph4D(t_max=2.0 * scale)
    signs = list(itertools.product((-1.0, 1.0), repeat=4))
    ids = [f"T{i}" for i in range(len(signs))]
    for nid, s in zip(ids, signs, strict=True):
        g.add_node(nid, coords=(scale * s[0], scale * s[1], scale * s[2], (s[3] + 1.0) * scale))
    for (i, a), (j, b) in itertools.combinations(enumerate(signs), 2):
        if sum(u != v for u, v in zip(a, b, strict=True)) == 1:
            g.add_edge(ids[i], ids[j])
    return g


def generate_simplex4(scale: float = 1.5) -> Graph4D:
    """4-simplex (K5). Parte espacial com raio `scale`; w reescalado para [0, 4]."""
    s5 = np.sqrt(5)
    raw = np.array(
        [
            [1, 1, 1, -1 / s5],
            [1, -1, -1, -1 / s5],
            [-1, 1, -1, -1 / s5],
            [-1, -1, 1, -1 / s5],
            [0, 0, 0, s5 - 1 / s5],
        ],
        dtype=float,
    )
    raw[:, :3] -= raw[:, :3].mean(axis=0)
    raw[:, :3] *= scale / (np.linalg.norm(raw[0, :3]) or 1.0)
    w = raw[:, 3]
    raw[:, 3] = (w - w.min()) / (w.max() - w.min()) * 4.0

    g = Graph4D(t_max=4.0)
    ids = [f"P{i}" for i in range(5)]
    for nid, c in zip(ids, raw, strict=True):
        g.add_node(nid, coords=c)
    for i, j in itertools.combinations(range(5), 2):
        g.add_edge(ids[i], ids[j])
    return g


KNOWLEDGE_DEMO_QUERY = "quem trabalha no Atlas"

# (id, nascimento, texto)
_DEMO_NODES = [
    ("Acme", 0, "Acme, empresa de software que desenvolve plataformas e aplicativos."),
    ("Ana", 0, "Ana, engenheira de dados. Constrói pipelines e análises de dados."),
    ("Bruno", 0, "Bruno, gerente de produto. Define prioridades e conversa com clientes."),
    ("Atlas", 1, "Projeto Atlas, plataforma de análise de dados para varejo."),
    ("Orion", 1, "Orion, servidor de nuvem que hospeda plataformas e aplicativos."),
    ("Carla", 2, "Carla, designer de interface. Desenha telas de aplicativos."),
    ("Boreal", 3, "Projeto Boreal, aplicativo móvel de compras."),
    ("Delta", 4, "Delta, cliente do varejo que usa a plataforma de análise."),
    ("Diego", 5, "Diego, analista de segurança. Revisa o acesso aos servidores."),
]

# (origem, destino, início, fim ou None, relação). O intervalo é [início, fim).
_DEMO_EDGES = [
    ("Ana", "Acme", 0, None, "trabalha_na"),
    ("Bruno", "Acme", 0, None, "trabalha_na"),
    ("Carla", "Acme", 2, None, "trabalha_na"),
    ("Ana", "Atlas", 1, 5, "constrói"),
    ("Bruno", "Atlas", 1, None, "gerencia"),
    ("Atlas", "Orion", 1, None, "hospedado_em"),
    ("Carla", "Boreal", 3, None, "desenha"),
    ("Boreal", "Orion", 3, None, "hospedado_em"),
    ("Atlas", "Delta", 4, 7, "atende"),
    ("Ana", "Boreal", 5, None, "constrói"),
    ("Diego", "Orion", 5, None, "revisa_acesso"),
]


def generate_knowledge_demo() -> Graph4D:
    """Mini base de conhecimento temporal, fictícia, para testar a recuperação.

    Cada nó tem texto; (x, y, z) vem de um PCA dos embeddings (só para
    visualizar) e w é o instante em que a entidade aparece. As arestas têm
    início, fim opcional e relação. Ana deixa o Atlas em t = 5 e o contrato
    com a Delta termina em t = 7: a mesma consulta dá contextos diferentes
    conforme o instante.
    """
    g = Graph4D(t_max=8.0)
    xyz = project_3d(np.array([embed_text(text) for _, _, text in _DEMO_NODES]))
    for (nid, born, text), pos in zip(_DEMO_NODES, xyz, strict=True):
        g.add_node(nid, coords=(*pos, float(born)), text=text)
    for src, dst, start, end, relation in _DEMO_EDGES:
        g.add_edge(src, dst, t=float(start), t_end=None if end is None else float(end),
                   relation=relation)
    return g
