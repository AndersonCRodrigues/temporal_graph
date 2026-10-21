"""Recuperação temporal sobre o grafo, no estilo GraphRAG.

Passos, para uma consulta em um instante t:

  1. Snapshot: só vale o que existe em t (nós nascidos, arestas ativas).
  2. Sementes: os k nós do snapshot mais parecidos com a consulta.
  3. Expansão: vizinhos das sementes por até `hops` saltos, só por arestas ativas.
  4. Contexto: texto dos nós e das relações do subgrafo recuperado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .embedding import cosine, embed_text
from .model import Graph4D


@dataclass
class Hit:
    id: str
    score: float  # similaridade com a consulta (0 para vizinhos sem texto)
    hop: int  # 0 = semente; 1, 2... = distância da semente mais próxima


@dataclass
class RetrievalResult:
    query: str
    t: float
    hits: list[Hit] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def ids(self) -> set[str]:
        return {h.id for h in self.hits}

    def seeds(self) -> list[Hit]:
        return [h for h in self.hits if h.hop == 0]


def ensure_embeddings(graph: Graph4D) -> None:
    """Calcula o embedding dos nós que têm texto e ainda não têm vetor."""
    for n in graph.nodes.values():
        if n.embedding is None and n.text:
            n.embedding = embed_text(n.text)


def retrieve(
    graph: Graph4D,
    query: str,
    t: float | None = None,
    *,
    k: int = 3,
    hops: int = 1,
) -> RetrievalResult:
    """Busca em `graph` como ele era em `t` (padrão: t_max).

    Nós com similaridade 0 não viram semente, então uma consulta sem palavras
    em comum com o grafo devolve resultado vazio.
    """
    if k < 1:
        raise ValueError("k deve ser >= 1")
    if hops < 0:
        raise ValueError("hops deve ser >= 0")
    at = graph.t_max if t is None else float(t)
    result = RetrievalResult(query=query, t=at)

    ensure_embeddings(graph)
    snap = graph.snapshot(at)
    q = embed_text(query)
    scored = [
        (cosine(q, graph.nodes[nid].embedding), nid)
        for nid in snap.nodes
        if graph.nodes[nid].embedding is not None
    ]
    scored = sorted(((s, nid) for s, nid in scored if s > 0), key=lambda p: (-p[0], p[1]))[:k]
    if not scored:
        return result

    score_of = {nid: s for s, nid in scored}
    hop_of = dict.fromkeys(score_of, 0)
    frontier = set(hop_of)
    for depth in range(1, hops + 1):
        reached = {nb for nid in frontier for nb in snap.neighbors(nid)} - set(hop_of)
        hop_of.update(dict.fromkeys(reached, depth))
        frontier = reached

    result.hits = sorted(
        (Hit(nid, score_of.get(nid, 0.0), hop) for nid, hop in hop_of.items()),
        key=lambda h: (h.hop, -h.score, h.id),
    )
    result.edges = sorted({(min(a, b), max(a, b)) for a, b, *_ in snap.subgraph(hop_of).edges})
    return result


def build_context(graph: Graph4D, result: RetrievalResult) -> str:
    """Texto pronto para entrar num prompt: nós por proximidade, depois relações."""
    if not result.hits:
        return ""
    lines = [f"Contexto em t = {result.t:g}"]
    for h in result.hits:
        text = graph.nodes[h.id].text or "(sem texto)"
        kind = "semente" if h.hop == 0 else f"vizinho, {h.hop} salto(s)"
        lines.append(f"- {h.id} [{kind}]: {text}")
    shown = [
        e for e in graph.edges
        if e.is_active(result.t) and e.src in result.ids() and e.dst in result.ids()
    ]
    for e in sorted(shown, key=lambda e: (e.src, e.dst, e.relation, e.t)):
        end = "..." if e.t_end is None else f"{e.t_end:g}"
        relation = e.relation or "relacionado_a"
        lines.append(f"- {e.src} --{relation}--> {e.dst}  [vale de {e.t:g} até {end})")
    return "\n".join(lines)
