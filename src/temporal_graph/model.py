"""Modelo do grafo espaço-tempo em R^4.

  (x, y, z)  coordenadas espaciais
  w          tempo de nascimento do nó

Uma aresta existe no intervalo [t, t_end): começa em t e, se t_end for
definido, deixa de valer em t_end. Ela nunca começa antes do nascimento dos
dois extremos. Nós têm texto e embedding opcionais, usados na recuperação.

A divisão train/val/test depende só de w relativo a t_max (60% / 20% / 20%).

Corpos em movimento (planetas, luas...) são UM nó cada, com uma `Body` que
guarda a órbita relativa ao pai. A posição em qualquer instante sai de
`position_at`; o grafo não cresce com o número de amostras no tempo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import networkx as nx
import numpy as np

TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.8


class Trajectory(Protocol):
    """Qualquer objeto que dá a posição (..., 3) relativa ao pai em função de t."""

    def position(self, t) -> np.ndarray: ...


@dataclass
class Body:
    """Informação física de um nó que representa um corpo celeste."""

    kind: str = "planeta"  # estrela | planeta | planeta-anão | lua | asteroide | cometa
    parent: str | None = None  # id do corpo que ele orbita (None = origem do referencial)
    orbit: Trajectory | None = field(default=None, repr=False)  # posição relativa ao pai, em UA
    radius_km: float = 0.0
    gm_km3_s2: float = 0.0
    color: str | None = None
    real_phase: bool = True  # False: forma da órbita real, fase ilustrativa


@dataclass
class Node4D:
    id: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0
    label: int = 0
    split: str = "train"
    text: str = ""
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    body: Body | None = field(default=None, repr=False, compare=False)

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=float)

    @coords.setter
    def coords(self, value: Sequence[float]) -> None:
        values = [float(v) for v in value]
        if len(values) != 4:
            raise ValueError("coords deve ter 4 valores: (x, y, z, w)")
        self.x, self.y, self.z, self.w = values

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass
class Edge4D:
    src: str
    dst: str
    t: float = 0.0
    t_end: float | None = None
    relation: str = ""

    def endpoints(self) -> set[str]:
        return {self.src, self.dst}

    def is_active(self, t: float) -> bool:
        """Verdadeiro se a aresta vale no instante t, no intervalo [t, t_end)."""
        return self.t <= t and (self.t_end is None or t < self.t_end)

    def overlaps(self, start: float, end: float | None) -> bool:
        """Verdadeiro se [t, t_end) e [start, end) têm instantes em comum."""
        return (end is None or self.t < end) and (self.t_end is None or start < self.t_end)


@dataclass
class Graph4D:
    nodes: dict[str, Node4D] = field(default_factory=dict)
    edges: list[Edge4D] = field(default_factory=list)
    t_max: float = 6.0
    epoch: datetime | None = None  # data de t = 0; com ela, a unidade de t é o dia
    light_speed: float = 1.0  # distância percorrida pela luz por unidade de t (c = 1 por padrão)
    t_default: float | None = None  # instante inicial sugerido para a visualização
    t_step: float = 0.5  # passo sugerido para o controle de tempo

    @property
    def has_bodies(self) -> bool:
        return any(n.body is not None for n in self.nodes.values())

    def time_label(self, t: float) -> str:
        """Texto para o instante t: a data, se o grafo tiver época, senão o número."""
        if self.epoch is None:
            return f"{t:g}"
        return (self.epoch + timedelta(days=float(t))).strftime("%Y-%m-%d")

    # -- posição de corpos --------------------------------------------------

    def children(self, nid: str) -> list[str]:
        return [n.id for n in self.nodes.values() if n.body and n.body.parent == nid]

    def local_position_at(self, nid: str, t) -> np.ndarray:
        """Posição em UA relativa ao pai (zero para quem não tem órbita)."""
        t = np.asarray(t, dtype=float)
        body = self.nodes[nid].body
        if body is None or body.orbit is None:
            return np.zeros((*t.shape, 3))
        return np.asarray(body.orbit.position(t), dtype=float)

    def position_at(self, nid: str, t) -> np.ndarray:
        """Posição (..., 3) no instante t. Nós sem corpo ficam onde estão."""
        t = np.asarray(t, dtype=float)
        node = self.nodes[nid]
        if node.body is None:
            return np.broadcast_to(node.coords[:3], (*t.shape, 3)).copy()
        pos = self.local_position_at(nid, t)
        if node.body.parent is not None:
            pos = pos + self.position_at(node.body.parent, t)
        return pos

    # -- divisão temporal -------------------------------------------------

    def _recompute_split(self) -> None:
        limit_train = self.t_max * TRAIN_FRACTION
        limit_val = self.t_max * VAL_FRACTION
        for n in self.nodes.values():
            if n.w <= limit_train:
                n.split = "train"
            elif n.w <= limit_val:
                n.split = "val"
            else:
                n.split = "test"

    def _enforce_edge_times(self, nid: str) -> None:
        """Empurra para frente as arestas de `nid` que começariam antes dos extremos.

        A duração da aresta é preservada: se t_end existe, ele acompanha o deslocamento.
        """
        for e in self.edges:
            if nid in (e.src, e.dst):
                start = max(e.t, self.nodes[e.src].w, self.nodes[e.dst].w)
                if e.t_end is not None:
                    e.t_end += start - e.t
                e.t = start

    # -- edição -----------------------------------------------------------

    def add_node(
        self,
        nid: str,
        coords: Sequence[float] | None = None,
        *,
        label: int = 0,
        t_birth: float | None = None,
        text: str = "",
        embedding: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Adiciona um nó. Retorna False se o id for vazio ou já existir.

        Sem `coords`, sorteia (x, y, z) em [-2, 2] e w em [0, t_max].
        Se `t_birth` for dado, ele sobrescreve w.
        """
        if not nid or nid in self.nodes:
            return False
        if coords is None:
            gen = rng or np.random.default_rng()
            c = [*gen.uniform(-2, 2, size=3), gen.uniform(0, self.t_max)]
        else:
            c = [float(v) for v in coords]
            if len(c) != 4:
                raise ValueError("coords deve ter 4 valores: (x, y, z, w)")
        if t_birth is not None:
            c[3] = float(t_birth)
        self.nodes[nid] = Node4D(
            nid, *map(float, c), label=int(label), text=text, embedding=embedding
        )
        self._recompute_split()
        return True

    def add_body(
        self,
        nid: str,
        *,
        orbit: Trajectory | None = None,
        parent: str | None = None,
        kind: str = "planeta",
        radius_km: float = 0.0,
        gm_km3_s2: float = 0.0,
        color: str | None = None,
        text: str = "",
        t_birth: float = 0.0,
        real_phase: bool = True,
    ) -> bool:
        """Adiciona um corpo celeste como um único nó, com órbita relativa ao pai."""
        if parent is not None and parent not in self.nodes:
            return False
        if not self.add_node(nid, coords=(0, 0, 0, t_birth), text=text):
            return False
        node = self.nodes[nid]
        node.body = Body(kind, parent, orbit, radius_km, gm_km3_s2, color, real_phase)
        node.coords = (*self.position_at(nid, t_birth), t_birth)
        return True

    def set_text(self, nid: str, text: str) -> bool:
        """Troca o texto do nó e descarta o embedding antigo."""
        if nid not in self.nodes:
            return False
        self.nodes[nid].text = text
        self.nodes[nid].embedding = None
        return True

    def add_edge(
        self,
        src: str,
        dst: str,
        t: float = 0.0,
        *,
        t_end: float | None = None,
        relation: str = "",
    ) -> bool:
        """Adiciona uma aresta não direcionada, sem laços.

        O mesmo par pode ter várias arestas, desde que cada relação tenha
        intervalos sem sobreposição: assim uma relação pode reaparecer
        (oposições, eclipses) e relações diferentes convivem. Se `t` for
        anterior ao nascimento de algum extremo, é ajustado para o instante em
        que ambos já existem. Retorna False se `t_end` não for posterior ao
        início efetivo ou se houver sobreposição com a mesma relação.
        """
        if src == dst or src not in self.nodes or dst not in self.nodes:
            return False
        start = max(float(t), self.nodes[src].w, self.nodes[dst].w)
        end = None if t_end is None else float(t_end)
        if end is not None and end <= start:
            return False
        for e in self.edges:
            if e.endpoints() == {src, dst} and e.relation == relation and e.overlaps(start, end):
                return False
        self.edges.append(Edge4D(src, dst, start, end, relation))
        return True

    def remove_node(self, nid: str) -> bool:
        """Remove o nó, as arestas dele e, se for um corpo, os corpos que o orbitam."""
        if nid not in self.nodes:
            return False
        doomed = {nid}
        stack = [nid]
        while stack:
            for child in self.children(stack.pop()):
                if child not in doomed:
                    doomed.add(child)
                    stack.append(child)
        for d in doomed:
            del self.nodes[d]
        self.edges = [e for e in self.edges if not (e.endpoints() & doomed)]
        return True

    def remove_edge(self, src: str, dst: str, relation: str | None = None) -> bool:
        """Remove todas as arestas do par (só as da `relation`, se dada)."""
        before = len(self.edges)
        self.edges = [
            e for e in self.edges
            if not (e.endpoints() == {src, dst} and (relation is None or e.relation == relation))
        ]
        return len(self.edges) < before

    def set_coords(self, nid: str, coords: Sequence[float]) -> bool:
        if nid not in self.nodes:
            return False
        self.nodes[nid].coords = coords
        self._recompute_split()
        self._enforce_edge_times(nid)
        return True

    def set_t_max(self, t_max: float) -> None:
        self.t_max = max(1.0, float(t_max))
        self._recompute_split()

    # -- consulta ---------------------------------------------------------

    def snapshot(self, t: float) -> nx.MultiGraph:
        """Subgrafo em t: nós com w <= t e arestas ativas em t (t <= t < t_end).

        Corpos trazem `pos`, a posição em UA em t (não a de nascimento). É um
        multigrafo porque duas relações diferentes podem valer ao mesmo tempo.
        """
        g = nx.MultiGraph()
        for n in self.nodes.values():
            if n.w <= t:
                attrs = {"label": n.label, "split": n.split, "t_birth": n.w, "text": n.text}
                if n.body is not None:
                    attrs["pos"] = tuple(float(v) for v in self.position_at(n.id, t))
                g.add_node(n.id, **attrs)
        for e in self.edges:
            if e.is_active(t) and e.src in g and e.dst in g:
                g.add_edge(e.src, e.dst, t=e.t, t_end=e.t_end, relation=e.relation)
        return g

    def to_networkx(self) -> nx.MultiGraph:
        g = nx.MultiGraph()
        for n in self.nodes.values():
            g.add_node(
                n.id, label=n.label, split=n.split, t_birth=n.w, coords=n.as_tuple(), text=n.text
            )
        for e in self.edges:
            g.add_edge(e.src, e.dst, t=e.t, t_end=e.t_end, relation=e.relation)
        return g

    def coords_matrix(self) -> np.ndarray:
        """Matriz (N, 4) com os nós em ordem alfabética de id."""
        ids = sorted(self.nodes)
        if not ids:
            return np.zeros((0, 4))
        return np.array([self.nodes[i].coords for i in ids])

    def stats(self) -> dict[str, float]:
        splits = [n.split for n in self.nodes.values()]
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "t_max": self.t_max,
            "n_train": splits.count("train"),
            "n_val": splits.count("val"),
            "n_test": splits.count("test"),
        }
