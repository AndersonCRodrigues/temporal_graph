# Modelo e convenções

## Coordenadas

Cada nó é um ponto `(x, y, z, w)`. `(x, y, z)` é a posição no espaço e `w` é o instante de nascimento. Não há "morte" de nós: depois de nascer, o nó permanece.

Cada aresta vale no intervalo `[t, t_end)`: começa em `t` e, se `t_end` estiver definido, deixa de valer nele. Sem `t_end`, vale para sempre. Também pode ter uma `relation` (texto curto, usado no contexto da busca).

`Graph4D.add_edge` nunca aceita `t` menor que o nascimento de um dos extremos; nesse caso `t` é elevado para `max(w_src, w_dst)`, e se `t_end` não ficar posterior a esse início a aresta é recusada. Quando um nó é movido para um `w` maior, as arestas dele são empurradas do mesmo jeito, e `t_end` acompanha o deslocamento para manter a duração. Só há uma aresta por par de nós, ou seja, uma relação não pode reaparecer depois de expirar.

Nós podem ter `text` e `embedding`, usados só na recuperação.

## Corpos e trajetórias

Um nó pode ser um evento (um ponto fixo `(x, y, z, w)`) ou um corpo com trajetória (`Body`, com uma órbita relativa ao pai). Nesse caso o nó guarda as coordenadas do nascimento, e a posição em qualquer instante vem de `position_at(id, t)`. O grafo não cresce com o número de amostras no tempo. Detalhes e o sistema solar completo em [`celestial.md`](celestial.md).

O mesmo par de nós pode ter várias arestas: cada relação precisa ter janelas sem sobreposição, e relações diferentes coexistem. Por isso `snapshot` e `to_networkx` devolvem um `MultiGraph`.

## Snapshot

`snapshot(t)` devolve o subgrafo com os nós de `w <= t` e as arestas ativas em `t` (`t_aresta <= t < t_end`) cujos dois extremos estão presentes. Na interface, o slider de `t` aplica esse filtro antes de qualquer rotação, usando o `w` original dos nós.

## Divisão train/val/test

Depende só do tempo de nascimento, relativo a `t_max`:

| Condição | Split |
|---|---|
| `w <= 0.6 * t_max` | train |
| `w <= 0.8 * t_max` | val |
| caso contrário | test |

Nós criados ou movidos na interface têm o split recalculado. Mudar `t_max` recalcula todos.

## Projeções R⁴ -> R³

| Modo | Resultado |
|---|---|
| `drop_w` | Espaço puro, `(x, y, z)` |
| `drop_x` | `(y, z, w)` |
| `drop_y` | `(x, z, w)` |
| `drop_z` | `(x, y, w)` |
| `perspective` | `(x, y, z)` multiplicado por `camera_w / (camera_w - w)` |

Na perspectiva, a profundidade `camera_w - w` é limitada por baixo em 0,15. Pontos com `w` perto ou acima de `camera_w` (possível depois de uma rotação) ficam todos com a mesma ampliação máxima.

## Rotações

`rotation_matrix_4d(plano, ângulo)` devolve uma rotação ortogonal no plano escolhido (`xy`, `xz`, `xw`, `yz`, `yw`, `zw`). As rotações da interface são aplicadas na ordem xw, yw, zw e não comutam entre si.

Estas rotações são euclidianas: o tempo é tratado como uma dimensão espacial. Depois de uma rotação, o novo `w` já não significa "instante de nascimento".

Para a transformação de Lorentz, o parâmetro `boosts` das projeções aplica `lorentz_boost_matrix(eixo, beta)`: uma rotação hiperbólica na métrica `(+, +, +, -)`, que exige `w = c·t` na mesma unidade da distância. Em grafos genéricos, `c = 1`. A vista Minkowski da interface centra a cena no evento do nó selecionado e desenha uma fatia `z = z0` do cone de luz; o boost deforma as linhas de mundo, mas o cone continua sendo o cone (o intervalo é invariante). Não há relatividade geral nem tempo próprio.

## Geradores

| Gerador | Nós | Arestas | Tempo |
|---|---|---|---|
| `generate_knowledge_demo` | 9 | 11 | `w` de 0 a 5, `t_max = 8`; duas arestas com fim |
| `generate_random` | `n_nodes` | até `n_edges` (limitado ao grafo completo) | `w` inteiro em `[0, n_steps]`, mesma faixa para as arestas |
| `generate_tesseract` | 16 | 32 | `w` em `{0, 2 * scale}`, `t_max = 2 * scale` |
| `generate_simplex4` | 5 | 10 | quatro nós em `w = 0` e um em `w = 4` |

No aleatório, o rótulo do nó é 1 quando o grau final é pelo menos 3. Nos outros dois, o rótulo é sempre 0. Com a mesma seed, o grafo aleatório é idêntico.
