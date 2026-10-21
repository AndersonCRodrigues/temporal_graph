# Recuperação temporal (GraphRAG didático)

A recuperação responde a uma pergunta do tipo "o que o grafo sabia sobre isto em t?". Está em `retrieval.py` e é separada da interface: `retrieve` e `build_context` só dependem de NumPy e NetworkX.

## Passo a passo

1. **Snapshot em t.** Só entram nós já nascidos (`w <= t`) e arestas ativas (`t_aresta <= t < t_end`).
2. **Sementes.** Cada nó com texto tem um embedding. Os `k` nós do snapshot com maior similaridade de cosseno com a consulta viram sementes. Similaridade 0 não vira semente.
3. **Expansão.** A partir das sementes, anda-se até `hops` saltos, só por arestas ativas em `t`. Cada nó guarda a distância à semente mais próxima.
4. **Contexto.** `build_context` lista os nós (por distância, depois por similaridade) e as relações entre eles, com o intervalo de validade de cada uma.

## Exemplo da demo

Consulta `quem trabalha no Atlas`, `k = 3`, `hops = 1`, em `generate_knowledge_demo()`:

| t | Semente | Vizinhos |
|---|---|---|
| 3 | Atlas | Ana, Bruno, Orion |
| 4 | Atlas | Ana, Bruno, Delta, Orion |
| 6 | Atlas | Bruno, Delta, Orion |
| 8 | Atlas | Bruno, Orion |

A Ana trabalha no Atlas em `[1, 5)` e a Delta é atendida em `[4, 7)`. A consulta é a mesma; só muda o instante.

## No sistema solar

Em modo "Sistema solar", a consulta `satélites de Júpiter` com `k = 6` e `hops = 1` devolve as seis luas de Júpiter do catálogo como sementes e o próprio Júpiter como vizinho. O contexto traz as relações de órbita e as ressonâncias entre as luas (Io-Europa, Europa-Ganimedes). Cada corpo é um nó com texto próprio, então as amostras de tempo não competem entre si na busca.

## O que a cena mostra

| Elemento | Significado |
|---|---|
| Nó verde, maior | Semente (com a similaridade no rótulo) |
| Nó ciano | Vizinho dentro do alcance de `hops` |
| Nó esmaecido | Visível em `t`, mas fora do contexto |
| Aresta branca grossa | Relação incluída no contexto |
| Aresta vermelha escura | Já começou e já terminou em `t` |

## Limites desta implementação

- **Embedding lexical.** `embed_text` conta palavras (sem acento e sem stopwords) em 1024 baldes por hash. Não há semântica: "carro" e "automóvel" não se aproximam. Palavras diferentes podem colidir no mesmo balde e gerar uma similaridade pequena e espúria. Serve para rodar offline e de forma determinística, não para medir qualidade de recuperação.
- **Posição na cena.** Na demo, `(x, y, z)` vem de um PCA dos embeddings. Distância no gráfico não é similaridade, e a busca usa o vetor completo, não essas três coordenadas.
- **Só as arestas expiram.** Nós nascem e ficam. Cada par de nós tem no máximo uma aresta, então uma relação não pode reaparecer depois de terminar.
- **Um só tempo.** Há o tempo em que o fato vale, mas não o tempo em que foi registrado. Sem isso não dá para perguntar "o que o sistema sabia em t, antes de uma correção posterior".
- **Sem geração e sem avaliação.** O resultado é um contexto em texto, sem modelo de linguagem, e não há métrica de acerto.

## O que viria depois

Trocar `embed_text` por um modelo de embeddings real (a interface é só texto de entrada e vetor de saída), usar um corpus de verdade, separar tempo de validade e tempo de registro, e medir recall por instante contra consultas com resposta conhecida.
