# Grafo espaço-tempo em R⁴

Simulação para entender como um grafo se comporta quando cada nó vive em quatro coordenadas, `(x, y, z, w)`, em que `x, y, z` são espaço e `w` é tempo. A mesma aplicação tem dois modos, escolhidos na barra superior:

- **Grafos 4D**: grafos genéricos com nós em pontos fixos, snapshots no tempo, arestas com início e fim, projeções, edição e uma demonstração de recuperação temporal no estilo GraphRAG.
- **Sistema solar**: um caso de estudo em que cada nó é um corpo com trajetória (planetas, luas, planetas-anões, asteroides e cometas), com posição atual, escala de exibição, linhas de mundo e uma vista de relatividade restrita.

O objetivo continua sendo estudar o básico de grafos 4D como ponte para o GraphRAG 4D. Não é um pipeline de aprendizado, nem um sistema de GraphRAG completo, nem uma efeméride. É uma ferramenta de visualização com um modelo pequeno por trás.

## Rodando

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

temporal-graph          # ou: python -m temporal_graph
```

A interface sobe em `http://localhost:8080`. Para escolher a porta sem abrir o navegador:

```bash
temporal-graph --port 8765 --server
```

Com Docker:

```bash
docker compose up --build
```

Sobe em `http://localhost:8080`. A imagem usa Xvfb e Mesa para renderizar sem tela.

O PyVista renderiza no servidor, então a máquina precisa de OpenGL (ou OSMesa, em ambiente sem tela). O estado vive no servidor: duas abas abertas mexem nos mesmos grafos.

Testes e lint:

```bash
pytest
ruff check src tests
```

Os extras `[dev]` incluem o `astropy`, usado só nos testes que comparam os planetas com uma efeméride independente; sem ele, esses testes são pulados. Validado com Python 3.12, pyvista 0.49, trame 4.0, trame-vuetify 3.3, trame-pyvista 0.2, vtk 9.7 e astropy 8.0.

## Modo Grafos 4D

- **Snapshot em t**: só nós com `w <= t` e arestas ativas em `t`. Arestas valem em `[t, t_end)`; as que já terminaram aparecem em vermelho escuro.
- **Projeção**: descartar `w` mostra o espaço puro; descartar `x`, `y` ou `z` põe o tempo como eixo; há também uma perspectiva com a câmera em `w`.
- **Rotação ou boost**: rotações euclidianas nos planos xw, yw e zw, ou boosts de Lorentz com `beta` por eixo.
- **Vista Minkowski**: centra a cena no nó selecionado e desenha o cone de luz dele.
- **Recuperação**: consulta, `k` sementes e `hops` saltos. Sementes em verde, vizinhos em ciano, o resto esmaece, e a busca é refeita ao mexer em `t`.
- **Edição e presets**: selecionar, mover, criar, remover e conectar nós; base de conhecimento, aleatório, tesseract e 4-simplex.

## Modo Sistema solar

52 corpos: Sol, 8 planetas, 5 planetas-anões, 5 asteroides, Sedna, 3 cometas e 29 luas. Cada um é um nó, e a trajetória é uma função do tempo.

- **Data**: o slider de tempo é a data (dias desde J2000). A cena mostra a posição atual de cada corpo.
- **Escala**: logarítmica, raiz quadrada ou real. As luas ficam ao redor dos planetas, com afastamento ajustável.
- **Trilhas**: linhas de mundo dos corpos; com a projeção que descarta `z`, o tempo vira um eixo e os planetas viram hélices.
- **Dinâmica**: Kepler (analítico) ou N-corpos integrado para os planetas.
- **Vista Minkowski**: cone de luz do observador (padrão: Terra) e, em branco, os eventos retardados, ou seja, onde cada corpo estava quando saiu a luz que chega agora.
- **Eventos recorrentes**: oposições de Marte, conjunções de Júpiter e Saturno, eclipses candidatos e ressonâncias viram arestas com janelas próprias.
- **Busca**: `satélites de Júpiter` devolve as seis luas de Júpiter do catálogo e o contexto com as ressonâncias.

As posições dos planetas e da Lua foram validadas contra a efeméride do astropy; as fases das demais luas e dos corpos menores são **ilustrativas** (a forma da órbita é real). Os números e os limites estão em [`docs/celestial.md`](docs/celestial.md).

## Estrutura

```
temporal_graph/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── docs/
│   ├── conceito.md          modelo, convenções e limitações do núcleo
│   ├── graphrag.md          como funciona a recuperação e o que falta
│   └── celestial.md         sistema solar: escala, dinâmica, relatividade, validação
├── src/temporal_graph/
│   ├── model.py             Node4D, Edge4D, Body, Graph4D
│   ├── projection.py        rotações 4D, boosts e projeções R⁴ -> R³
│   ├── relativity.py        boost de Lorentz, intervalo, tempo-luz
│   ├── embedding.py         embedding lexical e PCA 3D
│   ├── retrieval.py         busca temporal: sementes, expansão, contexto
│   ├── frame.py             quadro de cena (posições, trilhas, cone)
│   ├── generator.py         demo de conhecimento, aleatório, tesseract, 4-simplex
│   ├── renderer.py          cena PyVista
│   ├── ui.py                layout Trame/Vuetify
│   ├── app.py               estado, callbacks, main()
│   └── celestial/           usa o núcleo; o núcleo não o importa
│       ├── orbits.py        Kepler e órbitas tabeladas
│       ├── scale.py         escala de exibição hierárquica
│       ├── catalog.py       catálogo do sistema solar e construção do grafo
│       ├── events.py        alinhamentos, eclipses, ressonâncias
│       ├── nbody.py         integração N-corpos
│       └── scene.py         quadro do sistema solar e vista Minkowski
└── tests/
```

O núcleo depende só de NumPy e NetworkX. PyVista e Trame ficam em `renderer.py`, `ui.py` e `app.py`, então o modelo funciona sem a interface:

```python
from temporal_graph import build_context, generate_knowledge_demo, retrieve
from temporal_graph.celestial import generate_solar_system

g = generate_knowledge_demo()
result = retrieve(g, "quem trabalha no Atlas", t=6, k=3, hops=1)
print(build_context(g, result))        # texto pronto para um prompt

sistema = generate_solar_system()
t = sistema.t_default                  # 2026-10-19, em dias desde J2000
sistema.position_at("Marte", t)        # posição em UA, eclíptica J2000
sistema.snapshot(t).nodes["Terra"]["pos"]
```

## Limitações

- O embedding da busca é lexical: só enxerga palavras em comum, sem semântica. Ver [`docs/graphrag.md`](docs/graphrag.md).
- As rotações são euclidianas. A transformação de Lorentz está em `boosts`, mas é só relatividade restrita: sem curvatura e sem tempo próprio. Ver [`docs/conceito.md`](docs/conceito.md).
- O catálogo do sistema solar não é completo e usa Kepler de dois corpos. O N-corpos erra mais que o Kepler de elementos médios. Ver [`docs/celestial.md`](docs/celestial.md).

## Licença

MIT. Veja [`LICENSE`](LICENSE).
