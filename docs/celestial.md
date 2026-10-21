# Sistema solar como grafo 4D

O modo "Sistema solar" usa o mesmo núcleo dos grafos 4D (`Graph4D`, projeções, snapshots, arestas com validade, recuperação) e acrescenta o que um corpo em movimento exige. Está no subpacote `temporal_graph.celestial`; o núcleo não o importa.

## Como cada limitação do modelo de pontos foi tratada

| Limitação | Tratamento | Onde |
|---|---|---|
| Não havia "posição atual"; o snapshot mostrava todas as amostras | Um corpo tem uma órbita, e `position_at(nó, t)` dá a posição em `t`. `snapshot(t)` devolve `pos` em `t`, e a cena desenha só a posição atual. Trilhas são opcionais | `model.py`, `scene.py` |
| O grafo crescia com corpos × passos | Um nó por corpo (52 no catálogo). A trajetória é uma função, não nós; as trilhas são amostradas na hora de desenhar | `model.py` |
| Sliders de -3 a 3 e raios fixos | Escala de exibição hierárquica (log, raiz ou real), com tamanhos próprios e afastamento mínimo para as luas | `scale.py` |
| O código não calculava órbitas | Elipses keplerianas com taxas seculares, órbitas tabeladas (CSV) e integração N-corpos dos planetas | `orbits.py`, `nbody.py` |
| Rotações euclidianas, tempo absoluto | Boosts de Lorentz, intervalo de Minkowski, cone de luz, tempo-luz e eventos retardados | `relativity.py`, `scene.py` |
| Uma aresta por par | O mesmo par pode ter várias arestas: uma relação reaparece se as janelas não se sobrepõem. Oposições, conjunções, eclipses e ressonâncias usam isso | `model.py`, `events.py` |
| Amostras do mesmo corpo empatavam na busca | Um nó por corpo, cada um com texto próprio | `catalog.py` |

## Escala de exibição

Netuno está a 30 UA e a Lua a 0,0026 UA da Terra, então uma escala linear não mostra as duas coisas. A posição na cena é hierárquica: cada corpo fica em torno da posição (já comprimida) do pai, na direção da posição física relativa ao pai.

| Nível | Modo `log` | Modo `sqrt` | Modo `real` |
|---|---|---|---|
| Heliocêntrico | `3 · log10(1 + d / 0,05 UA)` | `1,5 · sqrt(d)` | `0,25 · d` |
| Luas | raios do pai e da lua mais `0,08 · log10(1 + d / 1000 km)` | igual ao `log` | `0,25 · d` |
| Tamanho | `0,03 + 0,07 · log10(R / 100 km)`, mínimo 0,025 | igual | proporcional ao raio real, mínimo 0,025 |

Direções e ordem de grandeza se preservam; razões de distância, não. O modo `real` mostra o quanto o sistema é vazio: as luas colapsam sobre os planetas e os corpos viram pontos do mesmo tamanho.

## Dinâmica

- **Kepler (padrão).** Elementos orbitais com taxas por século. A equação de Kepler é resolvida por Newton (resíduo abaixo de 1e-11 até e = 0,999).
- **Órbitas tabeladas.** `TableOrbit` interpola posições amostradas (Hermite cúbico, se houver velocidades). `TableOrbit.from_csv` lê um CSV `jd,x,y,z` (ou `t,x,y,z`), em UA ou km, relativo ao pai e em eixos da eclíptica J2000. É o caminho para usar uma efeméride do JPL: exporte os vetores, ajuste as colunas e troque o `orbit` do corpo.
- **N-corpos.** `generate_solar_system(dynamics="nbody")` integra o Sol e os nove corpos principais (integrador simplético de 4ª ordem, passo de 1 dia) a partir de vetores de estado em J2000. Corpos menores e luas seguem Kepler.

## Relatividade restrita

Eventos são `(x, y, z, w)` com `w = c·t` na mesma unidade da distância; para o sistema solar, `c = 173,1446 UA/dia`.

- `lorentz_boost_matrix(eixo, beta)` e o parâmetro `boosts` das projeções substituem as rotações euclidianas. O intervalo `|dx|² − dw²` é preservado (testado).
- `classify_interval` diz se dois eventos são tipo-tempo, tipo-luz ou tipo-espaço.
- A **vista Minkowski** centra a cena no evento "agora" do observador, desenha a fatia `z = z0` dos cones de luz passado e futuro, as linhas de mundo e, em branco, os **eventos retardados**: onde cada corpo estava quando saiu a luz que chega ao observador agora. Esses pontos ficam sobre o cone passado (testado como intervalo tipo-luz).
- A janela de tempo é curta (padrão 0,25 dia, o que cobre até Netuno). Só entram na vista os corpos dentro de 1,3 vezes o raio do cone.

## Dados e confiança

| Grupo | Fonte | Confiança |
|---|---|---|
| Planetas | Elementos médios e taxas do JPL (aproximação para 1800 a 2050) | Validados contra a efeméride do astropy |
| Lua | Elementos médios de Meeus, com regressão dos nós e precessão do apogeu | Erro de até ~3,7° em longitude |
| Demais luas (28) | Semieixo, excentricidade, inclinação e período reais | Orientação e fase **ilustrativas** |
| Planetas-anões e asteroides | Forma da órbita real | Fase **ilustrativa** (Plutão usa a tabela do JPL) |
| Cometas | Forma e data do periélio | Fase real, sem perturbações |

Cada corpo tem a marca `real_phase`, e a interface avisa quando a fase é ilustrativa. As luas além da Lua não estão na posição real em nenhuma data.

## Validação

Planetas contra a efeméride embutida do astropy, erro máximo em 25 datas entre 2000 e 2100 (UA):

| Corpo | Kepler | N-corpos |
|---|---|---|
| Mercúrio | 0,0000 | 0,064 |
| Vênus | 0,0001 | 0,026 |
| Terra | 0,0001 | 0,003 |
| Marte | 0,0005 | 0,036 |
| Júpiter | 0,011 | 0,150 |
| Saturno | 0,053 | 0,214 |
| Urano | 0,018 | 0,117 |
| Netuno | 0,015 | 0,070 |

O N-corpos erra **mais** que o Kepler de elementos médios, que foi ajustado à efeméride. Ele serve para mostrar a dinâmica (atração mútua), não para ganhar precisão.

Outras verificações:

- Lua contra o astropy em 100 anos: erro máximo de 3,7° em longitude, 0,34° em latitude e ~6.750 km em distância.
- Terceira lei de Kepler para todo o catálogo, com a massa do pai: mediana de 0,01% de desvio, pior caso (Nix) 0,65%.
- Oposições de Marte de 2020 a 2029: a menos de 1 dia das datas reais.
- Conjunção de Júpiter e Saturno vista da Terra: a 1,2 dia da data real (2020-12-21).
- Eclipses de 2025 a 2027: os 10 eclipses solares e lunares umbrais conhecidos foram encontrados, a 0,5 dia em média e no máximo 0,84 dia, sem candidatos a mais. Os penumbrais ficam de fora por projeto.
- Eclipses solares candidatos de 2000 a 2100: 225. Os catálogos citam 224 para o século (valor que cito de memória, sem conferir).
- Energia do N-corpos conservada com deriva relativa de 3,6e-8 em 100 anos.

## O que continua fora

- **Catálogo.** São 52 corpos: Sol, 8 planetas, 5 planetas-anões, 5 asteroides, Sedna, 3 cometas e 29 luas. Não são todas as cerca de 290 luas nem os milhões de asteroides. Para ampliar, use `TableOrbit` ou novos elementos em `catalog.py`.
- **Perturbações.** Kepler é de dois corpos. O próximo periélio do Halley sai em 2061-06-04 no modelo e em 2061-07-28 na realidade, porque faltam as perturbações planetárias.
- **Referencial.** O Sol fica na origem (ele se move até cerca de 0,01 UA em torno do baricentro). A "Terra" é o baricentro Terra-Lua da tabela do JPL, a até ~4.700 km do centro da Terra. Nix e Hidra orbitam o baricentro Plutão-Caronte, mas aqui orbitam Plutão.
- **Relatividade.** É só a restrita: referenciais inerciais. As órbitas são newtonianas, sem relatividade geral (falta, por exemplo, o avanço do periélio de Mercúrio de 43"/século) e sem tempo próprio. O cone de luz é uma fatia `z = z0`.
- **Eclipses.** São candidatos do modelo: sizígia com a Lua perto da eclíptica. Não distinguem total de anular nem dão a região de visibilidade, e casos limítrofes podem entrar ou sair.
- **Ressonâncias.** O teste é a razão de períodos próxima de p:q (p e q até 4, tolerância de 0,9%), sem verificar o ângulo de ressonância. Júpiter e Saturno (5:2) ficam de fora por causa do limite de ordem.
