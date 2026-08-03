# Cómo quedaron funcionando los encoders

Estado al 2 de agosto de 2026. Este documento explica **qué hace cada encoder,
por qué está donde está, y qué se probó antes de dejarlo así**. Si vas a tocar
la arquitectura de recuperación, empezá acá.

---

## 1. El resumen en una frase

**Un encoder busca y dos corrigen.** MiniLM recupera 200 candidatos, y
`gte-multilingual-base` y `multilingual-e5-base` los re-puntúan sumando su
similitud con peso 0,25 cada uno. Ninguno de los dos aporta candidatos
propios.

```
consulta
   │
   ├─ MiniLM la vectoriza  ──► FAISS ──► 200 candidatos          [RECALL]
   │                                          │
   ├─ gte vectoriza la consulta ──────────────┤ +0,25 × similitud [PRECISIÓN]
   ├─ e5 vectoriza la consulta ───────────────┤ +0,25 × similitud [PRECISIÓN]
   │                                          ▼
   │                                   reordenar, quedarse con 60
   │                                          │
   ├─ agregar a documento (suma de scores) ──► 3 documentos
   └─ ordenar fragmentos hacia esos 3 docs ──► 10 fragmentos ≤250 palabras
```

**El coste por consulta son tres vectorizaciones de una frase corta.** Los
vectores de los *pasajes* no se recalculan nunca: se leen del índice del
encoder correspondiente con `reconstruct(fila)`. Eso solo es posible porque
los tres índices describen **los mismos chunks en el mismo orden**.

---

## 2. Los tres encoders

| | papel | parámetros | ventana | dim | licencia |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | **primario** (recall) | 118 M | 128 | 384 | Apache 2.0 |
| `Alibaba-NLP/gte-multilingual-base` | re-puntuador | 305 M | 512 | 768 | Apache 2.0 |
| `intfloat/multilingual-e5-base` | re-puntuador | 278 M | 512 | 768 | MIT |

Los tres son **encoder-only tipo BERT**. Ninguno es un decoder, que es lo que
prohíbe la sec. 8.3. Esa restricción dejó fuera a lo mejor de 2026
(`microsoft/harrier-oss-v1`, familia Qwen3-Embedding): son decoder-only con
last-token pooling y, aunque se usaran solo para vectorizar, un evaluador
estricto puede leerlo como incumplimiento. No vale arriesgar la exclusión.

---

## 3. Por qué cada uno está donde está

### 3.1 MiniLM es el primario, y no es por casualidad

Es el **más chico y el de ventana más corta** — trunca en 128 tokens y el
96% de los chunks son más largos. Suena a que debería perder, y sin embargo
es el mejor primario por bastante. Se probó lo contrario dos veces:

| primario | F1@3 (41) | F1@3 (10 indep.) |
|---|---|---|
| **MiniLM** | **0,386** | **0,333** |
| e5-base | 0,250 | 0,267 |
| gte | 0,385 | **0,200** |

El caso de gte es el más instructivo y hay que recordarlo: **luce igual de
bien que MiniLM en las 41 y se derrumba a 0,200 en las independientes.** No
es ruido: las 41 se anotaron sobre candidatos que propuso MiniLM, así que
premian a quien recupera lo mismo que MiniLM. Es la misma firma por la que se
descartó `doc_rrf`. Para reabrirlo habría que anotar un pool propio de gte.

**Consecuencia práctica:** la truncación a 128 tokens es real y está
documentada, pero **no hace peor recuperador a MiniLM**. Si la ventana fuera
el cuello de botella, e5 o gte habrían ganado como primarios. También cierra
la idea de re-fragmentar el corpus a 128 tokens.

### 3.2 Por qué cascada y no fusión

Fusionar las listas de dos encoders con RRF **empeora**: 0,268 contra 0,306.
La razón está medida: los dos encoders comparten solo el **11,3%** de los
documentos del top-3. RRF premia el acuerdo, y con ese desacuerdo no fusiona
—intercala la lista buena con la mala.

La cascada funciona porque **separa dos trabajos distintos**:

- **el primario responde por el recall**: tiene que traer el documento
  correcto dentro de los 200. Ahí manda la cobertura.
- **los secundarios responden por la precisión**: solo reordenan lo que ya
  llegó. Que se equivoquen en el recall no importa, porque no recuperan nada.

Por eso e5 sirve como re-puntuador aunque sea malísimo como primario (0,250).
Son dos habilidades separadas.

### 3.3 Por qué dos re-puntuadores y no uno

Se midieron **cinco estructuras** con los tres índices ya construidos, a coste
cero de codificación (`scripts/barrido_estructuras.py`):

| estructura | F1(41) | NDCG(41) | F1(10) | NDCG(10) |
|---|---|---|---|---|
| MiniLM→e5 | 0,344 | 0,338 | 0,333 | 0,329 |
| MiniLM→gte | 0,378 | 0,393 | 0,333 | 0,362 |
| gte solo | 0,311 | 0,309 | 0,333 | 0,287 |
| gte→MiniLM | 0,385 | 0,393 | **0,200** | **0,202** |
| gte→e5 | 0,339 | 0,341 | 0,400 | 0,322 |
| **MiniLM→gte+e5** | **0,386** | **0,406** | 0,333 | 0,360 |

La triple es la **única ADOPTABLE en las cuatro mediciones** por el criterio
del proyecto (IC al 90% que excluya una pérdida de 0,02, ver
`lecciones_metodologia.md`).

**Sé honesto al citarla: la ganancia sobre MiniLM→gte es chica** — +0,014 de
NDCG en las 41 y −0,002 en las 10. Se adoptó porque cumple el criterio y
cuesta una vectorización más, no porque sea un salto.

### 3.4 Por qué peso 0,25

Pesos mayores (0,5 y 1,0) promedian mejor sobre las 41 pero **empiezan a
perder consultas en las 10 independientes**, que es la señal clásica de
sobreajuste al pooling. Con 0,25 la cascada no empeora ninguna consulta de
ninguna muestra. Se prefirió la variante que nunca hace daño sobre la que
promedia mejor. **No re-buscar el peso óptimo**: sería sobreajustar a 41
consultas cuyo efecto mínimo detectable es 0,059.

---

## 4. El invariante que sostiene todo

**El chunking se hace UNA sola vez** (con el tokenizer del primer encoder) y
esos mismos records se indexan con los tres. Nunca re-fragmentar dentro del
bucle de encoders.

Si se rompe: distintos tokenizers producen fragmentaciones distintas, los
`chunk_id` colisionan apuntando a textos diferentes, y `reconstruct(fila)`
devuelve **el vector de otro chunk**. No hay excepción — solo resultados
peores, que uno atribuiría al encoder.

Tres defensas, en orden de cuándo actúan:

1. `scripts/indexar_desde_metadata.py` construye índices nuevos **desde la
   metadata ya entregada**, así la alineación queda garantizada por
   construcción.
2. `scripts/verificar_alineacion.py` comprueba número de vectores, `chunk_id`
   idénticos **y en el mismo orden**, normalización y ausencia de NaN.
   Obligatorio antes de usar cualquier índice nuevo.
3. `generador.py` valida los `chunk_id` al cargar y aborta con exit 2.

---

## 5. Trampas de gte que costaron horas

**Se carga ROTO, y de dos formas.** Declara sus buffers de RoPE con
`persistent=False`: no viajan en el checkpoint, y `transformers` los
materializa desde **memoria sin inicializar**.

```
position_ids[0] = 2635600166912   (debería ser 0)   → revienta con IndexError
inv_freq = [6.4e+20, 1.6e-42, 0]                    → NO revienta
```

El segundo es el peligroso: el modelo codifica **sin información posicional**
y devuelve vectores normalizados, con la forma correcta y semánticamente
basura. Con el modelo así, un pasaje irrelevante puntuaba por encima de uno
relevante — y se estuvo a punto de concluir "gte es malo".

`_reparar_buffers_no_persistentes()` lo arregla, **está aplanada dentro de
`Entrega/generador.py` y no se puede quitar**. Solo reescribe si el contenido
difiere, así que en una versión de `transformers` sin el problema es un no-op.

**Además:** en CPU hay que apagar `unpad_inputs` y
`use_memory_efficient_attention` (asumen GPU), y recortar `max_seq_length` de
8192 a 512.

**Y el coste.** Medido sobre chunks reales, no extrapolado:

| | ms/chunk | índice completo (128.526) |
|---|---|---|
| MiniLM | 28 | 1,0 h |
| gte en CPU | 2.716 | **97 h** |
| gte en GPU (GTX 1650) | 204 | **7,3 h** |

**97× más lento que MiniLM en CPU, no 2,6× como decía la extrapolación por
número de parámetros.** Se multiplican tres factores: parámetros, tokens
(MiniLM trunca en 128) y la atención densa obligatoria en CPU. El índice se
construyó en GPU, en `.venv-cuda`, un entorno **aparte** de `.venv` para no
tocar el que produce la entrega.

**Regla que salió de acá: medir ms/chunk sobre 64 chunks reales antes de
lanzar cualquier corrida larga.** Cuesta tres minutos y habría evitado lanzar
una corrida estimada en 23 minutos que eran 4,7 horas.

---

## 6. Qué pasa si gte no carga en el entorno del evaluador

gte necesita `trust_remote_code=True` y descargar ~1,2 GB de HuggingFace. Si
el entorno lo bloquea, `generador.py` **aborta con un mensaje que dice qué
correr**, no con un traceback:

```
python generador.py --consultas <archivo> --rerank-encoder multilingual-e5-base
```

El índice de e5 se conserva en la entrega justamente para eso. **No hay
fallback automático a propósito**: cambiar de re-puntuador en silencio
produciría resultados distintos a los entregados, que es exactamente lo que
la sec. 1.4 penaliza. Mejor fallar diciendo la verdad.

Ambos caminos están probados en `scripts/pruebas_robustez.py`, junto con
`--rerank-encoder none`, que apaga la cascada del todo.

---

## 7. Lo que NO hay que volver a intentar

Todo esto está medido y en la sección "Medido y descartado" de `las notas del proyecto`:

- **Fusión RRF simétrica** de dos encoders — 0,268 vs 0,306.
- **e5 o gte como primario** — 0,250 y 0,200 en las independientes.
- **Híbrido BM25 + denso** — pierde 15-4. La unión de pools también, 15-2.
- **Un tercer encoder nuevo** — la ganancia de cada uno es marginal y cada
  índice son 376 MB y horas de GPU.
- **Re-buscar el peso o la profundidad** — `rerank-depth` 400 y 600 dieron
  **51 empates de 51**.
- **Re-fragmentar a 128 tokens** para la ventana de MiniLM — si la ventana
  fuera el problema, e5 y gte habrían ganado como primarios.

**Lo que sí queda abierto**, y es lo único: la **construcción del pool**. El
fallo documentado es cross-lingual (consulta en español, documento en inglés:
NBQR/CBRN, "reabastecimiento en órbita"/on-orbit servicing). gte tiene la
penalización por idioma casi en cero (−0,027/+0,036 contra +0,052/+0,091 de
MiniLM), así que **es el candidato natural a primario el día que exista un
ground truth que no esté sesgado hacia el pool de MiniLM**.
