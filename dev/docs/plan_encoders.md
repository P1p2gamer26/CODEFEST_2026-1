# Plan de encoders: qué hacer funcione o no la cascada

Escrito el 1 de agosto de 2026, mientras `multilingual-e5-base` se re-codifica
sobre el texto con los guiones reparados. Cubre los dos desenlaces posibles y
el hallazgo que, medido hoy, pesa más que cualquier cambio de encoder.

---

## 0. El hallazgo que cambia el orden de prioridades

**El encoder primario solo ve la primera mitad de casi todos los fragmentos.**

`paraphrase-multilingual-MiniLM-L12-v2` tiene `max_seq_length = 128` tokens.
La distribución real de los 128.680 chunks del corpus:

| | tokens |
|---|---|
| mediana | **256** |
| p90 | 277 |
| p99 | 466 |
| máximo | 2.642 |

| umbral | chunks por encima |
|---|---|
| 128 tokens (ventana de MiniLM) | **123.471 — el 96,0 %** |
| 256 tokens | 63.153 (49,1 %) |
| 512 tokens (ventana de e5) | 955 (0,7 %) |

O sea: en el 96 % de los fragmentos, **todo lo que viene después del token 128
se descarta silenciosamente al codificar**. El vector que se indexa representa
aproximadamente la primera mitad del fragmento. El texto completo sí se
entrega en `resultados.jsonl` y sí lo lee el evaluador — la pérdida es solo en
la recuperación, que es justo la mitad de la nota.

De dónde viene: el chunking se presupuestó contra el límite de **250 palabras
de la sec. 9.3.2** (≈256 tokens), no contra la ventana del encoder. Son dos
límites distintos y nadie los cruzó.

`multilingual-e5-base` **no tiene este problema**: su ventana es de 512 tokens
y cubre el 99,3 % de los chunks enteros. Esto reordena la lectura de las
mediciones viejas: cuando se comparó "e5 solo" contra "MiniLM solo" y e5 salió
peor, se estaban comparando dos cosas que ni siquiera leían el mismo texto.

**Consecuencia práctica:** antes de buscar un encoder nuevo, hay dos
experimentos baratos que atacan esto directamente (sección 3).

---

## 1. Plan A — la cascada funciona

Se confirma que funciona si, con el ground truth en 50 consultas, la cascada
**no pierde consultas** contra MiniLM solo. La regla ya estaba fijada de
antemano en `las notas del proyecto`: al llegar a 50 no se re-busca el peso óptimo, solo se
verifica que no haga daño.

Qué hacer entonces, por orden de impacto:

1. **Cerrar el ground truth.** Faltan **q001, q007, q008, q011, q012, q015,
   q028, q038, q048**. Sin las 50, toda medición sigue teniendo ±0.1 de ruido
   y ninguna decisión posterior es defendible.
2. **Probar `k_pool` más profundo con la cascada.** El re-puntuador solo puede
   reordenar lo que el primario le pasó: hoy son 200 candidatos. Subirlo a
   400–600 le da más material sin costo de recall, y el costo por consulta es
   solo aritmética (los vectores del secundario se leen con `reconstruct`, no
   se recodifica nada).
3. **Re-medir el peso de la cascada** solo si el paso 1 muestra pérdidas. Hoy
   está en 0.25 porque es la variante que no empeora ninguna consulta; 0.5 y
   1.0 promedian mejor pero pierden casos.
4. **NO tocar** la agregación (`sum`, `k_pool=60`), ni reabrir BM25, ni el
   grafo en la fusión: los tres están medidos y descartados con datos.

---

## 2. Plan B — la cascada no funciona

Se descarta si la cascada pierde consultas contra MiniLM solo sobre las 50, o
si el índice del e5 sale corrupto o no llega a tiempo.

**Salida inmediata, sin riesgo:** entregar con **MiniLM limpio solo**. Ya está
construido y verificado (128.526 vectores, metadata alineada, 0 cortes
pendientes) y da F1@3 0.310. La entrega no depende del e5: se apaga con
`--rerank-encoder none` y `generador.py` sigue reproduciendo `resultados.jsonl`
sin flags. **Esta es la red de seguridad y ya está tendida.**

El resto de esta sección es qué probar si además sobra tiempo de cómputo.

---

## 3. Los dos experimentos baratos, antes de cambiar de encoder

Los dos atacan el hallazgo de la sección 0 y **no requieren descargar nada**.

### 3.1 Hacer primario al e5 (o a cualquier ventana de 512)

Costo: **cero cómputo nuevo** una vez que termine la corrida en curso. Es
cambiar qué índice manda y cuál re-puntúa. La hipótesis es directa: el e5 ve
el fragmento entero y MiniLM no; que el e5 rindiera peor en las mediciones
viejas pudo deberse a los prefijos, al texto roto, o al sesgo de pooling del
ground truth — pero también a que se lo comparó contra un rival que jugaba con
la mitad del texto.

Cómo medirlo: `--encoder-name multilingual-e5-base --rerank-encoder
paraphrase-multilingual-MiniLM-L12-v2`, y comparar con
`eval_mini.py --comparar-con` **contando victorias**, no promedios, y con
`--sin-pooling` para las 10 independientes.

### 3.2 Fragmentos de ≤128 tokens para MiniLM

Costo: re-chunkear y re-codificar (~1 h de MiniLM), y **rompe el invariante
del chunking único** — habría que rehacer también el índice del secundario, o
renunciar a la cascada. Por eso va segundo.

La sec. 9.3.2 pone un techo de 250 palabras, no un piso: fragmentos más cortos
son válidos. El riesgo es que un fragmento corto tenga menos contexto para el
juicio de relevancia del evaluador (NDCG@10), así que esto mejora una métrica
a costa posible de la otra. **Medir antes de adoptar.**

---

## 4. Si hay que traer un encoder nuevo de Hugging Face

### Restricciones no negociables de este proyecto

- **Sin modelos generativos** (sec. 8.3). Los encoders bidireccionales
  (BERT-like) están bien. Los *embedders* construidos sobre un LLM decoder —
  familia Qwen3-Embedding, e5-mistral, y en general todo lo que salga de un
  modelo de 7B — **son terreno resbaladizo**: aunque se usen solo para
  vectorizar, el backbone es un modelo generativo y un evaluador estricto
  puede leerlo como incumplimiento. No vale el riesgo de exclusión por unas
  décimas.
- **CPU, sin GPU.** Es la restricción que decide de verdad. Referencias
  medidas hoy en esta máquina, sobre los mismos 128.526 chunks:

  | modelo | parámetros | ventana | tiempo real de codificación |
  |---|---|---|---|
  | MiniLM-L12 | 118 M | 128 | **62 min** |
  | e5-base | 278 M | 512 | **~6 h** |

  El costo escala con parámetros **y** con tokens efectivos. Un modelo de
  ~560 M con ventana de 512 se va a **12 h o más**. Cualquier candidato hay
  que evaluarlo con ese reloj en la mano, no por su puesto en MTEB.

  > **CORRECCIÓN (2 ago 2026): la regla de tres por parámetros es
  > inservible, y por mucho.** Medido sobre chunks reales del corpus, con
  > `scratchpad/medir_ritmo.py`:
  >
  > | modelo | ms/chunk | índice completo (128.526) |
  > |---|---|---|
  > | MiniLM-L12 | 28 | **1,0 h** (confirma los 62 min) |
  > | gte-multilingual-base | **2.716** | **97 h** |
  >
  > gte es **97× más lento por chunk**, no 2,6× como decía la extrapolación
  > por parámetros de esta misma sección. Tres factores se multiplican:
  > 2,6× de parámetros, 4× de tokens (MiniLM trunca en 128, gte procesa 512)
  > y la atención **cuadrática y densa** que hay que activar para que gte
  > funcione en CPU (`use_memory_efficient_attention=False`, obligatorio: sin
  > eso el modelo directamente revienta).
  >
  > Parte de la ventaja de MiniLM es su truncación a 128 tokens — o sea que
  > el defecto documentado en la sección 0 **es también lo que lo hace
  > barato**. No hay almuerzo gratis ahí.
  >
  > **Regla nueva: medir ms/chunk sobre 64 chunks reales antes de lanzar
  > cualquier corrida larga.** Cuesta tres minutos y habría evitado lanzar
  > una corrida estimada en 23 minutos que en realidad eran 4,7 horas.
- **Licencia permisiva** (Apache 2.0 o MIT) y **español nativo**, no
  traducción.
- **La ventana debe cubrir 256 tokens**, o volvemos al problema de la
  sección 0.

### Candidatos

| modelo | lic. | par. | ventana | dim | por qué / por qué no |
|---|---|---|---|---|---|
| **`Alibaba-NLP/gte-multilingual-base`** | Apache 2.0 | ~305 M | 8192 | 768 | **El mejor candidato.** Tamaño casi idéntico al e5-base que ya sabemos costear (~6 h), ventana que sobra, 70+ idiomas, y **no necesita prefijos** de consulta/pasaje, o sea una fuente de error menos. |
| `intfloat/multilingual-e5-small` | MIT | 118 M | 512 | 384 | **El más barato que arregla la truncación.** Mismo tamaño que MiniLM (~1 h) pero con ventana de 512. Si hay poco tiempo, es la prueba con mejor relación costo/beneficio. |
| `intfloat/multilingual-e5-large` | MIT | 560 M | 512 | 1024 | Más fuerte, pero **~12 h de CPU** y 1024 dim (el índice pesaría ~527 MB). Solo si sobra una noche entera. |
| `BAAI/bge-m3` | MIT | 568 M | 8192 | 1024 | Muy fuerte en multilingüe y trae denso + léxico en un solo modelo. Mismo problema de costo que e5-large, y su parte léxica reabre BM25, que **ya medimos y perdió 15-4**. |
| familia Qwen3-Embedding | Apache 2.0 | 0,6–8 B | 32k | var. | Lo mejor en los rankings de 2026 y **descartado igual**: backbone decoder (riesgo sec. 8.3) y directamente infactible en CPU. |

### Qué pares tienen sentido en cascada, y con qué comportamiento

La lección ya medida en este proyecto es que **la fusión RRF simétrica entre
dos encoders que no se ponen de acuerdo no funciona** — comparten 11,3 % de
los documentos del top-3 y RRF, que premia el acuerdo, termina intercalando la
lista buena con la mala. La cascada sí funciona porque separa dos trabajos:

- **primario = recall.** Tiene que traer el documento correcto dentro de los
  200 candidatos. Aquí manda la cobertura, y **la ventana de contexto**.
- **secundario = precisión.** Solo reordena lo que ya llegó. Aquí puede ser
  un modelo más caro, porque no codifica nada nuevo: lee vectores ya
  calculados por `reconstruct(fila)`.

Pares recomendados, en orden:

1. **e5-base (primario) + MiniLM (secundario)** — costo cero, ya está todo
   construido. Es el experimento 3.1.
2. **gte-multilingual-base (primario) + e5-base (secundario)** — dos ventanas
   grandes, arquitecturas y datos de entrenamiento distintos (Alibaba vs.
   Microsoft), que es lo que hace que el desacuerdo sea informativo y no
   ruido. ~6 h de codificación.
3. **e5-small (primario) + e5-base (secundario)** — el más barato. Ojo: son
   la **misma familia**, entrenados igual; van a coincidir mucho y el
   secundario aporta poco. Sirve para aislar el efecto de la ventana, no para
   ganar diversidad.

**Regla que vale para cualquier par:** dos modelos de la misma familia se
equivocan igual. La diversidad útil viene de arquitecturas o corpus de
entrenamiento distintos — y aun así hay que medirla contando victorias por
consulta, porque el promedio sobre 41 consultas no distingue un efecto real de
dos consultas que cambiaron de lado por azar.

### Lo que hay que respetar al agregar un encoder

- Declararlo en `KNOWN_ENCODERS` (`src/embedding/encoders.py`) **con sus
  prefijos**: la familia E5 exige `"query: "` / `"passage: "` y omitirlos
  degrada la calidad en silencio. GTE y MiniLM no llevan prefijo.
- **El chunking se hace una sola vez** (invariante del punto 8 de
  `las notas del proyecto`). Nunca re-fragmentar dentro del bucle de encoders: los
  `chunk_id` colisionarían apuntando a textos distintos.
- Índices nuevos siempre a `dev/intermedios/` con `--out-base`, jamás
  directo a `Entrega/base_vectorial/`.
- Cada índice nuevo son cientos de MB: publicarlos por **GitHub Release**, no
  por LFS (punto 16 de `las notas del proyecto`).

---

## 5. Orden sugerido

1. Anotar las 9 consultas que faltan del ground truth. *Bloquea todo lo demás:
   sin esto no se puede decidir nada con evidencia.*
2. Terminar la corrida del e5 y medir la cascada sobre texto limpio.
3. Experimento 3.1 (e5 primario). Cero cómputo nuevo.
4. Solo si 3.1 no alcanza: `multilingual-e5-small` o
   `gte-multilingual-base`, una noche de CPU cada uno.
5. Experimento 3.2 (chunks de 128 tokens) al final, porque rompe el
   invariante del chunking único.

Pase lo que pase, **MiniLM limpio solo es una entrega válida y ya está
construida**. Ninguno de estos experimentos pone en riesgo esa base.

Fuentes de los datos de los modelos: fichas de Hugging Face de
[gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base),
[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
y [bge-m3](https://huggingface.co/BAAI/bge-m3). Los tiempos de CPU son
mediciones propias de esta máquina, no cifras publicadas.
