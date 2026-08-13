# E42 - normalizacion por tamano de documento en la agregacion (12 ago 2026)

## Hipotesis (copiada del brief, previa a la medicion)

Dividir el score agregado por documento por el numero de chunks que aporta
al pool (o al corpus), elevado a un exponente pequeno alpha, penaliza al
documento que inunda el pool con muchos chunks mediocres sin castigar al que
aporta pocos chunks buenos.

## Justificacion mecanica (copiada del brief, previa a la medicion)

E41 midio con arnes fiel que los 11 ceros no son de pool ciego sino de
ranking: en q003/q011/q034/q037/q044/q046/q047 el documento relevante esta
dentro del top-100 y tiene el **mejor chunk del pool** (score top1
1.667-1.751, por encima del ganador), pero aporta **2-9 chunks** mientras el
ganador aporta **8-38**. Con `top5` el ganador satura el tope de cinco
sumandos y el relevante no llega. Los margenes son minusculos: q037 7.6414
vs 7.6501 (-0.0087), q047 5.8903 vs 5.9011 (-0.0108).

Esto **no es el eje de `topM`**, que E07/E19/E20/E33/E37 cerraron: `topM` es
un **tope** y ya esta saturado en los dos documentos, asi que ninguna M los
distingue (por eso `top8` y `top12` dieron resultados identicos en E33). Lo
que se propone es un **normalizador**, una operacion distinta: dividir el
score agregado por el numero de chunks que el documento aporta al pool,
elevado a un exponente pequeno. Con alpha pequeno el efecto es del orden del
1% del score, que es exactamente el orden de los margenes que E41 midio, y
penaliza al que inunda sin castigar al que aporta pocos chunks buenos.

**Riesgo declarado antes de medir:** un alpha grande degenera en `mean`, que
E41 midio y es catastrofico (0.0567, 43 ceros). La grilla se pre-registra
pegada a cero y **no se extiende hacia arriba** aunque el borde superior
gane.

## Grilla pre-registrada, cerrada

alpha en {0.00, 0.02, 0.05, 0.10, 0.20}, denominador en {`n_pool` (chunks
del documento en el pool de 100), `n_corpus` (chunks del documento en todo
el corpus, via `metadata.jsonl`)}. Diez celdas, alpha=0.00 es la base y se
calcula una sola vez (el divisor es 1.0 sin importar el denominador). Nueve
celdas efectivas. **No se anadieron celdas despues de ver los resultados.**

**Bug de arnes encontrado y corregido antes de leer ningun numero:** el
`main()` del brief tenia `if alpha == 0.0: break` dentro del loop de alphas,
que rompia el loop entero de alphas para los dos denominadores -- solo
corrian 2 de las 9 celdas (los dos `a0.00`). Corregido a
`if alpha == 0.0 and denom != DENOMINADORES[0]: continue`, que salta SOLO la
repeticion de la celda base en el segundo denominador. Verificado: la
segunda corrida imprimio las 9 celdas.

## Puerta de fidelidad del arnes (Step 6)

`top-3 identicos: 50 de 50` contra `Entrega/resultados.jsonl`. La celda base
(`n_pool:a0.00`) reproduce exactamente el sistema entregado.

## Tabla completa de las 9 celdas

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | ceros |
|---|---|---|---|---|---|---|
| **n_pool:a0.00 (BASE)** | **0.4547** | **0.5149** | **0.4979** | **0.4333** | **0.4735** | **11** |
| n_pool:a0.02 | 0.4727 | 0.5138 | 0.4962 | 0.4000 | 0.4584 | 10 |
| n_pool:a0.05 | 0.4527 | 0.4906 | 0.4698 | 0.3333 | 0.3731 | 11 |
| n_pool:a0.10 | 0.4460 | 0.4831 | 0.4630 | 0.3333 | 0.3502 | 11 |
| n_pool:a0.20 | 0.4380 | 0.4697 | 0.4510 | 0.3333 | 0.3649 | **12 (veto)** |
| **n_corpus:a0.02** | **0.4740** | **0.5345** | **0.5137** | **0.4333** | **0.4735** | **9** |
| n_corpus:a0.05 | 0.4540 | 0.5135 | 0.4946 | 0.4000 | 0.4351 | 11 |
| n_corpus:a0.10 | 0.4193 | 0.4802 | 0.4609 | 0.3000 | 0.3530 | **12 (veto)** |
| n_corpus:a0.20 | 0.3780 | 0.4406 | 0.4238 | 0.2667 | 0.3127 | **15 (veto)** |

La base reproduce 0.455 / 0.516 / 0.499 de dev/docs/PLAN_MAESTRO.md (redondeo).

## IC al 90% del delta pareado contra la base

| celda | metrica | delta | IC 90% |
|---|---|---|---|
| n_pool:a0.02 | ND(50) | -0.0011 | [-0.0248, +0.0232] — **no pasa** (limite inferior < -0.02) |
| n_pool:a0.02 | ND(ind) | -0.0151 | [-0.0828, +0.0525] |
| n_pool:a0.05 | ND(50) | -0.0243 | [-0.0655, +0.0136] — no pasa |
| n_pool:a0.10 | ND(50) | -0.0318 | [-0.0858, +0.0169] — no pasa |
| **n_corpus:a0.02** | **ND(50)** | **+0.0197** | **[-0.0080, +0.0491] — pasa** |
| **n_corpus:a0.02** | **ND(ind)** | **+0.0000** | **[+0.0000, +0.0000] — pasa** |
| n_corpus:a0.05 | ND(50) | -0.0014 | [-0.0456, +0.0418] — no pasa |

(las celdas con mas de 11 ceros no se evaluaron por el criterio de IC: caen
antes, en el veto).

## Veredicto por la regla del Step 8, en orden

1. **Veto (>11 ceros):** descarta `n_pool:a0.20` (12), `n_corpus:a0.10` (12)
   y `n_corpus:a0.20` (15).
2. **IC al 90% de ND excluyendo -0.02 en las dos muestras:** de las seis
   celdas restantes (mas la base), solo **`n_corpus:a0.02`** tiene el limite
   inferior de ND(50) y ND(ind) por encima de -0.02. Las otras cuatro
   (`n_pool:a0.02`, `n_pool:a0.05`, `n_pool:a0.10`, `n_corpus:a0.05`) fallan
   en ND(50).
3. **Alpha mas chico entre las que sobreviven:** trivial, sobrevive una sola
   celda.
4. No aplica (no es el caso "ninguna sobrevive").

**E42 es POSITIVO: `n_corpus:a0.02` pasa el criterio completo.**
F1(50) 0.4547 -> 0.4740, ND(50) 0.5149 -> 0.5345, ceros 11 -> 9,
**F1(ind)/ND(ind) exactamente iguales (delta 0.0000, IC [0,0])** — no
confirma ganancia en la muestra independiente, pero tampoco la dana ni una
milesima; ninguna de las 10 consultas independientes cambio de documentos.

## Consultas que ganan y pierden (`n_corpus:a0.02` contra la base), por procedencia

29 de 50 lineas de `documents` cambian de orden o composicion (el
normalizador reordena mucho mas de lo que mueve el F1). De ellas, el F1@3
cambia en 6:

**Ganan:**
- **q003** (humana): F1 0.00 -> 0.50. Rescata `F1-CSET-103` en el 3er
  cupo, desplazando a `F3-SIPRI-100`. Es exactamente el patron de E41
  (relevante con mejor chunk, pocos chunks).
- **q032** (humana): F1 0.67 -> 1.00. `F3-SIPRI-002`/`F1-DAIO-015` etc.
  reordenan; entra el 3er relevante.
- **q037** (humana): F1 0.00 -> 0.40. Rescata `F3-MAPPOEA-031` en el 2do
  cupo, el caso citado en la justificacion mecanica (margen -0.0087).
- **q047** (humana): F1 0.00 -> 0.40. Rescata `F3-MAPPOEA-032` en el 3er
  cupo, el otro caso citado (margen -0.0108).

**Pierden:**
- **q028** (anotacion-asistida): F1 0.67 -> 0.33. Pierde `F2-SWF-124`, gana
  `F2-CSIS-208` (no relevante).
- **q029** (humana): F1 0.33 -> 0.00. Pierde `F2-SWF-081`, gana
  `F2-CSIS-035` (no relevante).

**Lectura:** 4 ganancias son consultas **humanas** y solo 1 perdida es de
**anotacion-asistida** (la otra perdida tambien es humana). Es la firma
**contraria** a la de E31/E37/E39 (donde la ganancia se concentraba en
consultas de anotacion asistida) — aca la ganancia neta se sostiene mayormente sobre
etiquetas confiables. Las 6 consultas **independientes** que cambian de
orden (q005, q014, q017, q020, q026, q044) no cambian su F1@3 ni su NDCG@10:
son reordenamientos internos del top-3 que no alteran el conjunto de
documentos o no alteran cuales son relevantes.

## Que no se decidio en esta tarea

Este experimento **mide** que `n_corpus:a0.02` pasa el criterio de adopcion
del proyecto. **No se aplico a `Entrega/generador.py`** ni se toco ningun
encoder ni indice: la tarea pedia solo la medicion y su registro en la
bitacora. Llevarlo a la entrega (aplanar `agregar_normalizado` en
`generador.py`, decidir si reemplaza a `aggregate_documents` con
`agg_strategy="top5"`, medir el coste de leer `metadata.jsonl` completo para
`conteos_del_corpus` en la fase online) es una decision posterior, fuera del
alcance de esta tarea.

---

## CIERRE: REFUTADO bajo E46 (12 ago 2026)

Todo lo de arriba se midio con `top5` y `k_pool=100`. **E46 cambio las dos
cosas** (calibracion min-max de los encoders, `top6`, `k_pool=200`), y la
justificacion mecanica de E42 dependia justamente de ellas: la premisa era que
el documento ganador **satura los cinco sumandos** de `top5` y desplaza al
relevante por margenes de 0.01. Con seis sumandos sobre un pool del doble, ese
regimen no existe.

Re-medido con el **generador real** (no con el arnes, ver la nota de abajo),
alpha aplicado en `aggregate_documents` despues del `topM`:

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | ceros | lineas |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BASE (alpha=0)** | **0.4993** | **0.5580** | **0.5387** | 0.4333 | 0.4766 | **8** | 0 |
| alpha=0.02 | 0.4793 | 0.5346 | 0.5165 | 0.4333 | 0.4766 | 9 | 12 |
| alpha=0.05 | 0.4660 | 0.5145 | 0.4978 | 0.4000 | 0.4488 | 9 | 25 |

Deltas pareados con IC al 90% contra la base, para alpha=0.02: F1(50)
**-0.0200 [-0.0400, -0.0067]**, ND(50) **-0.0234 [-0.0408, -0.0086]**, NDp(50)
**-0.0222 [-0.0387, -0.0080]**. Los tres IC quedan **enteramente bajo cero**, y
ademas **dispara el veto**: las consultas con F1@3 = 0 suben de 8 a 9. En las
10 independientes es exactamente inerte (delta 0.0000, IC [0, 0]).

**Pierde monotonamente con alpha**, igual que el borde superior de la grilla
original. **No reabrir sin un cambio de regimen en la agregacion.**

El control se corrio primero: con `alpha=0.0` el generador reproduce
`Entrega/resultados.jsonl` **byte a byte** (sha256 `fcd5f423...`), asi que la
rama nueva es inerte por construccion y la comparacion es limpia.

**El codigo se revirtio.** Un parametro que solo puede empeorar es superficie
de riesgo sin contrapartida; la medicion queda aca, que es donde sirve.

### Advertencia sobre los arneses de E42 y E45

`barrido_norm_doc_e42.py` **no tiene puerta de fidelidad**: nunca se comprobo
que su celda base reprodujera `Entrega/resultados.jsonl`. El arnes hermano
`barrido_grafo_e42_e45.py` **si la tiene y falla**: su celda base diverge de la
entrega en los **fragmentos** de 7 consultas (q001, q008, q018, q019, q022,
q025, q026) con los 150 documentos identicos. Se descarto que fuera el grafo
(medido inerte: 0 de 50 lineas cambian al pasar `--sin-grafo`) y que fuera el
volcado de pools (se regenero y da igual); los fragmentos entregados **si**
estan todos en el pool volcado, o sea que la infidelidad esta en como el arnes
reconstruye o desempata.

Es la regla que E09 ya habia dejado escrita —*si la fila base del arnes no
reproduce digito a digito lo que da `eval_mini.py` sobre `Entrega/`, el barrido
no se lee*— y explica por que este cierre se hizo con el generador real, que
ademas cuesta 4 minutos por celda.
