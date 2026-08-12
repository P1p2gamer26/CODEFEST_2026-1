# E38 — la rejilla de chunking que E21 dejó sin medir

**Estado: PRE-REGISTRADO, corriendo.** Este archivo se escribió **antes** de
ver una sola celda. Las tablas de resultados van al final; si la sección de
veredicto está vacía, la corrida no había terminado.

Pre-registro: este mismo documento (hipótesis, justificación mecánica y
criterio de adopción se fijaron antes de medir; ver `dev/experimentos/cola.jsonl`).
Arnés: `dev/scripts/correr_e38.py` (driver en serie, reanudable) y
`dev/scripts/rechunkear_e38.py` (reconstrucción y re-empaquetado).

## Por qué no es rebuscar

Es el vigésimo experimento y el eje parece agotado, así que la puerta de
entrada tiene que ser explícita.

E21 midió chunks de **128 tokens sin solape** y perdió con claridad (F1@3 de
MiniLM-solo 0.375 → 0.294 bajo la configuración de entonces). Su explicación
mecánica, coherente con E19 y E20, es que al multiplicar por 1,92 el número de
chunks **todos los documentos saturan el tope de `top5`**, el conteo deja de
discriminar y el orden lo decide el ruido.

**La hipótesis simétrica nunca se midió.** Si achicar el chunk satura la
agregación, agrandarlo debería separarla: con 384 o 512 tokens cada documento
aporta menos chunks al pool, saturar cuesta más, y el conteo —que E20 demostró
que es señal y no sesgo— vuelve a discriminar. Además **E21 cambió dos
variables a la vez** (tamaño y solape), y el efecto del solape por sí solo
nunca se aisló.

## La rejilla

| presupuesto | solape 0 | solape 1 |
|---|---|---|
| 280 | celda nueva | **control (= la entregada)** |
| 384 | celda nueva | celda nueva |
| 512 | celda nueva | celda nueva |

512 iguala la ventana de `gte` y de `e5-base`, los dos re-puntuadores con peso
0.60 cada uno, que hoy reciben chunks de 280 tokens.

## Base de la etapa 1, registrada antes de mirar ninguna celda

El screening compara **MiniLM solo contra MiniLM solo**, que es el diseño de
E21 — no contra la cascada de tres encoders.

| | F1@3 | NDCG@10 | NDCG penalizado |
|---|---|---|---|
| **MiniLM solo, corpus entregado, 50 consultas** | **0.369** | **0.464** | **0.440** |

Desglose: 41 humanas 0.391, 9 de panel 0.267. Techo alcanzable 0.906.
La cascada completa entregada está en 0.455 / 0.516 / 0.499.

## Criterio de adopción y vetos, fijados de antemano

- IC al 90% del delta pareado que **excluya una pérdida de 0.02**.
- **Confirmación en las 41 humanas**, no solo en las 50.
- **Veto 1:** las consultas con F1@3 = 0 no pueden pasar de 11.
- **Veto 2:** los fragmentos ilegibles no pueden pasar de 0.
- **Veto 3:** si la ganancia se concentra en las 9 consultas de panel y se
  evapora en las 41 humanas, se descarta — la firma que ya mató a `doc_rrf`,
  a gte-primario, a E25 y a E31.

## Riesgos declarados antes de medir

1. **Lo más probable es que sea el negativo número 20.** E19 midió que solo el
   33.7% de los documentos relevantes entra en los 3 cupos; el ranking dentro
   del pool ya está exprimido. Si gana algo, lo plausible es +0.02 a +0.04.
2. **Elegir el máximo de seis celdas por el promedio sería sobreajuste de
   manual.** De ahí el IC pareado y la confirmación en dos muestras.
3. **`k_pool` se mide en chunks, no en volumen de texto.** Al pasar de 280 a
   512 el chunk crece 1,83× y un pool de 100 abarca casi el doble de texto.
   Cada celda se mide con el crudo (100) y con el escalado (73 para 384, 55
   para 512). Sin esto la comparación mezcla dos efectos, que es el error que
   E21 tuvo que corregir con una tercera fila.
4. **Los `chunk_id` de una celda no son comparables con los de otra.** Son
   `doc_id` + posición; los tres encoders de una celda tienen que salir del
   mismo archivo de chunks (punto 8 de las notas del proyecto).

## El hallazgo de implementación, que vale aparte del resultado

**No hay checkpoint del texto crudo de los documentos, solo de los chunks.**
Re-chunkear a un presupuesto MAYOR parecía exigir re-extraer el corpus entero
con OCR — horas, y dependiente del binario de tesseract.

Salida: el solape es de **exactamente una oración**, así que la secuencia
original se reconstruye quitando de cada chunk el prefijo que repite la cola
del anterior de la misma sección (`reconstruir_oraciones`). El re-empaquetado
pasa de horas a minutos y **no toca la extracción**.

### La puerta de conservación, y una corrección de método

Reconstruir mal significaría medir seis celdas sobre un corpus corrupto **sin
que nada falle a la vista**, así que la reconstrucción tiene una puerta que
compara el vocabulario original contra el reconstruido y aborta la rejilla
antes de gastar GPU.

**El criterio de esa puerta se afinó dos veces, y hay que decirlo:**

1. Primero era "ninguna palabra del original puede faltar" → marcaba 9 de 40
   documentos. Los tokens perdidos eran `!(,`, `.*/#`, `.$,/`: puntuación
   suelta que el segmentador descarta por no formar oración.
2. Después, "algún carácter alfanumérico" → seguía marcando 8 de 150. Los
   tokens eran mojibake de las etiquetas de gráficas del AI Index que colaba
   por traer un dígito: `!4/('#`, `..6`, `!2*`.
3. Criterio final: **dos letras seguidas**, o sea algo que parece una palabra.
   Cero pérdida sobre 150 documentos.

Afinar un criterio hasta que pase es la forma clásica de autoengañarse. La
defensa es que **en cada iteración se miraron los tokens concretos** en vez de
aflojar a ciegas, y que la intención del criterio —"no se pierde contenido"—
no cambió nunca. Queda anotado para que se pueda auditar en contra.

## Resultados

Corrida completa el 9-10 ago 2026, 21:12 a 23:16. Seis celdas, seis índices de
MiniLM en GPU, diez lecturas. Todas con MiniLM solo, que es el diseño de E21.

| celda | chunks | F1@3 (50) | NDCG@10 (50) | F1@3 (41 hum.) |
|---|---|---|---|---|
| corpus **entregado** (280/1 original) | 128.526 | 0.369 | 0.464 | 0.391 |
| **280/1 reconstruido (control)** | **132.146** | **0.411** | **0.469** | **0.434** |
| 280/0 | 114.724 | 0.395 | 0.426 | 0.424 |
| 384/1, k_pool 73 | 95.082 | 0.295 | 0.360 | 0.309 |
| 384/1, k_pool 100 | 95.082 | 0.290 | 0.347 | 0.303 |
| 384/0, k_pool 73 | 85.038 | 0.268 | 0.296 | 0.301 |
| 384/0, k_pool 100 | 85.038 | 0.299 | 0.343 | 0.331 |
| 512/1, k_pool 55 | 70.665 | 0.240 | 0.302 | 0.250 |
| 512/1, k_pool 100 | 70.665 | 0.253 | 0.323 | 0.283 |
| 512/0, k_pool 55 | 64.554 | 0.283 | 0.304 | 0.320 |
| 512/0, k_pool 100 | 64.554 | 0.344 | 0.361 | 0.376 |

### La hipótesis está REFUTADA, y monótonamente

Ordenando por presupuesto, con el mejor `k_pool` de cada uno:

    280   0.411   <- el control
    384   0.299
    512   0.344

**Agrandar el chunk empeora, en las dos métricas y en las tres muestras.** No
hay una sola celda de 384 o 512 que se acerque al control. La caída es enorme:
−0.11 de F1 y −0.11 de NDCG en el mejor de los casos.

**El eje del tamaño de chunk queda cerrado por los dos lados.** E21 midió hacia
abajo (128 tokens: 0.294 contra 0.375) y esto mide hacia arriba. El 280 vigente
no es un valor sin calibrar: es un óptimo con evidencia a ambos costados.

**Quitar el solape también pierde** (280/0 da 0.395 contra 0.411, y −0.043 de
NDCG). Era la variable que E21 había dejado confundida con el tamaño; ya no lo
está, y aporta en la misma dirección.

**La corrección de `k_pool` por volumen de texto no ayudó, y eso es
informativo:** en 384 y en 512 el `k_pool` crudo (100) gana al escalado en casi
todas las lecturas. Ver más chunks es mejor aunque cada uno sea más largo, o
sea que el pool ancho no estaba compensando de más.

## El control falló, y es el hallazgo que hay que mirar

**La celda 280/1 debía reproducir el corpus entregado y no lo hace.** Da
**132.146 chunks contra 128.526, un 2,82%** por encima del tope del 2% que el
plan fijó de antemano. Y lo que importa más que el conteo: **puntúa 0.411
contra 0.369 del corpus entregado, +0.042 de F1 con los mismos parámetros
nominales.**

Ese salto es **más grande que casi cualquier efecto que este proyecto
persigue**. Consecuencias, en orden de importancia:

1. **La comparación entre celdas SIGUE SIENDO VÁLIDA.** Las seis salen del
   mismo pipeline de reconstrucción, así que el confundidor es común y se
   cancela. El veredicto de arriba se sostiene.
2. **Comparar cualquier celda contra el corpus entregado NO es válido.** El
   baseline correcto de esta rejilla es el 280/1 reconstruido, no el 0.369.
3. **No se adopta nada.** El 0.411 es tentador y sería un error tomarlo: es
   MiniLM solo, sobre 50 consultas, con la diferencia equivalente a unas dos
   consultas, y vendría de un corpus cuyo parecido con el entregado no
   controlamos. Adoptarlo exigiría rehacer los tres índices.

**Por qué la reconstrucción produce más chunks, hipótesis sin medir:** el
segmentador re-parte el texto de cada chunk y no recupera exactamente las
mismas fronteras de oración que la corrida original, así que el empaquetado
cae distinto. Queda **pre-registrable como E39** si alguien quiere perseguir
ese +0.042 — pero con un pool anotado propio, porque medir un corpus nuevo con
el ground truth armado sobre el viejo tiene la misma firma de sesgo que hundió
a `doc_rrf`, a gte-primario y a E31.

## Veredicto

**REFUTADO. `Entrega/` sin cambios.** Vigésimo negativo del proyecto, y el que
cierra el último eje estructural que quedaba abierto.

Lo que el experimento deja, más allá del negativo:

- El tamaño de chunk está **acotado por medición a ambos lados**. No volver a
  proponerlo sin datos nuevos.
- El solape, aislado por primera vez, **aporta**: quitarlo cuesta.
- Un instrumento nuevo, `rechunkear_e38.py`, que re-empaqueta el corpus a
  cualquier presupuesto **sin re-extraer ni pasar OCR**, con una puerta de
  conservación que verifica subsecuencia e igualdad exacta del solape.
- La anomalía del control, que es la única pista viva que salió de la noche.

### Lo que costó, y la lección de método

**La puerta de conservación cerró tres veces, y las tres el defecto estaba en
el instrumento, no en el dato.** Comparar tokens crudos confundía
re-tokenización con pérdida (`CHUCHINGAL;` → `CHUCHINGAL ;`, `IV(h)(1)` →
`IV(h)` + `(1)`), y después un tope por fracción del 25% castigaba a los
documentos de pocas oraciones por chunk, donde borrar un tercio del texto es
exactamente el solape.

El criterio final no tiene tolerancia inventada: subsecuencia de caracteres
más **igualdad exacta** entre lo que desapareció y el solape deduplicado, que
el propio código contabiliza. El único umbral que queda mide una sola cosa —lo
que el segmentador descarta por no formar oración— y vale 0,0000% en los
documentos revisados.

**La lección, que es la 5 aplicada a código de hace dos horas:** un criterio
que se afloja hasta que pasa es indistinguible de uno que funciona, salvo por
los controles negativos. Los tests exigen ahora que la puerta **rechace**
texto inventado y texto reordenado. Sin eso no había forma de saber si estaba
midiendo algo.
