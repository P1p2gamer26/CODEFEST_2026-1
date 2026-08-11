# E38 — la rejilla de chunking que E21 dejó sin medir

**Estado: PRE-REGISTRADO, corriendo.** Este archivo se escribió **antes** de
ver una sola celda. Las tablas de resultados van al final; si la sección de
veredicto está vacía, la corrida no había terminado.

Spec: `dev/docs/superpowers/specs/2026-08-09-rejilla-chunking-design.md`
Plan: `dev/docs/superpowers/plans/2026-08-09-e38-rejilla-chunking.md`
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
   mismo archivo de chunks (punto 8 de CLAUDE.md).

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

Pendiente: la corrida no había terminado cuando se escribió este archivo.

## Veredicto

Pendiente.
