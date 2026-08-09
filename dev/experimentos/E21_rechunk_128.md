# E21 — Re-chunking a 128 tokens sin solape

Estado: **en curso**. Pre-registrado el 9 de agosto de 2026 antes de medir.

## Hipotesis

Re-chunkear el corpus a 128 tokens sin solape mejora el NDCG@10. Es el unico
eje estructural que E01–E20 no tocaron: los veinte midieron aguas abajo de un
chunking fijo de 280 tokens con solape de una oracion.

## Justificacion mecanica (escrita antes de medir)

1. **El primario no lee la mitad del corpus.** MiniLM trunca en 128 tokens y
   la mediana de los chunks entregados es 256 (p90 = 277). Se intento
   esquivarlo usando e5 (ventana 512) como primario y **empeoro** (0.250 contra
   0.344). Esa medicion refuta "una ventana mayor recupera mejor"; **no**
   refuta "ajustar el chunk a la ventana del primario recupera mejor".
2. **El solape hace inservible el presupuesto de fragmento.** Los fragmentos
   entregados tienen mediana de 153 palabras de las 250 permitidas.
   Concatenar vecinos para llenarlas se implemento y se revirtio porque
   duplicaba texto, y la causa era el solape. Con solape 0 desaparece.

## Criterio de adopcion, fijado de antemano

IC al 90% del delta pareado de **NDCG@10** excluyendo una perdida de 0.02, en
las 50 y en las 10 independientes. **Punto de corte tras el indice de MiniLM:**
si el NDCG@10 no sube ni en las 41 ni en las 10, se cierra sin construir gte
ni e5 (se ahorran ~15 h de GPU).

## B1 — el corpus nuevo, verificado

`dev/scripts/subdividir_chunks.py` parte de `chunks_intermedios_limpio.jsonl`
(texto ya limpio, guiones reparados): **no re-extrae ni pasa OCR**, porque el
defecto no esta en el dato sino en el tamano de la unidad. Reusa
`_pack_sentences` del pipeline en vez de copiar el empaquetador.

| | viejo | nuevo |
|---|---|---|
| chunks | 128.680 | **247.522** (1,92x) |
| documentos | 1.829 | 1.829 (ninguno perdido) |
| tokens p50 | 256 | **110** |
| tokens p90 | 277 | 139 |
| chunks que exceden la ventana de MiniLM | ~50% | **11,8%** |
| palabras p50 | 153 | 64 |
| `chunk_id` unicos | si | si |

**Tres cosas que hay que saber de este corpus:**

1. **La cobertura de palabras por documento baja a 0,86 (min 0,79) y eso es
   correcto**: es el solape de una oracion que se quito, no texto perdido.
   Verificado directamente — de 25 documentos PDF al azar, 300 oraciones del
   corpus viejo no aparecen como cadena exacta en el nuevo, y **las 300 estan
   presentes como subcadena del texto del documento**. La diferencia es
   re-segmentacion en las fronteras de chunk, no perdida.
2. **El 11,8% que pasa de 128 tokens son oraciones sueltas mas largas que el
   presupuesto.** El chunker nunca parte una oracion (garantia de completitud
   linguistica, sec. 3.3), asi que ese piso es estructural y no se puede bajar
   sin romper la garantia.
3. **Se pierde `titulo_seccion`.** Los encabezados Markdown no sobrevivieron a
   la extraccion a chunks, asi que al re-chunkear desde el checkpoint no son
   recuperables. No es metadata obligatoria de la Tabla 1. Si alguna vez
   importa, hay que re-chunkear desde la extraccion y eso si cuesta las horas
   de OCR.

Los formatos tabulares (csv, xlsx) pasan intactos: cada fila ya es una unidad
atomica y no llevaba solape.

## Invariante que este experimento NO rompe

El chunking sigue haciendose una sola vez por corpus. Lo que hay ahora son
**dos corpus distintos** —el de `Entrega/` y el de 128— y sus indices no se
mezclan nunca: los `chunk_id` no son comparables entre ellos. `Entrega/` no se
toca y sigue reproducible byte a byte durante todo E21.

## Traspie de entorno, por si se repite

`.venv-cuda` no tenia `spacy`. No hace falta para reindexar con
`--desde-chunks`, pero la cadena de imports de `build_index` lo arrastra via
`ingestion.pipeline`. Se instalo en `.venv-cuda` sin tocar `.venv`, que es el
que produce la entrega reproducible.

## B2 — pendiente

Indice MiniLM sobre `chunks_128.jsonl` en GPU, salida a
`dev/intermedios/rechunk128/`.

## B3 — pendiente

Veredicto con `eval_mini.py` sobre las 50, las 41 humanas y las 10
independientes.
