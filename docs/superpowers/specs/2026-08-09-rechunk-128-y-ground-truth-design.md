# E21 — Re-chunking a 128 tokens sin solape, y cierre del ground truth

Fecha: 9 de agosto de 2026. Estado: aprobado, pendiente de plan de implementación.

## Por qué

Veinte experimentos cerrados (E01–E20) midieron **aguas abajo de una decisión
de chunking que nunca se varió**: `CHUNK_TOKEN_BUDGET = 280`,
`CHUNK_OVERLAP_SENTENCES = 1`. La familia de la agregación a documento quedó
cerrada por E19/E20, el pool quedó descartado como cuello de botella por E18 y
el eje "otro encoder" por E04. El chunking es el único eje estructural que
queda sin medir.

Dos hechos mecánicos lo justifican, escritos antes de medir:

1. **El primario no lee la mitad del corpus.** `paraphrase-multilingual-MiniLM-L12-v2`
   trunca en 128 tokens y la mediana de los chunks entregados es de **256
   tokens** (p90 = 277). Se intentó esquivar esto cambiando de encoder —usar
   e5, de ventana 512, como primario— y **empeoró** (0.250 contra 0.344). Esa
   medición refuta "una ventana mayor recupera mejor"; **no** refuta "ajustar
   el chunk a la ventana del primario recupera mejor". Son hipótesis distintas
   y la segunda no está medida.
2. **El solape hace inservible el presupuesto de fragmento.** Los fragmentos
   entregados tienen mediana de **153 palabras de las 250 permitidas**. Se
   implementó concatenar chunks vecinos para llenarlo y se revirtió porque
   **duplicaba texto**: el chunker solapa una oración entre chunks
   consecutivos por diseño. Con solape 0 la causa desaparece y la
   concatenación vuelve a ser posible.

El punto 2 apunta a **NDCG@10**, que es la mitad de la nota, que se midió una
sola vez antes de agosto y que es la métrica con 6–9× más n efectivo que el
F1@3 — la única con la que este ground truth puede resolver algo.

## Lo que NO promete

No hay garantía de ganancia. Con 50 consultas el efecto mínimo detectable en
F1@3 es ~0.05; una mejora real de 0.02 saldría como "no concluyente". El
diseño acota el coste (≈3 h de máquina hasta el veredicto) y deja la entrega
intacta, no promete el resultado.

## Vía B — máquina

### B0. Pre-registro

Entrada E21 en `dev/experimentos/cola.jsonl` con hipótesis, justificación
mecánica, comando, riesgo y **criterio de adopción fijado antes de medir**
(lección 3). Criterio: IC al 90% del delta pareado de **NDCG@10** excluyendo
una pérdida de 0.02, en las 50 y en las 10 independientes.

### B1. Subdividir, no re-extraer

`dev/scripts/subdividir_chunks.py` parte cada registro de
`dev/intermedios/chunks_intermedios_limpio.jsonl` (texto ya limpio, guiones
reparados) en trozos de ≤128 tokens en frontera de oración, **solape 0**.
Salida: `dev/intermedios/chunks_128.jsonl`.

No se re-extrae ni se pasa OCR: el texto limpio ya existe en disco.

**Trampa que hay que manejar:** el corpus de partida ya trae el solape de una
oración. Al subdividir hay que **descartar la primera oración de cada chunk
padre salvo el primero de cada documento**, porque es duplicado del final del
padre anterior. Sin eso, el solape se cuela en el corpus nuevo y B5 vuelve a
duplicar texto.

`chunk_id` se re-numera por documento y posición. Toda la metadata obligatoria
de la Tabla 1 se conserva; `num_tokens` se recalcula con el tokenizer de
MiniLM.

Chequeo runnable (`assert` en `__main__`, sin framework):
- ningún chunk supera 128 tokens;
- ninguna oración aparece en dos chunks consecutivos del mismo documento;
- el texto concatenado por documento cubre el del corpus viejo menos el solape.

### B2. Índice MiniLM

`build_corpus_index.py --desde-chunks dev/intermedios/chunks_128.jsonl`, con
`.venv-cuda` (GPU), salida a `dev/intermedios/rechunk128/`.
**`Entrega/` no se toca.** ≈2 h.

### B3. Veredicto — punto de corte

`eval_mini.py` sobre las 50, las 41 humanas y las 10 independientes: F1@3,
NDCG@10 binario y penalizado, con IC al 90%.

- Si **NDCG@10 no sube ni en las 41 ni en las 10**: se cierra E21, se escribe
  el resultado negativo en `dev/experimentos/E21_*.md` y **no se construyen
  gte ni e5**. Coste total: 3 h.
- Si sube: sigue B4.

### B4. Los otros dos encoders

gte en GPU (≈15 h) y e5, **uno a la vez** — quedan 24 GB de disco y el corpus
re-chunkeado aproximadamente duplica el número de vectores. Re-medir la
cascada completa. Solo con eso en verde se plantea tocar `Entrega/`.

### B5. Reabrir la concatenación de vecinos

Con solape 0, rellenar los fragmentos hasta las 250 palabras con el chunk
contiguo deja de duplicar texto. Es el segundo tiro a NDCG@10 y su coste es
casi nulo una vez existe el corpus nuevo. Se mide como experimento aparte, con
su propio pre-registro.

## Vía A — humano, en paralelo

### A3 — ahora

Re-anotar a mano las 9 consultas con etiqueta de agente (q001, q007, q008,
q011, q012, q015, q028, q038, q048). El panel de agentes ya se midió y
reproduce al humano con F1 0.23: **no es tarea delegable**.

Claude prepara el material (`anotar_candidatos.py --generar --rescate --solo …`,
con `--terminos` donde la consulta va en español y el corpus en inglés) y el
humano marca. Las etiquetas son por `doc_id`, así que **B no las invalida**:
este trabajo sube el piso gane o pierda E21.

Cuidado con q001 y q038: sus documentos relevantes existen pero el pool denso
no los contiene. No anotarlas con ceros — los candidatos salen del rescate
léxico (CBRN para q001, subcorpus ALERTAS para q038).

### A1 — después de B3

Generar `fragmentos.md` sobre la **unión** de los fragmentos de la
configuración vieja y la nueva (pooling estándar), escala graduada
2 / 1 / 0, y anotar **una vez** para las dos. Es la única medición no-proxy
que tendría el proyecto y sirve para comparar las dos configuraciones sin
re-anotar.

Sesgo a declarar al reportarlo: se anota sobre lo que los sistemas
entregaron, así que sirve para comparar configuraciones, no para estimar la
nota de ADL.

## Invariantes que no se rompen

1. `Entrega/` sigue reproducible byte a byte durante todo B. El corpus nuevo
   vive en `dev/intermedios/`.
2. El chunking sigue haciéndose **una sola vez** por corpus: los tres índices
   de una misma configuración se construyen sobre los mismos `chunk_id`. Lo
   que cambia es que ahora hay **dos corpus**, el viejo y el de 128, y no se
   mezclan nunca.
3. `Entrega/generador.py` sigue autocontenido. No se toca hasta B4 en verde, y
   si se toca hay que re-aplanar y repetir la corrida en frío.
4. Sin modelos generativos en ningún punto (sec. 8.3).
