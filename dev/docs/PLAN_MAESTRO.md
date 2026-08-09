# Plan maestro — CODEFEST AD ASTRA 2026, Etapa 1

Estado al **9 de agosto de 2026**. Este documento es el punto de entrada para
cualquiera que retome el proyecto: qué hay, qué se probó, qué falló, qué queda
y qué hace falta para competir. Los detalles de implementación están en
`las notas del proyecto`; el método de decisión, en `lecciones_metodologia.md`.

**Fechas:** informe a finalistas **20 de agosto**. Fase final presencial en
Bogotá **18-19 de septiembre**.

---

## 1. Dónde estamos

### 1.1 La entrega está completa, válida y reproducible

| | |
|---|---|
| `resultados.jsonl` | 50 consultas, 3 documentos + 10 fragmentos cada una |
| Fragmentos que exceden 250 palabras | **0** |
| Fragmentos con palabras partidas | **0** (eran 166) |
| `base_vectorial/` | **3 encoders** × 128.526 vectores, metadata alineada |
| Grafo (bonus) | 224.101 nodos, 754.876 aristas |
| `informe_tecnico.pdf` | 8 de 8 páginas |
| `validar_entrega.py` | ✅ en verde (ahora falla también ante archivos de más) |
| `pytest dev/tests` | ✅ **138 passed** |
| `pruebas_robustez.py` | ✅ todas — el script corrido como lo correrá ADL |
| **Corrida en frío** | ✅ **reproduce byte a byte** (sha256 `987293ac…`) |

**Dos fallos que habrían excluido la entrega, encontrados y corregidos el
2 ago 2026** — ninguno se veía corriendo el camino feliz:

- **Sintaxis de Python 3.10** (`X | None`) sin `from __future__ import
  annotations`, cuando ADL evalúa con **≥ 3.9.5**. El script moría en el
  import. No se detectaba porque el venv local corre 3.13.
- **BOM en el archivo de consultas.** Se leía con `utf-8`; cualquier archivo
  guardado con Excel o Bloc de notas en Windows lleva BOM y la primera línea
  fallaba. El archivo lo entrega ADL y no controlamos cómo lo guardan.

Esa última línea es la que evita la exclusión: la sec. 1.4 dice que si el
evaluador no puede reproducir los resultados, el equipo queda fuera. Se
verifica corriendo `generador.py` desde un directorio fuera del repo, con
`PYTHONPATH` vacío y sin más flag que `--consultas`.

### 1.2 Las métricas

| métrica | valor | sobre qué |
|---|---|---|
| **F1@3** | **0,440** | las 50 consultas — **el 49% del techo de 0,906**, no el 44% de 1 |
| **NDCG@10** | **0,506** | las 50; aproximado, relevancia heredada del documento |
| **NDCG@10 penalizado** | **0,491** | las 50, descontando aparato bibliográfico |
| F1@3 / NDCG@10 | 0,468 / 0,510 | 41 de anotación humana |
| F1@3 / NDCG@10 | 0,400 / 0,436 | 10 sin sesgo de pooling |

**La métrica de decisión es la media sobre las 50** (forma de las ecs. 10 y 14
del PDF). Los desgloses son diagnóstico. Y el techo es **0,906, no 1**: hay que
entregar exactamente 3 documentos, así que una consulta con un solo relevante
topa en 0,50.

**Advertencia que hay que repetir siempre:** 9 de las 50 llevan etiqueta de
panel de agentes y dan F1@3 **0,311** contra **0,468** de las 41 humanas. Ese
es el eslabón podrido del promedio, y re-anotarlas a mano sigue siendo la tarea
de mayor impacto del proyecto.

El 2 de agosto eran 0,344 y **0,206**. **El NDCG@10 se ha multiplicado por
2,5** y es la mitad del puntaje; había sido medido **una sola vez** y nunca
optimizado, mientras todo el esfuerzo iba al F1@3. Ahí estaba el margen.

**Y el ranking del concurso es relativo.** Dos tablas independientes (NDCG@10
y F1@3) combinadas por Conteo de Borda: importa la posición contra los otros
equipos, no el valor absoluto.

### 1.3 La arquitectura, en un párrafo

**Offline:** extracción por formato (`pypdfium2` + OCR para escaneos,
`mapbox-vector-tile` para los PBF) → limpieza → chunking híbrido → FAISS
`IndexFlatIP` con vectores normalizados, **un índice por encoder sobre los
mismos chunks**.

**Online:** la consulta se vectoriza con MiniLM y se recuperan 200
candidatos —con la consulta **expandida por el glosario bilingüe ES→EN**—;
**`gte-multilingual-base` y `multilingual-e5-base` los re-puntúan**, cada uno
sumando su similitud con peso **0,60** (cascada, no fusión); se agregan a
documento **sumando los 5 mejores chunks** sobre un pool de **100**; y los
fragmentos se ordenan por **idioma legible, alineación con los 3 documentos
entregados, aparato bibliográfico y cobertura léxica de la consulta**, antes de
truncar a 250 palabras.

**Sin modelos generativos en ningún punto** — lo prohíbe la sec. 8.3, y por
eso quedaron descartadas las familias Harrier y Qwen3-Embedding pese a
encabezar los rankings de 2026: son decoder-only.

---

## 1.5 La jornada del 9 de agosto, que cambió el mapa

Se midieron **ocho hipótesis más** (E13, E17, E21, E22, E23, E24, E25, y el
diagnóstico de los ceros). **Dos adoptadas, seis negativas.** Detalle en
`las notas del proyecto` y en `dev/experimentos/`.

**Lo adoptado — E22 y E23, el orden de los fragmentos.** NDCG@10 de 0,490 a
**0,506** (+0,016 [+0,004, +0,029], 11-4, IC entero sobre cero), fragmentos
ilegibles **19 → 0**, fragmentos sin una palabra de la consulta **175 → 122**.
F1@3 no se mueve, y es lo correcto: no tocan los documentos.

**Lo que cerró definitivamente tres ejes:**

- **E21 (re-chunking a 128 tokens sin solape)** refuta el último eje
  estructural sin tocar. Cierra además la ventana de MiniLM **por los dos
  lados**: ya se sabía que un primario de ventana mayor empeora, y ahora que
  ajustar el chunk a la ventana empeora más. Mata de paso la concatenación de
  vecinos enteros.
- **E17 (unión de pools)**: a profundidad 200 gte aporta **149 candidatos por
  consulta** que MiniLM no trae, y solo **8 pares** (consulta, documento
  relevante) en las 50 son exclusivos suyos. **El desacuerdo entre encoders es
  casi todo ruido.**
- **E25 (bge-m3)**: el mejor candidato del proyecto y el único con control de
  peso limpio detrás — aporta +0,019 de F1 **por encima del efecto del peso**.
  No se adopta porque las independientes no confirman y exige 14 h de GPU.

**El diagnóstico que más vale:** las **11 consultas con F1@3 = 0 son un solo
fallo, no once problemas**. Ninguna es fallo de pool — las once tienen
documentos relevantes dentro y los pierden en la agregación, seis en las
posiciones **4-8**, y el mejor chunk del perdedor **no es peor** que el del
ganador (1,55-1,69 contra mediana 1,672). Pierde por aportar 2-5 chunks donde
el ganador aporta 7.

**Y dos fallos de cumplimiento que valían más que cualquier décima:** un
archivo de consultas en **cp1252** (lo que escribe PowerShell por defecto, y lo
entrega ADL) mataba `generador.py` antes de leer una consulta; y
`resultados.jsonl` **había dejado de reproducirse byte a byte** por deriva de
versiones de las librerías. Los dos corregidos.

## 2. Lo que ya se probó y falló

**Veintiséis hipótesis medidas, tres adoptadas.** Está todo en la sección
"Medido y descartado" de `las notas del proyecto` con sus números. Resumen para no
repetirlo:

### 2.1 Reordenar lo que el pool ya trajo — agotado

| hipótesis | resultado |
|---|---|
| Híbrido BM25 + denso | pierde 15-4 (p = 0,019) |
| Fusión RRF simétrica de los dos encoders | 0,268 contra 0,306 |
| `doc_rrf` (RRF de rankings de documento) | gana en las 41, **pierde 4-1 en las independientes** |
| Invertir la cascada (e5 primario) | **0,250 / 0,267**, IC entero bajo cero |
| Agregación `max` / `mean` / `top3` | pierde o no resuelve |
| Agregación `top5` | +0,010, IC [+0,000, +0,029], pero **40 de 41 empates** |
| `rerank-depth` 400 y 600 | **51 empates de 51** — efecto exactamente cero |
| Deduplicar documentos idénticos | gana q020, pierde q032 |
| Filtrar el pool por fenómeno | gana 4, **pierde 2**; ningún umbral lo arregla |
| Corrección de *hubness* | 39-40 empates de 41 |
| Usar el ranking de fragmentos para el de documentos | 0,344 → **0,293** |
| Pool profundo (k=1500) con normalización θ | 0,300 → **0,200** en las independientes |
| Concatenar chunks vecinos (sec. 9.2.1) | **duplica texto**: el chunker solapa por diseño |

### 2.2 Diagnósticos que resultaron falsos

- **"q001 es irrecuperable"** — falso. Hay 20 documentos con CBRN y ninguno
  entra al pool. El documento existe; el recuperador no llega.
- **"la agregación premia documentos largos"** — falso. El intruso es más
  largo que el documento perdido en 127 de 235 casos: **54%, una moneda**.
- **"los fragmentos deben salir de los 3 documentos entregados"** — falso, lo
  decía un resumen de terceros; el PDF oficial no lo exige.
- **"la truncación a 128 tokens de MiniLM lo hace peor recuperador"** — el
  96% de los chunks supera esa ventana, pero al usar e5 (ventana 512) como
  primario el resultado **empeora**. La truncación es real y no importa.

### 2.3 Lo que sí se arregló

- **Guiones de fin de línea (U+FFFE)**: partían palabras en el 30% de los
  chunks y en 166 de los 500 fragmentos entregados. Reparado con un mapa de
  desempate construido del propio corpus. **No mejoró el F1@3** (3-3-35) y se
  hizo igual: es un defecto del dato.
- **Un checkpoint que habría corrompido el índice en silencio**: validaba
  cantidad y `chunk_id` pero no el texto. Tras 7 h de CPU habría entregado
  vectores del texto roto.
- **La GUI corría las 50 consultas inventadas** en vez de las oficiales.

---

## 3. El problema de fondo: no es el recuperador, es el instrumento

Esto es lo más importante del documento.

Un análisis de poder estadístico sobre el propio diseño mostró que **con 41
consultas y F1@3 no se puede detectar casi nada**:

- El F1@3 se mueve en escalones de **0,315** por consulta. No hay valores
  intermedios.
- La prueba de signos necesita **≥6 consultas discordantes** para bajar de
  p=0,05. La cascada dio 5-0 con p=0,062: **estaba fuera del alcance del test
  antes de correrla**.
- El efecto mínimo detectable con potencia 0,80 es **0,059** en el mejor caso,
  e **inalcanzable a cualquier tamaño** con un cambio realista.
- **Trece "no concluyente" seguidos con potencia ~0,3 es exactamente lo
  esperable aunque las trece fueran mejoras reales.**
- Para detectar ΔF1 = 0,03 harían falta **n = 140 a 455 consultas**, no 50.

Y la regla de decisión que usábamos ("no adoptar nada que pierda consultas")
estaba **anti-correlacionada con la calidad**: adoptaba una mejora de +0,016
con probabilidad 0,275 y una de +0,126 con 0,073. Filtraba por cuán poco
tocaba el sistema.

### 3.1 Lo que se cambió

`eval_mini.py` ahora reporta el **delta pareado con IC al 90% por bootstrap**,
y adopta con **umbral asimétrico**: en un torneo el coste de adoptar una
mejora nula es ~0 y el de rechazar una real es perder posiciones, así que el
criterio es *descartar solo lo que probablemente daña*, no *probar que
mejora*.

### 3.2 Lo que NO se pudo arreglar

- **Llegar a 50 consultas no sirve**: el MDE pasa de 0,059 a 0,049.
- **Decidir con NDCG@10** multiplica por 6-9 el n efectivo… **pero solo para
  cambios que tocan fragmentos**. Para cambios de agregación a documento el
  Δ de NDCG es **exactamente 0,000**, porque la lista de fragmentos no depende
  de la estrategia de agregación.
- **Cribar con anotadores-agente: medido y descartado.** El acuerdo
  agente-vs-humano sobre 12 consultas ya anotadas es **F1 = 0,28** (precisión
  0,30, recall 0,26). De cada 10 documentos que marca el agente, el humano
  marcó 3. Es casi el mismo nivel que el recuperador que queríamos evaluar:
  usarlo como ground truth mediría si el sistema se parece a otro modelo.
  *(Salvedad: un solo anotador, el más estricto, sobre 12 consultas; los otros
  dos cayeron por límite de sesión. El consenso de tres podría ser mejor y
  vale re-medirlo antes de cerrar el camino del todo.)*

---

## 4. Lo que queda por hacer

### 4.1 Prioridad 1 — Ampliar el ground truth a mano

Es **la única palanca que no está bloqueada**. Faltan 9 consultas (q001, q007,
q008, q011, q012, q015, q028, q038, q048) y, sobre todo, hace falta
**profundidad**: las 5 consultas con un solo documento relevante tienen
escalón 0,50 y son las que más ruido inyectan; llevarlas a ~5 relevantes baja
su escalón a 0,25 sin agregar consultas.

`dev/eval/candidatos.md` tiene las casillas listas.
`anotar_candidatos.py --recolectar` las funde.

**Advertencia sobre q001 y q038:** no anotarlas con ceros. Sus documentos
relevantes existen pero el pool de 12 candidatos no los contiene, así que
anotarlas con lo que hay mete una afirmación falsa. Para q001 hay que sacar
candidatos de los 20 documentos con CBRN; para q038, del subcorpus ALERTAS.

### 4.2 Prioridad 2 — NDCG@10, la mitad de la nota sin explorar

Nunca se optimizó nada a nivel fragmento. Hace falta:

1. **Ground truth de fragmentos**, aunque sea de 10 consultas, para saber
   cuánto miente la aproximación actual.
2. Implementar NDCG con **relevancia graduada**, que es lo que pide la
   sec. 10.2.1, en vez de binaria.
3. Probar ideas de membresía (no de orden — reordenar ya se midió: 15-15).

### 4.3 ~~Prioridad 3 — La construcción del pool~~ — CERRADA (E18, 9 ago)

**Este eje está muerto y no hay que reabrirlo.** El dato que lo motivaba era
que el pool de 60 alcanzaba solo el 52% de los documentos relevantes. Con la
configuración actual (pool 100 sobre 200 candidatos re-puntuados) **E18 midió
que el pool ya trae el 93,2%**, y el diagnóstico de las 11 consultas con
F1@3 = 0 lo remata: **ninguna es fallo de pool** — las once tienen documentos
relevantes dentro y los pierden en la agregación.

**E17 lo confirma desde el otro lado:** unir el pool de MiniLM con el de gte
mete 149 candidatos nuevos por consulta y solo **8 pares** (consulta,
documento) relevantes en las 50 son exclusivos de gte. Ampliar el pool ya no
compra recall, solo dilución.

Lo que queda de esta sección es histórico:

- ~~**Pseudo-relevance feedback denso**~~ — **medido el 2 ago, no acumulable.**
  PRF sobre MiniLM solo sube de 0,300 a **0,333** en las independientes
  (gana 1, pierde 0) — pero eso es **exactamente lo que ya da la cascada**, y
  PRF *sobre* la cascada **pierde 3-0** (0,333 → 0,233). Son dos vías al mismo
  techo. Se conserva la medición como alternativa **si alguna vez hay que
  renunciar al segundo encoder**: PRF consigue lo mismo sin construir ni
  publicar un índice de 395 MB.
- **Desempate intra-colección con el grafo**, que está construido y sin usar.
  El dato que lo motiva: **acertamos la colección el 74% de las veces y el
  documento solo el 33%**. Lo que falta es elegir entre hermanos de la misma
  serie, no encontrar el tema.
- **Prior de recencia**: 508 documentos llevan año en el nombre y hay 7
  consultas con marca temporal ("recientemente", "actual"). En q029 el ground
  truth es 2021-2026 y entregamos 2019, 2019, 2025.

### 4.4 Prioridad 4 — Un encoder nuevo

Solo si sobra tiempo de CPU. Detalle en `plan_encoders.md`. El candidato es
`Alibaba-NLP/gte-multilingual-base` (Apache 2.0, 305M, ventana 8192, sin
prefijos, ~6 h de CPU). **Descartada la familia Qwen3-Embedding**: backbone
decoder, o sea riesgo bajo la sec. 8.3, y de todas formas infactible en CPU.

**Para la cascada el par debe ser diverso**: dos modelos de la misma familia
se equivocan igual.

---

## 5. Lo que no se puede romper

1. **`Entrega/generador.py` es autocontenido.** No importa nada de `dev/src/`.
   Al tocar un módulo del camino online hay que re-aplanar el cambio ahí y
   repetir la corrida en frío.
2. **El chunking se hace UNA vez**, con el tokenizer del primer encoder. Si se
   re-fragmenta, los `chunk_id` cambian y la cascada deja de poder leer el
   vector del secundario por posición de fila.
3. **`Entrega/` nunca lleva archivos comprimidos.** La sec. 1.4 exige
   `index.faiss` cargable sin dependencias y `metadata.jsonl` en JSON Lines.
   Los `.gz` son solo para los assets del Release.
4. **Los binarios van por GitHub Release, no por LFS.** La cuota gratuita es
   1 GB y no se puede liberar.
5. **`doc_id` es la clave de emparejamiento**, no `fuente`. Y el manifest se
   indexa por **ruta relativa**, no por nombre de archivo.
6. **Sin modelos generativos en ningún punto** del pipeline entregado.

---

## 6. Riesgos abiertos

| riesgo | mitigación |
|---|---|
| El ground truth propio no representa el de ADL | Es el riesgo de fondo y no tiene mitigación real. Todo lo medido puede no transferir. |
| Veintitrés hipótesis fallidas ⇒ tentación de adoptar por promedio | La regla y su corrección están escritas en `lecciones_metodologia.md`. Leerlo antes de proponer. |
| Cuota de LFS ya consumida (~1 GB) | Los índices nuevos van por Release. No commitear binarios. |
| Corrida en frío rota por un cambio en `dev/src/` | `tests/test_retrieval_schema.py` lo vigila, pero la verificación final es manual. |

---

## 7. Si tuviera que apostar

La entrega actual es **sólida y defendible**: reproducible, sin defectos de
formato, con el texto limpio, dos encoders justificados por medición y un
grafo de bonus. Eso ya la pone por encima de cualquier equipo que entregue el
top-10 crudo de un encoder sin verificar.

**El margen que queda no está en el recuperador.** Veintiséis intentos lo
dicen, y el 9 de agosto lo confirmó por partida triple: E18 probó que el pool
ya trae el 93% de lo relevante, E20 que la agregación agotó sus regímenes y
E21 que el chunking tampoco es la palanca. Los veintiséis operan sobre **un
solo canal de evidencia: el score de embedding de los chunks**, y ese canal
está cerrado por los tres lados.

**Lo que sí se movió fue NDCG@10, y por dónde importa.** Las dos adopciones del
9 de agosto no cambian qué se recupera: cambian **qué texto se entrega**, que
es lo que la sec. 10.2.1 dice que se juzga. Fragmentos que el evaluador no
puede leer: cero. Fragmentos que no mencionan el objeto de la consulta: un
tercio menos. Ese es el patrón de lo que funciona en este proyecto — **arreglar
defectos del dato, no calibrar parámetros.**

Si solo hubiera tiempo para una cosa: **anotar más y más profundo, a mano.**
Sigue siendo cierto, y ahora con número: las 9 consultas de etiqueta de agente
dan **0,311** contra **0,468** de las humanas. Mientras eso siga así, ninguna
medición nuestra —ni el 0,506 ni el 0,460 de bge-m3— dice de verdad dónde
estamos frente al ground truth de ADL.

## 8. Decisiones abiertas, para quien retome

1. **E15 (desempate estable por `chunk_id`).** Da inmunidad permanente a la
   deriva de versiones, que el 9 de agosto rompió la reproducibilidad sin que
   nadie tocara el código. E15 lo midió y pierde — pero su propio cierre dice
   que lo que pierde es *una moneda al aire entre copias duplicadas*. **Hoy
   pesa distinto que cuando se midió: la máquina de ADL no es esta.**
2. **E25 (bge-m3).** El índice completo se construyó para poder decidir con el
   sistema entero en la mano. Gana en las 50 y en las humanas, no confirma en
   las independientes.
3. **E24 condicional.** El nombre del documento como desempate **solo cuando el
   ranking agregado es plano**. El uso incondicional está refutado; el
   condicional no se midió.
