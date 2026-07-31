# Explicación reto final — Sesión de entrenamiento Codefest (Q&A ADL)

Este documento recoge la transcripción y el resumen de la sesión final de
entrenamiento del Codefest AD ASTRA 2026 (Q&A técnica con el equipo ADL y el
profesor Rubén), tal como fue compartida por el equipo, seguida de un repaso
de qué tanto cubre este proyecto lo que ahí se aclaró.

## 1. Transcripción de la Q&A (minuto ~37 en adelante)

**Pregunta — Doc ID / campo `fuente`:** En el array del documento que no
incluye la fuente, los evaluadores van a cruzar nuestro `doc_id` contra
nuestro propio `metadata.jsonl` para obtener la fuente. Sería bueno
confirmar esto explícitamente porque de eso depende que la trazabilidad sea
perfectamente consistente entre nuestros archivos.

**Respuesta:** La última frase del cuadro de la sección 10.2 corresponde a
un error de versionamiento del documento y ya se está corrigiendo. **El
emparejamiento a nivel de documento se realiza mediante el campo `doc_id`,
el cual será suministrado por ADL junto con los datos.**

---

En el siguiente slide se agruparon varias preguntas con respuestas en común:

**Preguntas agrupadas:**
1. Solicitan que se envíe el corpus para poder iniciar con lo propuesto, que
   hasta el momento no se ha recibido dentro del material.
2. No se especifica cuándo se publica el archivo de las 50 consultas, y esto
   es necesario para el cronograma de entrega.
3. Cuando se indica el tamaño aproximado del corpus, esto cambia decisiones
   reales: tipo de índice FAISS (plano vs. aproximado), tamaño razonable de
   la muestra piloto y tiempos de cómputo esperados.

**Pregunta:** ¿Cuándo se entregarán las fuentes documentales por parte de
ADL para construir la base de conocimiento, y cuándo se entregarán las
consultas para evaluar el sistema de recuperación?

**Respuesta:** Steven (ADL) contará más sobre el corpus y cuándo se va a
entregar. **Las 50 consultas no se van a entregar** porque son el conjunto
de pruebas (ciego); **el corpus sí se entregará** cuando se vaya a usar para
construir la base de conocimiento.

---

**Pregunta — Entorno de reproducción de `generador.py`:** Dice que si no es
reproducible se excluye de la evaluación, pero no aclara en qué entorno lo
van a correr. ¿Debemos entregar un archivo describiendo el entorno? ¿Cuáles
son las características de hardware mínimas de los computadores de los
evaluadores?

**Respuesta:** La evaluación se realizará con **Python ≥ 3.9.5**
(recomendado para FAISS). Se dispone tanto de GPUs **RTX 4090** como de un
servidor **GPU A40**; sin embargo, una vez construido el índice, el volumen
esperado de vectores no requiere GPU para la evaluación — **la ejecución en
CPU es suficiente**. La entrega debe incluir explícitamente el índice y la
metadata: **no se replicará el proceso de generación de vectores**, sino
únicamente la generación de resultados en formato JSON a partir del índice
ya construido. **No es necesario** un archivo de texto adicional describiendo
el entorno, siempre que se siga la convención estándar de `sentence-transformers`
y modelos abiertos de HuggingFace.

---

**Pregunta — Formato PBF:** Por el contexto parece referirse a OSM PBF.
¿Pueden confirmar qué es y si recomiendan alguna librería específica?

**Respuesta:** OSM = OpenStreetMap, PBF = Protocol Buffer Binary Format (un
formato alternativo al XML para datos geográficos). Librerías recomendadas:
**`pyosmium`** y **`osmnx`** (mencionado como "Pesonim" en la transcripción
automática — muy probablemente `osmnx`).

---

**Aclaración final (Sara):** Las preguntas que se comparten a los
participantes **no incluyen el ground truth**: no se comparten los
documentos ni los chunks relevantes esperados, solo el texto de las
preguntas. ADL guarda esos resultados esperados para comparar contra lo que
cada equipo genere y así construir el ranking.

---

Cierre con frases motivacionales sobre trabajo en equipo ("Ningún jugador es
tan bueno como todos juntos" — Di Stéfano; "Si quieres ir rápido, ve solo.
Si quieres llegar lejos, ve acompañado" — refrán africano; "Sic Itur Ad
Astra").

## 2. Resumen de la sesión

La sesión final del entrenamiento Codefest abordó la construcción y
evaluación de un índice vectorial para recuperación de información en el
campo aeroespacial con énfasis en Latinoamérica. La capitán Sara Vigoya
inició la presentación, seguida por el profesor Rubén, quien explicó
detalladamente las especificaciones técnicas del reto principal: construir
un índice vectorial a partir de diversos documentos, lo que incluye
preprocesamiento, división en fragmentos (chunks), codificación semántica
con encoders y construcción de embeddings para optimizar la recuperación.
Se destacó la opción de un bono adicional: construir un grafo de
conocimiento para mejorar la precisión de las respuestas.

El reto tiene como objetivo desarrollar soluciones capaces de recuperar
documentos y fragmentos relevantes a partir de 50 consultas específicas,
retornando resultados en JSON evaluados con métricas combinadas: F1-Score
para documentos y DCG (NDCG@10) para fragmentos. La entrega debe incluir
además un informe técnico explicando las decisiones de diseño, el índice
vectorial, el generador en Python y, opcionalmente, el grafo de
conocimiento (bono).

Steven (ADL) presentó la estructura y variedad del corpus documental: unos
**1826 archivos** de diversos formatos (PDF, JSON, Excel, imágenes, PBF).
También se detalló el proceso de comunicación con los participantes, fechas
clave y aclaraciones sobre formatos, evaluación, entorno técnico y buenas
prácticas para el desarrollo de sistemas multiagente (Etapa 2).

### Highlights

- 🚀 Explicación detallada del reto: creación de un índice vectorial para recuperación documental.
- 📑 Uso de chunking, codificación semántica y selección estratégica de encoders.
- 🌐 Arquitectura multiagente orientada a análisis global, regional y nacional (Etapa 2).
- 💾 Corpus documental diverso: ~1826 archivos en múltiples formatos.
- 🏆 Evaluación con métricas combinadas de precisión en documentos y fragmentos.
- 🧩 Bonus: grafo de conocimiento para mejorar la recuperación.
- 🤝 Importancia del trabajo en equipo y la documentación técnica.

### Insights clave

- **Índice vectorial como base de recuperación eficiente**: preprocesamiento + chunking + embeddings, con impacto directo del encoder elegido.
- **Evaluación dual (documentos y fragmentos)** mejora la granularidad de la calidad de recuperación.
- **Arquitectura multiagente jerárquica vs. plana** (Etapa 2): empezar simple, escalar solo si es necesario.
- **Metadata y formatos variados**: crítico para trazabilidad y consistencia.
- **Contexto latinoamericano**: relevancia estratégica global/regional/nacional.
- **Replicabilidad y documentación**: `generador.py` + informe técnico validan metodología y transparencia.
- **Buenas prácticas multiagente**: roles claros, canales únicos de comunicación, frameworks declarativos.

### Tabla de fechas clave

| Evento | Fecha |
|---|---|
| Envío de resultados para evaluación | No especificada explícitamente (antes del 20 de agosto) |
| Informe a equipos finalistas | 20 de agosto |
| Fase final presencial (Bogotá) | 18 y 19 de septiembre |

### Detalles técnicos clave

- Formato de salida: JSON con ranking para documentos y chunks de ≤250 palabras.
- Métricas: F1@3 (top 3 documentos) y NDCG@10 ponderado (fragmentos).
- Entorno de ejecución: Python ≥ 3.9.5; GPUs disponibles pero CPU suficiente tras construir el índice.
- Bonus grafo de conocimiento: `grafo.graphml`.

### Consejos para participantes

- Empezar con soluciones simples, evolucionar a multiagente solo cuando sea necesario.
- Documentar cada decisión técnica, metodología y librerías usadas.
- Seguir rigurosamente la estructura de carpetas y formato requerido para evitar penalizaciones.
- Usar frameworks fáciles de mantener para el desarrollo multiagente (Etapa 2).
- Aprovechar el material adicional y notas de clase de los organizadores.

## 3. Resumen adicional de reglas (segunda fuente, NoteGPT)

Complemento al resumen de la sección 2, con foco en reglas y puntos que hay
que respetar sí o sí.

### 3.1 Descripción general del reto

- Objetivo: construir un índice vectorial para recuperación de información
  en un corpus documental amplio.
- Fases: preprocesamiento multiformato → chunking (máx. 250 palabras) →
  codificación semántica con encoders → construcción del índice vectorial →
  flujo de recuperación por consulta.

### 3.2 Pasos clave

- **Chunking**: bajo una estrategia definida, límite de 250 palabras; los
  chunks deben coincidir con la indexación y tener IDs únicos relacionados
  con el índice.
- **Encoders**: selección justificada según las características de los
  documentos; se puede usar uno o varios (estrategia híbrida), y si hay
  varios, cada uno va en su propia carpeta con sus propios archivos.
- **Índice vectorial**: FAISS gestiona vectores + metadata asociada; **no se
  almacena el texto completo dentro del índice**, sino que la metadata
  permite recuperar los documentos; debe permitir consultas rápidas con
  resultados rankeados.

### 3.3 Entregables y estructura (reafirma la Sección 1.4 del PDF)

- `resultados.jsonl` (50 consultas, JSON Lines).
- `generador.py`: **con comentarios claros**, reproduce los resultados a
  partir del índice.
- Informe técnico en PDF: modelo(s) de encoder, métricas, librerías,
  estrategias.
- Carpeta `base_vectorial/`: archivos de vectores + `metadata.jsonl`.
- Bonus: grafo de conocimiento en carpeta `grafo/` con `grafo.graphml`.
- **No cumplir la estructura de carpetas exigida conlleva penalización
  severa o exclusión de la evaluación** — punto remarcado explícitamente.

### 3.4 Formato y evaluación

- Por consulta: top 3 documentos rankeados (por `doc_id`) + fragmentos
  rankeados dentro de esos documentos.
- Métricas: F1-score en top 3 (documentos) + DCG ponderado (fragmentos,
  NDCG@10).
- Puntuación final combinada mediante **Conteo de Borda** entre ambas
  métricas.

### 3.5 Buenas prácticas remarcadas

- Roles acotados por agente si se usa un sistema multiagente (Etapa 2);
  instrucciones claras y límites definidos; empezar simple y escalar solo
  si hace falta; usar frameworks trazables que faciliten depuración.
- Documentar TODO: decisiones de encoders, parámetros, librerías,
  estrategias de ranking. Código con comentarios claros. Informe técnico
  debe permitir replicar los resultados.
- **Usar la extensión real del archivo en minúsculas en la metadata** (no
  limitarse solo a los ejemplos ilustrativos `pdf`/`html`/`md` del PDF —
  reportar también `json`, `csv`, `xlsx`, `png`/`jpg`, `pbf`, etc. tal cual
  vienen).
- Formatos especiales como OSM PBF requieren librerías dedicadas
  (`pyosmium`).

### 3.6 Aspectos técnicos de ejecución (reafirma la charla de la sección 1)

- Evaluación con Python ≥ 3.9.5.
- GPUs disponibles (RTX 4090 / A40) pero **no obligatorias**: la evaluación
  se puede hacer solo con CPU.
- No se espera que se replique el proceso de vectorización del corpus, solo
  la generación del `resultados.jsonl` a partir del índice ya entregado.

## 4. Preguntas de profundización y respuestas (NoteGPT)

Preguntas de seguimiento generadas a partir del material del reto, con sus
respuestas, agrupadas por ronda. Sirven como guía de diseño, no como
requisitos obligatorios adicionales.

### Ronda 1

**1. ¿Ventajas y desventajas de una estrategia híbrida con múltiples encoders?**

*Ventajas:* mejor adaptabilidad a la diversidad del corpus (se puede elegir
el encoder más adecuado por tipo de documento), rankings potencialmente más
robustos al combinar espacios semánticos distintos, flexibilidad de diseño
(encoders complementarios para textos largos vs. cortos).

*Desventajas:* más complejidad de gestión (varios índices y metadatas que
mantener sincronizados), mayor costo computacional, y la obligación
estructural de que cada encoder tenga su propia carpeta/`metadata.jsonl`
bien sincronizada (riesgo de penalización si no cuadra).

**2. ¿Cómo influye la segmentación en chunks en la precisión y eficiencia?**

Un chunk bien delimitado (≤250 palabras, unidad semántica coherente) mejora
directamente el NDCG al facilitar que el fragmento devuelto sea realmente
relevante, y acelera la búsqueda al indexar fragmentos en vez de documentos
completos. Una segmentación mala (chunks demasiado pequeños o mal cortados)
pierde contexto, genera redundancia o resultados confusos, y puede romper
la sincronización entre `chunk_id` e índice.

**3. ¿Qué considerar para integrar un grafo de conocimiento?**

Es un complemento semántico explícito (relaciones entre entidades) al
índice vectorial, no obligatorio pero valorado. Debe entregarse en
`grafo/grafo.graphml`; el `generador.py` debe contemplar la lógica de
consultar el grafo además del índice; hay que documentar en el informe
técnico toda la lógica, librerías y algoritmos usados; y mantener
trazabilidad clara entre nodos del grafo, documentos e índice.

### Ronda 2

**1. ¿Cómo optimizar la gestión/sincronización de múltiples encoders?**

Carpetas bien organizadas por encoder, IDs consistentes de documento/chunk
para trazabilidad exacta, pipelines automatizados de indexación (reducir
intervención manual), documentación detallada de parámetros por encoder,
validación intermedia periódica (que cada vector tenga su metadata), y usar
solo los encoders que aporten valor diferencial real (no sumar por sumar).

**2. ¿Técnicas adicionales para mejorar la segmentación de chunks?**

Segmentar por unidades semánticas (oraciones/párrafos completos, no solo
conteo fijo de palabras), superposición controlada entre chunks
consecutivos (overlap) para no perder contexto en la frontera, tamaño
adaptable según la estructura del documento, limpieza previa de ruido
(boilerplate, etiquetas), y evitar cortar referencias/citas relevantes.

**3. ¿Cómo el grafo enriquece la interpretación semántica de las consultas?**

Representa relaciones explícitas (nodos/aristas) que el embedding solo
captura de forma implícita; permite refinar la lista de candidatos
combinando el resultado vectorial con rutas/relaciones del grafo; ayuda en
consultas con múltiples entidades; integra fuentes heterogéneas en una red
coherente; y es actualizable de forma incremental sin rehacer todo.

### Ronda 3

**1. ¿Cómo validar automáticamente inconsistencias entre resultados de distintos encoders?**

Comparación cruzada de rankings entre encoders para la misma consulta,
métricas homogéneas de comparación (F1@3/NDCG@10) para detectar outliers,
normalización previa de vectores/metadata, umbrales de tolerancia para
diferencias aceptables, reportes automáticos de discrepancias, y validación
de que los IDs de documento/chunk coincidan entre todos los índices —
integrado como paso del pipeline, no como revisión manual aparte.

**2. ¿Cómo adaptar dinámicamente tamaño/overlap de chunks según el tipo de texto?**

Clasificar el documento primero (técnico vs. narrativo); en técnicos usar
encabezados/secciones como delimitadores y overlap mayor (definiciones que
se repiten); en narrativos segmentar por oración/párrafo con menos overlap;
ajustar el tamaño máximo según densidad informativa; y evaluar
iterativamente con las métricas oficiales para calibrar la configuración.

**3. ¿Cómo actualizar un grafo de conocimiento de forma incremental?**

Diseño modular (nodos/aristas independientes que se pueden agregar o quitar
sin tocar el resto), inserciones/actualizaciones localizadas al detectar
información nueva, scripts de actualización que solo procesen las
novedades del corpus, versionado del grafo para poder revertir cambios
erróneos, y sincronización de esas actualizaciones con el índice vectorial
para no perder consistencia.

### Ronda 4 (preguntas abiertas, sin respuesta desarrollada en la fuente)

1. ¿Cuáles son los principales desafíos al normalizar/comparar automáticamente
   encoders con arquitecturas y salidas muy distintas?
2. ¿Cómo influye el dominio o idioma del texto en los parámetros de
   segmentación dinámica, y qué técnicas adicionales ayudarían a esa
   adaptación?
3. Con múltiples fuentes que cambian constantemente, ¿qué estrategias
   garantizan que la actualización incremental del grafo mantenga
   integridad semántica, escalabilidad y rendimiento?

## 5. Estado del proyecto frente a esta charla

| Punto de la charla | Estado en el repo |
|---|---|
| Índice FAISS + metadata + `generador.py` + informe técnico | ✅ Ya existen en `Entrega/` |
| Grafo de conocimiento (bono) | ✅ Implementado (`src/graph/`, `Entrega/base_vectorial/grafo/grafo.graphml`) |
| Salida JSON con ranking, fragmentos ≤250 palabras | ✅ Implementado y validado por `tests/test_retrieval_schema.py` |
| Python ≥ 3.9.5, ejecución en CPU suficiente | ✅ El proyecto ya corre 100% en CPU (`faiss-cpu`, `torch` CPU) |
| No hace falta archivo de texto describiendo el entorno | ℹ️ No aplica ninguna acción — ya no se entrega nada extra |
| Las 50 consultas oficiales nunca traen ground truth | ℹ️ Ya está claro en el repo (`consultas_prueba/` es explícitamente provisional, ver `README.md` sección 6) |
| Formatos JSON/CSV/XLSX del corpus real | ✅ Cubiertos (`src/extraction/json_extractor.py`, `tabular_extractor.py` con `pd.read_excel` + `openpyxl` en `requirements.txt`) |
| **Emparejamiento por `doc_id` suministrado por ADL** (no autogenerado) | ✅ **Resuelto.** `src/ingestion/doc_id.py::resolve_doc_id()` usa el `doc_id` de ADL cuando hay manifest y cae al hash de contenido cuando no. El manifest se pasa con `--doc-id-manifest` a `scripts/build_corpus_index.py` y acepta JSON (objeto plano o lista), JSONL y CSV, porque todavía no se sabe en qué formato lo entregará ADL. Si llega en otro formato, se adapta solo `load_doc_id_manifest()`. Verificado end-to-end: el `doc_id` de ADL propaga a `doc_id` y `chunk_id` en `metadata.jsonl`. |
| Extracción de PBF (OSM) | ❌ Sigue sin implementar (`src/extraction/pbf_extractor.py` lanza `NotImplementedError` a propósito). La charla confirma que es OSM PBF y recomienda `pyosmium`/`osmnx` — implementar en cuanto se confirme que el corpus real trae archivos `.pbf` y se pueda probar contra datos reales. |
| Tamaño real del corpus (~1826 archivos) | ⚠️ Aún no probado a esa escala — solo se ha corrido contra el corpus sintético (15 docs) y el corpus real de ejemplo (~30 PDFs). Con el volumen real hay que revalidar tiempos de indexación y si `IndexFlatIP` (el que se usa hoy) sigue siendo razonable o conviene pasar a `IndexIVFFlat`/`IndexHNSW` (sec. 5.2 de la especificación). |
| Corpus y las 50 consultas aún no entregados por ADL | ℹ️ Confirmado por la charla — no es una omisión nuestra, es un insumo externo pendiente. Nada que hacer hasta que llegue. |
| Usar la extensión real del archivo (minúsculas) en `formato`, no solo pdf/html/md | ✅ Ya se hace así: `html`, `json`, `pdf`, `csv`/`xlsx`, `img` (ver `src/extraction/*_extractor.py`) |
| Múltiples encoders, cada uno con su carpeta y su `metadata.jsonl` (sec. 4.4) | ✅ Soportado: `--encoder-name A B` crea `encoder_A/` y `encoder_B/`, y los rankings se fusionan con RRF. Detrás de flag hasta poder medir su impacto. |

### Insumo pendiente de ADL: el corpus y las 50 consultas (al 28 de julio de 2026)

A esta fecha **ADL todavía no ha entregado el corpus oficial (~1826 archivos),
los `doc_id` asociados ni las 50 consultas q001–q050**. Es un insumo externo:
no hay forma de sustituirlo desde el repo. Lo que hay hoy es corpus sintético
(`dev/corpus_ejemplo/`, 15 documentos escritos a mano) y consultas inventadas
(`dev/consultas_prueba/`), útiles solo para probar la mecánica.

Lo que queda bloqueado hasta que llegue:

- **Emparejamiento por `doc_id` real.** El código ya acepta el manifest de ADL
  (`--doc-id-manifest`, JSON/JSONL/CSV) y cae a hash de contenido si no hay.
  Sin el manifest real, el F1@3 contra el ground truth daría cero.
- **Calibrar el chunking** sobre documentos reales (PDFs de SIPRI/NASA/Banco
  Mundial), que pegan directo al NDCG@10.
- **Decidir con datos si se entrega con uno o dos encoders**
  (`scripts/compare_encoders.py` necesita corpus real para ser informativo).
- **Implementar `pbf_extractor.py`**: no se sabe aún si el corpus trae `.pbf`.
- **Generar el `resultados.jsonl` definitivo** con las 50 consultas oficiales.

Lo que **no** está bloqueado y ya se resolvió: el pipeline completo end-to-end,
el esquema de entrega validado, el grafo bonus, el multi-encoder con fusión RRF
y el dimensionamiento del índice (medido con vectores sintéticos: 50k vectores
→ 5.7 ms p50 por consulta, `IndexFlatIP` sobra a la escala esperada).

### Lo que hay que mejorar / vigilar de aquí a que llegue el corpus real

Ordenado por impacto sobre las métricas (NDCG@10 + F1@3), que es lo único
que evalúa la Etapa 1:

1. ~~**Ajustar `doc_id`** para aceptar el que entregue ADL~~ — ✅ hecho, ver la tabla de arriba.
2. ~~**Multi-encoder**~~ — ✅ hecho. `scripts/build_corpus_index.py` y `Entrega/generador.py` aceptan varios `--encoder-name`; los rankings se fusionan con RRF (sec. 8.4) junto con el grafo (sec. 8.5). Segundo encoder: `intfloat/multilingual-e5-base` (MIT, 768 dim), elegido por diversidad de objetivo de entrenamiento. Queda **detrás de flag**, no activo por defecto, hasta poder medir si mejora. `scripts/compare_encoders.py` reporta cuánto se solapan ambos rankings para decidirlo con datos.
3. **Revalidar el tipo de índice FAISS** con el volumen real de ~1826 documentos (probablemente sigue siendo manejable con `IndexFlatIP`, que además garantiza resultados exactos, pero hay que medirlo, no asumirlo).
4. **Calibrar el chunking** con documentos reales: los PDFs de SIPRI/NASA/Banco Mundial tienen estructura muy distinta al corpus sintético, y el tamaño de chunk y el overlap pegan directo al NDCG.
5. **Ampliar el informe técnico**: va en 3 páginas de las 8 permitidas, y ahí es exactamente donde se evalúan las justificaciones de diseño.
6. **Implementar `pbf_extractor.py`** con `pyosmium` cuando se confirme que hay archivos PBF en el corpus real.
7. **Correr las 50 consultas oficiales** en cuanto ADL las publique y regenerar `Entrega/resultados.jsonl` (los pasos ya existen, ver `README.md` sección 4, paso 4).
8. Nada que hacer respecto al entorno de reproducción — ya se confirmó que solo hace falta el índice + metadata + `generador.py`, sin archivo de entorno adicional.

## 6. Flujo del proyecto, explicado

El sistema tiene **dos fases** completamente separadas:

**Fase OFFLINE — "compilar la base de conocimiento" (se corre una vez por corpus):**

```
Documentos de ADL (PDF, HTML, JSON, CSV, XLSX, imágenes, PBF)
   │
   ├─ 1. Extracción de texto  (un extractor por formato, src/extraction/)
   ├─ 2. Limpieza + detección de idioma  (src/cleaning/)
   ├─ 3. Chunking (fragmentación en trozos con oraciones completas)  (src/chunking/)
   ├─ 4. Embeddings (un vector por fragmento, con el encoder elegido)  (src/embedding/)
   ├─ 5. Índice FAISS + metadata.jsonl  (src/embedding/build_index.py)
   └─ 6. (Bono) Grafo de conocimiento: entidades + relaciones  (src/graph/)
```

Esto se corre una sola vez (o cada vez que cambie el corpus). El resultado
queda guardado en disco en `Entrega/base_vectorial/`.

**Fase ONLINE — "responder consultas" (se corre por cada tanda de preguntas):**

```
Consulta en lenguaje natural
   │
   ├─ 1. Se vectoriza con el MISMO encoder usado en la indexación
   ├─ 2. Búsqueda en FAISS → top-k fragmentos más similares
   ├─ 3. Agregación de fragmentos a nivel de documento (top 3 docs)
   ├─ 4. (Si aplica) fusión con el grafo de conocimiento
   ├─ 5. Recorte de cada fragmento a ≤250 palabras
   └─ 6. Escribe Entrega/resultados.jsonl (3 docs + 10 fragmentos por consulta)
```

Esta fase es rápida porque no vuelve a calcular embeddings del corpus — solo
busca en el índice ya construido. `Entrega/generador.py` es el script que
ejecuta exactamente esta fase y es lo que los evaluadores van a correr para
verificar reproducibilidad (por eso no hace falta re-generar el índice: eso
ya está hecho y entregado).

En ningún punto de ninguna de las dos fases se usa un modelo generativo — es
retrieval puro sobre vectores, FAISS y metadata (restricción explícita de la
especificación, sec. 8.3).

## 7. Propuesta de reparto para un equipo de 4 personas

La propuesta sigue la separación por módulos que ya existe en `src/`, para
que cada persona tenga una zona del código bien delimitada y no haya
choques constantes de merge:

**Persona 1 — Extracción e ingesta** (`src/extraction/`, `src/cleaning/`, `src/ingestion/`)
- Mantener y ampliar los extractores por formato.
- Implementar `pbf_extractor.py` con `pyosmium` cuando llegue el corpus real.
- Ajustar `doc_id.py`/`derive_fuente()` para usar el `doc_id` que entregue ADL en vez del hash propio.
- Validar que la limpieza y detección de idioma funcionen bien sobre el corpus real (boilerplate distinto al sintético).

**Persona 2 — Chunking, encoders e índice** (`src/chunking/`, `src/embedding/`)
- Afinar la estrategia de chunking híbrida con texto real (los PDFs reales pueden tener estructura muy distinta a la sintética).
- Correr `scripts/build_corpus_index.py` contra el corpus real y medir tiempos/memoria con ~1826 documentos.
- Decidir si el índice plano (`IndexFlatIP`) sigue siendo suficiente a esa escala o si conviene `IndexIVFFlat`/`IndexHNSW` (sec. 5.2 de la especificación).

**Persona 3 — Retrieval y grafo de conocimiento** (`src/retrieval/`, `src/graph/`)
- Búsqueda, agregación a documento, fusión de rankings (RRF/CombSUM/CombMNZ).
- Mantenimiento del grafo bonus: NER, relaciones, integración con la recuperación vectorial.
- Ajustar post-filtros (por fenómeno, idioma, umbral de similitud) si se necesitan.

**Persona 4 — Entrega, QA y reporte** (`Entrega/`, `tests/`, informe técnico)
- Dueño de `Entrega/generador.py` y de que `resultados.jsonl` cumpla el esquema exacto (correr el validador de la sección 4 del README cada vez que se regenera).
- Mantener y ampliar la suite de `pytest` a medida que el resto del equipo cambia código.
- Redactar y actualizar `informe_tecnico.pdf` con las decisiones de diseño de los otros 3.
- Correr las 50 consultas oficiales apenas ADL las publique y coordinar la entrega final (checklist de la sección 7 del README).

La GUI (`src/gui/`) y cualquier arquitectura multiagente son parte de la
**Etapa 2** (presencial) — no son responsabilidad de ninguno de los 4 roles
de arriba para esta entrega, según lo aclarado en la charla ("empezar
simple, escalar a multiagente solo si es necesario").
