# CODEFEST 2026 — explicación completa del proyecto

Documento para explicar el proyecto de punta a punta (a un compañero de equipo,
un jurado, o uno mismo dentro de un mes). No reemplaza el `README.md` (que es
el manual de "cómo correrlo"): esto es el mapa de "qué hay y por qué".

## 1. Qué hace el proyecto

Es la Etapa 1 del reto CODEFEST AD ASTRA 2026: construir una **base de
conocimiento vectorial** a partir de documentos (PDF/HTML/texto/tablas/imágenes)
sobre 3 "fenómenos" (IA en defensa, órbita baja espacial, dinámicas
territoriales), y responder consultas devolviendo los documentos y fragmentos
más relevantes — **sin ningún modelo generativo** (prohibido por la
especificación, sección 8.3). Es retrieval puro: extracción → limpieza →
fragmentación (chunking) → embeddings → índice FAISS → búsqueda → agregación →
(opcional) grafo de conocimiento como refuerzo.

## 2. Flujo de datos (las 2 fases)

```
FASE OFFLINE (una vez por corpus, "compilar")
  corpus/*.{pdf,html,txt,csv,json,png...}
    -> src/extraction/*      (texto crudo por formato)
    -> src/cleaning/*        (limpieza + deteccion de idioma ES/EN/PT)
    -> src/chunking/*        (fragmentos por presupuesto de tokens)
    -> src/embedding/*       (vector por fragmento, encoder real o fake)
    -> Entrega/base_vectorial/encoder_.../index.faiss + metadata.jsonl
    -> src/graph/*           (grafo de entidades/relaciones, opcional, bonus)
    -> Entrega/base_vectorial/grafo/grafo.graphml

FASE ONLINE (cada vez que llega una consulta)
  consulta de texto
    -> src/embedding  (mismo encoder, vectoriza la consulta)
    -> src/retrieval/search.py    (top-k en FAISS)
    -> src/retrieval/aggregate.py (fragmentos -> 3 documentos)
    -> src/graph/graph_retrieval.py (fusiona con vecinos del grafo, opcional)
    -> src/retrieval/fusion.py + truncate.py (arma 10 fragmentos <=250 palabras)
    -> Entrega/resultados.jsonl
```

`Entrega/generador.py` es el script que orquesta toda la fase online.
`scripts/build_corpus_index.py` orquesta toda la fase offline.

## 3. Módulo por módulo (`src/`)

- **`extraction/`** — un extractor por formato (`pdf_extractor.py`,
  `html_extractor.py`, `text_extractor.py`, `tabular_extractor.py`,
  `json_extractor.py`, `image_extractor.py` con OCR vía pytesseract,
  `pbf_extractor.py`), todos implementando la interfaz de `base.py`.
  `registry.py` elige el extractor según la extensión del archivo.
- **`cleaning/`** — `clean.py` normaliza el texto extraído (espacios, saltos de
  línea, ruido de OCR); `lang_detect.py` detecta ES/EN/PT con `langdetect`.
- **`chunking/`** — `chunker.py` corta el texto limpio en fragmentos que caben
  en un presupuesto de tokens (usa `sentence_split.py` para no cortar a mitad
  de oración, y recibe el `Encoder.count_tokens` inyectado para contar tokens
  con el tokenizer real del modelo elegido).
- **`embedding/`** — `encoders.py` define la interfaz `Encoder` (ABC) con dos
  implementaciones: `SentenceTransformerEncoder` (real, usa
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384
  dimensiones, ES/EN/PT nativo) y `HashingFakeEncoder` (determinista, sin red,
  solo para probar la mecánica del pipeline sin depender de internet ni de
  calidad semántica real). `build_index.py` vectoriza todos los fragmentos y
  arma el índice FAISS + `metadata.jsonl` paralelo.
- **`retrieval/`** — `search.py` busca los k vecinos más cercanos en FAISS;
  `aggregate.py` colapsa fragmentos a nivel de documento (para elegir los 3
  documentos más relevantes); `fusion.py` combina el ranking vectorial con el
  del grafo si está activado; `truncate.py` recorta cada fragmento a <=250
  palabras según pide el esquema de entrega.
- **`graph/`** — bonus. `ner.py` extrae entidades con spaCy
  (`es/en/pt_core_news_sm`); `relations.py` infiere relaciones simples entre
  entidades co-ocurrentes; `build_graph.py` arma un grafo `networkx` y lo
  serializa a GraphML; `graph_retrieval.py` dado un resultado vectorial, busca
  vecinos de primer orden en el grafo para reforzar/expandir la recuperación.
- **`gui/`** — `runner.py` es la capa de orquestación que la GUI usa (carga
  perezosa del índice/encoder en background, ejecuta consultas, dispara
  reconstrucción del índice); `history.py` guarda cada interacción en
  `intermedios/historial_ejecuciones.jsonl`.
- **`ingestion/`** — `pipeline.py` conecta extracción+limpieza+chunking en un
  solo paso por documento y calcula `derive_fuente()` (la clave de
  emparejamiento con el ground truth); `doc_id.py` genera IDs estables de
  documento (hash) para que el mismo documento tenga siempre el mismo id.
- **`config.py`** — constantes centrales: nombre del encoder por defecto,
  mapeo de idioma → modelo spaCy, límites de tokens/palabras, rutas por
  defecto.

## 4. Scripts (`scripts/`)

- `build_corpus_index.py` — CLI de la fase offline (ver sección 2).
- `gen_synthetic_corpus.py` — genera `corpus_ejemplo/` (solo desarrollo, no es
  parte del pipeline de entrega).
- `gen_informe_tecnico.py` — genera `Entrega/informe_tecnico.pdf`.
- `inspect_results.py` — imprime en consola, de forma legible, qué documentos y
  fragmentos trajo cada consulta de `Entrega/resultados.jsonl`.
- `gui_app.py` — lanza la interfaz gráfica (Tkinter, sin dependencias nuevas).

## 5. Tests (`tests/`, 31 pruebas, todas en verde)

`test_chunking.py`, `test_doc_id.py`, `test_extraction_smoke.py` (un extractor
por formato), `test_faiss_alignment.py` (que el índice y `metadata.jsonl`
queden alineados en el mismo orden), `test_graph.py` (NER + vecinos del
grafo — requiere los modelos spaCy instalados), `test_retrieval.py`,
`test_retrieval_schema.py` (que `resultados.jsonl` cumpla el esquema oficial:
3 documentos + 10 fragmentos <=250 palabras). Corren con `HashingFakeEncoder`
para no depender de red; validan mecánica, no calidad semántica.

## 6. Carpeta `Entrega/` (lo que se entrega al reto)

- `generador.py` — script de la fase online (ver sección 2).
- `base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/` —
  `index.faiss` (384 dim, confirmado con el encoder real, no el fake) +
  `metadata.jsonl` paralelo.
- `base_vectorial/grafo/grafo.graphml` — grafo de conocimiento (bonus).
- `resultados.jsonl` — 50 líneas, una por consulta, esquema
  `{query_id, documents[3], fragments[10]}`.
- `informe_tecnico.pdf` — informe técnico exigido por la especificación.

## 7. Qué es provisional (se reemplaza sin tocar `src/`)

- `corpus_ejemplo/` y `corpus_real_ejemplo/` — corpus **sintético**, no los
  documentos reales de ADL (bloqueados por el proxy de este entorno de
  desarrollo). No citar como fuentes reales.
- `consultas_prueba/consultas_prueba.jsonl` (15) y `consultas_50.jsonl` (50) —
  consultas de prueba escritas a mano, no las oficiales q001-q050.
- El NER usa spaCy en vez del modelo de HuggingFace originalmente considerado
  (mismo motivo: sin acceso a `huggingface.co` para ese modelo en particular).

Para pasar a los datos oficiales: colocar el corpus real donde hoy está
`corpus_ejemplo/` (o `--corpus-dir`) y correr de nuevo los pasos 4 y 5 del
`README.md`. Ningún archivo de `src/` necesita cambiar.

## 8. Estado actual (verificado hoy, 2026-07-26)

- `pytest` → **31 passed, 0 failed** (antes había 2 fallos por faltar los
  modelos de spaCy `es/en/pt_core_news_sm`; ya instalados y corregidos).
- El índice y `resultados.jsonl` en `Entrega/` fueron generados con el encoder
  **real** (`paraphrase-multilingual-MiniLM-L12-v2`), no con el fake —
  confirmado por el nombre de la carpeta del índice y su dimensión (384).
- Pendiente real (no técnico, sino de datos): reemplazar corpus y consultas
  sintéticos por los oficiales de ADL cuando estén disponibles, y regenerar
  `Entrega/` con ellos antes de la entrega final.
