# Pipeline — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para el reto. Ver la especificacion completa en
`Material de apoyo/CODEFEST_2026-1.pdf` y el plan de esta iteracion en el
historial de la rama `Julian_Africano`.

## Que es provisional vs. que es infraestructura reusable

**Provisional (se reemplaza cuando ADL entregue lo oficial, sin tocar `src/`):**

- `corpus_ejemplo/`: 15 documentos **sinteticos** (PDF/HTML, ES/EN/PT), redactados a
  mano con `scripts/gen_synthetic_corpus.py` porque el proxy de salida de este
  entorno de desarrollo bloquea la descarga de documentos reales (ver
  `informe_tecnico.pdf`, seccion de limitaciones, y `corpus_ejemplo/fuentes.md`).
  **No son documentos reales, no citar como tales.**
- `consultas_prueba/consultas_prueba.jsonl`: 15 consultas de prueba ES/EN/PT
  escritas a mano, NO son las 50 consultas oficiales q001-q050 (ADL aun no las
  entrega). El formato exacto del archivo oficial de consultas tampoco se conoce
  todavia; el adaptador esta aislado en `load_consultas()` dentro de
  `Entrega/generador.py`.
- El campo `fuente` (clave real de emparejamiento con el ground truth a nivel
  documento) se deriva hoy del nombre de archivo o la URL detectada
  (`derive_fuente()` en `src/ingestion/pipeline.py`) -- ajustar ahi si ADL usa
  otra convencion.

**Reusable sin cambios:** todo `src/` (extraction, cleaning, chunking, embedding,
retrieval, graph, ingestion), y la logica central de `Entrega/generador.py`.
Para usar el corpus real: colocarlo donde hoy esta `corpus_ejemplo/` (o apuntar
`--corpus-dir` a otra carpeta) y volver a correr `scripts/build_corpus_index.py`.

## Limitacion de red de este entorno de desarrollo

El proxy de salida de este sandbox bloquea (403) tanto los dominios de las
fuentes documentales reales (sipri.org, esa.int, cepal.org, etc.) como
`huggingface.co`. Esto significa que:

1. No se pudo descargar un corpus real -> se genero uno sintetico (ver arriba).
2. No se pudieron descargar los pesos del encoder real
   (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) ni el modelo
   de NER de HuggingFace originalmente considerado para el grafo.

Mitigacion:
- El encoder esta detras de una interfaz intercambiable
  (`src/embedding/encoders.py`): `SentenceTransformerEncoder` (real, produccion)
  y `HashingFakeEncoder` (determinista, sin red, solo para pruebas de mecanica
  -- NO produce embeddings semanticamente validos). Todo el pipeline y la
  suite de tests corren con el fake encoder en este entorno; el codigo del
  encoder real esta completo y listo para correr donde haya acceso normal a
  `huggingface.co`.
- El grafo de conocimiento usa el NER ya incluido en los modelos spaCy
  `es/en/pt_core_news_sm` (se instalan como paquetes de pip, no requieren
  `huggingface.co`) en vez del modelo HF originalmente propuesto.

**Antes de la entrega final**, correr en un entorno con acceso normal a internet:

```bash
python scripts/build_corpus_index.py --with-graph
python Entrega/generador.py --consultas <archivo_oficial_de_ADL>
```

Esto sobrescribe `Entrega/base_vectorial/` y `Entrega/resultados.jsonl` con el
indice y los resultados reales (encoder real, corpus real).

## Estructura

```
src/                    codigo reusable del pipeline
corpus_ejemplo/         corpus de ejemplo SINTETICO (provisional)
consultas_prueba/       consultas de prueba (provisional)
scripts/
  gen_synthetic_corpus.py   genera corpus_ejemplo/ (dev only, no es parte del pipeline)
  build_corpus_index.py     OFFLINE: corpus -> indice FAISS + metadata + grafo
  inspect_results.py        inspeccion cualitativa manual de resultados.jsonl
tests/                  pytest (30 pruebas, corren con HashingFakeEncoder, sin red)
Entrega/                estructura oficial de entrega (ver especificacion, seccion 1.4)
```

## Como correr todo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm

pytest tests/ -v

# Demo end-to-end sin red (encoder falso, solo para validar la mecanica):
python scripts/build_corpus_index.py --use-fake-encoder --with-graph \
  --out-dir intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --graph-out-path intermedios/demo_index_fake_encoder/grafo.graphml
python Entrega/generador.py --consultas consultas_prueba/consultas_prueba.jsonl \
  --use-fake-encoder \
  --index-dir intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --out intermedios/demo_index_fake_encoder/resultados_demo.jsonl
python scripts/inspect_results.py \
  --consultas consultas_prueba/consultas_prueba.jsonl \
  --resultados intermedios/demo_index_fake_encoder/resultados_demo.jsonl \
  --index-dir intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2

# Entrega real (requiere acceso a huggingface.co y, cuando este disponible, el
# corpus real de ADL en vez de corpus_ejemplo/):
python scripts/build_corpus_index.py --with-graph
python Entrega/generador.py --consultas <consultas_oficiales_de_ADL>
```
