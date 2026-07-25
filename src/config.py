"""Configuracion centralizada del pipeline (paths, encoders, presupuestos).

Cambiar aqui para re-apuntar el pipeline al corpus/consultas reales de ADL:
no deberia ser necesario tocar el resto de `src/`.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Corpus y consultas (PROVISIONAL: se reemplaza por lo oficial de ADL) ---
CORPUS_DIR = ROOT_DIR / "corpus_ejemplo"
CONSULTAS_PRUEBA_PATH = ROOT_DIR / "consultas_prueba" / "consultas_prueba.jsonl"

# Carpeta de nivel superior dentro de CORPUS_DIR -> numero de fenomeno (1, 2 o 3).
# El corpus real de ADL puede no venir organizado por carpeta de fenomeno; en ese
# caso ajustar unicamente `resolve_fenomeno()` en src/ingestion/pipeline.py.
FENOMENO_DIR_MAP = {
    "fenomeno_1_ia_defensa": 1,
    "fenomeno_2_leo_espacial": 2,
    "fenomeno_3_dinamicas_territoriales": 3,
}

# Formatos de archivo soportados por extension -> nombre de formato normalizado
# (el campo obligatorio `formato` de la Tabla 1 solo admite pdf/html/md; los
# demas formatos de origen se registran igual para trazabilidad interna).
EXTENSION_TO_FORMATO = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".txt": "md",
    ".json": "json",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".png": "img",
    ".jpg": "img",
    ".jpeg": "img",
    ".pbf": "pbf",
}

# --- Artefactos intermedios (gitignored, fuera de Entrega/) ---
INTERMEDIOS_DIR = ROOT_DIR / "intermedios"
CHUNKS_INTERMEDIOS_PATH = INTERMEDIOS_DIR / "chunks_intermedios.jsonl"

# --- Entrega oficial ---
ENTREGA_DIR = ROOT_DIR / "Entrega"
BASE_VECTORIAL_DIR = ENTREGA_DIR / "base_vectorial"
RESULTADOS_PATH = ENTREGA_DIR / "resultados.jsonl"
GRAFO_PATH = BASE_VECTORIAL_DIR / "grafo" / "grafo.graphml"

# --- Encoders (HuggingFace, arquitectura encoder, sin modelos generativos) ---
# Primario: multilingue ES/EN/PT, Apache 2.0, 384 dim, limite 512 tokens.
ENCODER_PRIMARY_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_PRIMARY_HF_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_MAX_INPUT_TOKENS = 512

def encoder_dir(encoder_name: str) -> Path:
    return BASE_VECTORIAL_DIR / f"encoder_{encoder_name}"

# --- Chunking ---
CHUNK_TOKEN_BUDGET = 280  # margen bajo el limite de 512 tokens del encoder
CHUNK_OVERLAP_SENTENCES = 1

# --- Formato de salida (resultados.jsonl) ---
MAX_FRAGMENT_WORDS = 250
N_DOCUMENTS_PER_QUERY = 3
N_FRAGMENTS_PER_QUERY = 10

# --- Recuperacion ---
DEFAULT_K_CHUNKS = N_FRAGMENTS_PER_QUERY
OVERFETCH_FACTOR = 4  # sobre-recuperar antes de post-filtros/agregacion
RRF_K0 = 60

# --- Grafo de conocimiento (bonus) ---
NER_MODEL_HF_ID = "Babelscape/wikineural-multilingual-ner"
SPACY_MODEL_BY_LANG = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
}
