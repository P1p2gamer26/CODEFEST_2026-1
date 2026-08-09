"""Configuracion centralizada del pipeline (paths, encoders, presupuestos).

Cambiar aqui para re-apuntar el pipeline al corpus/consultas reales de ADL:
no deberia ser necesario tocar el resto de `src/`.
"""

from pathlib import Path

# dev/src/config.py -> DEV_DIR = <raiz>/dev, ROOT_DIR = <raiz>.
DEV_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = DEV_DIR.parent

# --- Corpus y consultas ---
# dev/corpus/ es el corpus real de ADL, ya descomprimido (gitignoreado por peso).
# dev/corpus_meta/ guarda los ZIP originales, el indice xlsx y el PDF de
# preguntas: son metadatos SOBRE el corpus, no documentos, y por eso viven
# fuera del arbol que recorre `iter_corpus_files()`.
CORPUS_DIR = DEV_DIR / "corpus"
CORPUS_META_DIR = DEV_DIR / "corpus_meta"
INDICE_DATOS_XLSX_PATH = CORPUS_META_DIR / "Indice_Datos_Codefest.xlsx"
CONSULTAS_PRUEBA_PATH = DEV_DIR / "consultas_prueba" / "consultas_prueba.jsonl"
CONSULTAS_OFICIALES_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"

# Carpeta de nivel superior dentro de CORPUS_DIR -> numero de fenomeno (1, 2 o 3).
# Son los nombres de las tres carpetas raiz de los ZIP de ADL.
FENOMENO_DIR_MAP = {
    "F1_IA_y_Capacidades_Estrategicas": 1,
    "F2_Seguridad_Entorno_Espacial": 2,
    "F3_Dinamicas_Territoriales": 3,
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
INTERMEDIOS_DIR = DEV_DIR / "intermedios"
CHUNKS_INTERMEDIOS_PATH = INTERMEDIOS_DIR / "chunks_intermedios.jsonl"
# Manifest {ruta, nombre, doc_id} derivado del xlsx de ADL
# (scripts/manifest_desde_xlsx.py).
DOC_ID_MANIFEST_PATH = INTERMEDIOS_DIR / "doc_id_manifest.csv"

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

# Secundario (opcional, sec. 4.4): se usa junto al primario fusionando ambos
# rankings con RRF (sec. 8.4). Elegido por DIVERSIDAD, no por ser "mejor": E5
# esta entrenado para recuperacion densa mientras que el primario esta afinado
# para similitud de parafrasis, asi que sus errores no estan correlacionados --
# que es la condicion para que fusionar dos rankings aporte algo. MIT, 768 dim,
# multilingue nativo, limite 512 tokens.
ENCODER_SECONDARY_NAME = "multilingual-e5-base"
ENCODER_SECONDARY_HF_ID = "intfloat/multilingual-e5-base"

# La familia E5 exige estos prefijos: sin ellos la calidad cae de forma
# silenciosa (el modelo fue entrenado siempre con ellos). Consulta y pasaje
# llevan prefijos DISTINTOS -- por eso `Encoder` expone codificacion asimetrica.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Candidato en evaluacion (2 ago 2026). Encoder-only tipo BERT, NO decoder:
# cumple la sec. 8.3 igual que los otros dos, a diferencia de la familia
# Harrier/Qwen3-Embedding, que son decoder-only y quedan descartadas por
# riesgo de incumplimiento aunque encabecen los rankings de 2026.
# Apache 2.0, 305 M parametros, ventana 8192 (MiniLM trunca en 128 y el 96%
# de los chunks son mas largos), 768 dim, sin prefijos de consulta/pasaje.
ENCODER_GTE_NAME = "gte-multilingual-base"
ENCODER_GTE_HF_ID = "Alibaba-NLP/gte-multilingual-base"

# E02 de dev/experimentos/cola.jsonl: mismo orden de parametros que el
# primario (118 M) y misma dimension (384), pero ventana 512 en vez de los
# 128 tokens en los que MiniLM trunca. Sirve para separar "la ventana no
# importa" de "e5-base era peor por otra razon". Lleva los prefijos de E5.
ENCODER_E5_SMALL_NAME = "multilingual-e5-small"
ENCODER_E5_SMALL_HF_ID = "intfloat/multilingual-e5-small"

# E04 de dev/experimentos/cola.jsonl: el encoder mas fuerte que cabe en esta
# CPU (560 M, dim 1024, ventana 512). Se evalua SOLO como re-puntuador -- E02
# midio que la familia E5 rinde mal como primario sobre este corpus. Lleva los
# mismos prefijos que el resto de la familia.
ENCODER_E5_LARGE_NAME = "multilingual-e5-large"
ENCODER_E5_LARGE_HF_ID = "intfloat/multilingual-e5-large"

# E25: el unico candidato de docs/plan_encoders.md que nunca se midio. Cumple
# las tres restricciones duras del proyecto -- backbone ENCODER-ONLY
# (XLM-RoBERTa, o sea sin riesgo bajo la sec. 8.3, que es lo que descarto a
# Qwen3, Harrier, KaLM/EmbeddingGemma y Jina v5), licencia MIT, y ventana de
# 8192 que sobra para chunks de 280 tokens. NO lleva prefijos: el propio model
# card dice que "no longer requires adding instructions to the queries", asi
# que ponerselos degradaria en silencio igual que a GTE.
#
# Su parte SPARSE no se usa y no hay que reabrirla: es BM25 con otro nombre, y
# el hibrido lexico ya se midio dos veces (RRF y union) y perdio 15-4 y 19-2.
# Aca se usa solo la representacion densa.
ENCODER_BGE_M3_NAME = "bge-m3"
ENCODER_BGE_M3_HF_ID = "BAAI/bge-m3"


def encoder_dir(encoder_name: str) -> Path:
    return BASE_VECTORIAL_DIR / f"encoder_{encoder_name}"

# --- Chunking ---
# El presupuesto debe respetar el limite MINIMO de entrada entre todos los
# encoders en uso (hoy 512 en ambos). 280 deja margen de sobra aunque los
# tokenizers difieran entre si.
CHUNK_TOKEN_BUDGET = 280
# Tope de filas por CSV/XLSX. El corpus real trae datasets bibliograficos del
# AI Index de hasta 35 MB (registros de PubMed) que no responden a ninguna de
# las 50 consultas: sin tope aportarian mas de 100.000 chunks de ruido, que
# inflan la entrega y bajan la precision. Con tope el documento sigue en el
# indice y puede ganar F1@3, solo que no domina el espacio vectorial.
MAX_FILAS_TABULARES = 500
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
