"""Orquestador: corpus crudo -> extraccion -> limpieza -> doc_id -> chunking
-> registros de metadata listos para indexar (sec. 6 de la especificacion,
pasos 1-3 y 7). La construccion del indice FAISS (paso 4-6) vive en
src/embedding/build_index.py, que consume la salida de este modulo.

Este es el punto UNICO donde se resuelven `fenomeno` (sec. 2.3, por carpeta
del corpus) y `fuente` (sec. 10.2.1, clave real de emparejamiento con el
ground truth a nivel documento) -- ajustar aqui si cambia la organizacion o
la convencion del corpus real de ADL.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from ..chunking import chunk_document
from ..cleaning import clean_text, detect_language
from ..config import CHUNKS_INTERMEDIOS_PATH, CORPUS_DIR, FENOMENO_DIR_MAP
from ..extraction import extract_file
from .doc_id import compute_doc_id

logger = logging.getLogger(__name__)

# .png/.jpg y .pbf se excluyen del recorrido por defecto: OCR sobre imagenes
# es costoso y opcional (activar explicitamente si se necesita), y PBF no
# esta implementado (src/extraction/pbf_extractor.py).
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".txt", ".json", ".csv", ".xlsx"}


@dataclass
class ChunkRecord:
    """Un objeto = una linea de metadata.jsonl (Tabla 1 de la especificacion)."""

    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int | None
    posicion: int
    num_tokens: int
    texto: str
    idioma: str | None = None
    titulo_seccion: str | None = None

    def to_metadata_dict(self) -> dict:
        return asdict(self)


def resolve_fenomeno(path: Path, corpus_dir: Path = CORPUS_DIR) -> int | None:
    """Infiere el fenomeno (1, 2 o 3) a partir de la carpeta de nivel
    superior dentro de `corpus_dir` (config.FENOMENO_DIR_MAP). Si el corpus
    real de ADL no viene organizado por carpeta de fenomeno, este es el
    UNICO lugar que habria que modificar."""
    try:
        relative = path.relative_to(corpus_dir)
    except ValueError:
        return None
    top_level = relative.parts[0] if relative.parts else None
    return FENOMENO_DIR_MAP.get(top_level)


def derive_fuente(path: Path, raw_extra_metadata: dict) -> str:
    """Campo `fuente` (Tabla 1): nombre o URL del archivo original provisto
    por ADL. Es la clave real de emparejamiento con el ground truth a nivel
    documento (sec. 10.2.1) -- NO el doc_id arbitrario asignado por el
    equipo. Se aisla en esta funcion para poder ajustarla en un solo lugar
    cuando se conozca la convencion exacta que use ADL en su ground truth
    (hoy: URL si el extractor la reporto -- caso HTML -- o nombre de archivo
    en caso contrario)."""
    url = (raw_extra_metadata or {}).get("url")
    if url:
        return str(url)
    return path.name


def iter_corpus_files(corpus_dir: Path = CORPUS_DIR):
    """Recorre `corpus_dir` y devuelve los archivos de documento a procesar.

    Se excluyen archivos que no resuelven a ningun fenomeno conocido (p. ej.
    `fuentes.md`, el manifest del corpus de ejemplo, que vive en la raiz de
    `corpus_dir` y no dentro de una carpeta `fenomeno_N_*`) -- no son
    documentos del corpus, son metadatos sobre el corpus.
    """
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if resolve_fenomeno(path, corpus_dir) is None:
            logger.debug("archivo fuera de una carpeta de fenomeno reconocida, se omite: %s", path)
            continue
        yield path


def process_document(
    path: Path, count_tokens=None, corpus_dir: Path = CORPUS_DIR
) -> list[ChunkRecord]:
    raw_doc = extract_file(path)
    texto_limpio = clean_text(raw_doc.texto_crudo)
    if not texto_limpio.strip():
        logger.warning("documento sin texto util tras limpieza, se omite: %s", path)
        return []

    idioma = detect_language(texto_limpio)
    doc_id = compute_doc_id(path)
    fenomeno = resolve_fenomeno(path, corpus_dir)
    fuente = derive_fuente(path, raw_doc.extra_metadata)

    chunks = chunk_document(
        texto_limpio, formato=raw_doc.formato, lang=idioma, count_tokens=count_tokens
    )

    return [
        ChunkRecord(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-c{chunk.posicion:04d}",
            fuente=fuente,
            formato=raw_doc.formato,
            fenomeno=fenomeno,
            posicion=chunk.posicion,
            num_tokens=chunk.num_tokens,
            texto=chunk.texto,
            idioma=idioma,
            titulo_seccion=chunk.titulo_seccion,
        )
        for chunk in chunks
    ]


def build_corpus_chunks(
    corpus_dir: Path = CORPUS_DIR, count_tokens=None
) -> list[ChunkRecord]:
    all_records: list[ChunkRecord] = []
    for path in iter_corpus_files(corpus_dir):
        try:
            records = process_document(path, count_tokens=count_tokens, corpus_dir=corpus_dir)
        except Exception:
            logger.exception("fallo al procesar %s, se omite el archivo", path)
            continue
        all_records.extend(records)
        logger.info("%s -> %d chunks", path.name, len(records))
    return all_records


def write_chunks_jsonl(
    records: list[ChunkRecord], out_path: Path = CHUNKS_INTERMEDIOS_PATH
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_metadata_dict(), ensure_ascii=False) + "\n")
