"""Orquestador: corpus crudo -> extraccion -> limpieza -> doc_id -> chunking
-> registros de metadata listos para indexar (sec. 6 de la especificacion,
pasos 1-3 y 7). La construccion del indice FAISS (paso 4-6) vive en
src/embedding/build_index.py, que consume la salida de este modulo.

Este es el punto UNICO donde se resuelven `fenomeno` (sec. 2.3, por carpeta
del corpus) y `fuente` (Tabla 1) -- ajustar aqui si cambia la organizacion o
la convencion del corpus real de ADL.

Nota sobre la clave de emparejamiento: ADL aclaro en la Q&A final que el
ground truth a nivel documento se empareja por `doc_id` (suministrado por
ellos), no por `fuente` como decia la sec. 10.2.1 del PDF -- esa frase fue
un error de versionamiento. Ver `src/ingestion/doc_id.py` y
`docs/Explicacion_reto_final.md`. `fuente` se sigue reportando porque es un
campo obligatorio de la Tabla 1 y sirve de trazabilidad.
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..chunking import chunk_document
from ..cleaning import clean_text, detect_language
from ..config import CHUNKS_INTERMEDIOS_PATH, CORPUS_DIR, FENOMENO_DIR_MAP
from ..extraction import extract_file
from .doc_id import DocIdManifest, resolve_doc_id

logger = logging.getLogger(__name__)

# Todos los formatos que ADL declara en la sec. 1.3. `.avif` queda fuera a
# proposito: es un unico archivo de imagen y Pillow no lo lee sin plugin.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".html", ".htm", ".md", ".txt", ".json", ".csv", ".xlsx",
    ".png", ".jpg", ".jpeg", ".pbf",
}


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
    url: str | None = None  # URL de origen cuando el extractor la reporto

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
    por ADL.

    Se usa SIEMPRE el nombre de archivo, que es exactamente el "Nombre
    estandarizado" del inventario de ADL. La sec. 10.2.1 del PDF dice que el
    emparejamiento a nivel documento se hace por `fuente` y la Q&A final lo
    corrigio a `doc_id`; poner el nombre de archivo satisface las dos
    lecturas, mientras que poner la URL de scraping (lo que hacia antes con
    HTML/JSON) solo funcionaria con la segunda. La URL, cuando existe, se
    conserva aparte en el campo `url` del registro."""
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
    path: Path,
    count_tokens=None,
    corpus_dir: Path = CORPUS_DIR,
    doc_id_manifest: DocIdManifest | None = None,
) -> list[ChunkRecord]:
    raw_doc = extract_file(path)
    texto_limpio = clean_text(raw_doc.texto_crudo)
    if not texto_limpio.strip():
        logger.warning("documento sin texto util tras limpieza, se omite: %s", path)
        return []

    idioma = detect_language(texto_limpio)
    doc_id = resolve_doc_id(path, doc_id_manifest, corpus_dir=corpus_dir)
    fenomeno = resolve_fenomeno(path, corpus_dir)
    fuente = derive_fuente(path, raw_doc.extra_metadata)
    url = (raw_doc.extra_metadata or {}).get("url")
    url = str(url) if url else None

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
            url=url,
        )
        for chunk in chunks
    ]


def build_corpus_chunks(
    corpus_dir: Path = CORPUS_DIR,
    count_tokens=None,
    doc_id_manifest: DocIdManifest | None = None,
    checkpoint_path: Path | None = None,
) -> list[ChunkRecord]:
    """Extrae, limpia y fragmenta todo el corpus.

    Con `checkpoint_path` la pasada es REANUDABLE: los registros de cada
    documento se anexan al archivo apenas se producen, y al arrancar se
    recuperan los ya escritos para saltarse esos documentos. Sobre el corpus
    real la extraccion tarda casi una hora (760 PDFs, ~30k paginas, mas el OCR
    de los escaneos), asi que perderla entera por una interrupcion no es
    aceptable. Los chunks se re-indexan sin volver a extraer.
    """
    all_records: list[ChunkRecord] = []
    ya_hechos: set[str] = set()

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.is_file():
            # Solo los doc_id, no los registros: releer 100.000 chunks enteros
            # a memoria solo para saber que archivos faltan agota la RAM de la
            # maquina de desarrollo (7,6 GB) antes de empezar.
            ya_hechos = read_done_doc_ids(checkpoint_path)
            logger.info(
                "reanudando: %d documentos ya estaban en %s",
                len(ya_hechos), checkpoint_path,
            )

    for path in iter_corpus_files(corpus_dir):
        # El doc_id se resuelve del manifest sin abrir el archivo, asi que
        # comprobar si ya esta hecho es barato.
        doc_id = resolve_doc_id(path, doc_id_manifest, corpus_dir=corpus_dir)
        if doc_id in ya_hechos:
            continue
        try:
            records = process_document(
                path,
                count_tokens=count_tokens,
                corpus_dir=corpus_dir,
                doc_id_manifest=doc_id_manifest,
            )
        except Exception:
            logger.exception("fallo al procesar %s, se omite el archivo", path)
            continue
        ya_hechos.add(doc_id)
        if checkpoint_path is not None:
            # Con checkpoint la fuente de verdad es el archivo, no la memoria:
            # el indexado lo relee despues con read_chunks_jsonl(). Acumular
            # ademas en RAM duplicaria el corpus entero en memoria.
            if records:
                append_chunks_jsonl(records, checkpoint_path)
        else:
            all_records.extend(records)
        logger.info("%s -> %d chunks", path.name, len(records))
    return all_records


def _dump(records: list[ChunkRecord], f) -> None:
    for record in records:
        f.write(json.dumps(record.to_metadata_dict(), ensure_ascii=False) + "\n")


def write_chunks_jsonl(
    records: list[ChunkRecord], out_path: Path = CHUNKS_INTERMEDIOS_PATH
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        _dump(records, f)


def append_chunks_jsonl(records: list[ChunkRecord], out_path: Path) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        _dump(records, f)


def read_done_doc_ids(path: Path) -> set[str]:
    """doc_id ya presentes en un jsonl de chunks, sin cargar los registros."""
    hechos: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                hechos.add(json.loads(linea)["doc_id"])
    return hechos


def read_chunks_jsonl(path: Path) -> list[ChunkRecord]:
    """Relee un jsonl de chunks como ChunkRecord (para reanudar o reindexar
    sin volver a extraer)."""
    campos = {f.name for f in fields(ChunkRecord)}
    records = []
    with path.open(encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                d = json.loads(linea)
                records.append(ChunkRecord(**{k: v for k, v in d.items() if k in campos}))
    return records
