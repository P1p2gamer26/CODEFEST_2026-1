"""Construccion y persistencia del indice FAISS (sec. 5 y 6, pasos 4-7).

Invariante critico: el orden de `records` (lista de ChunkRecord) debe ser
exactamente el orden en que se insertan en `index.add()`, porque FAISS solo
guarda vectores + un id interno entero -- `metadata.jsonl` debe tener una
linea por vector, en ese mismo orden (sec. 5.3), para que el id interno de
FAISS devuelto por una busqueda mapee a la linea correcta del metadata.
"""

import hashlib
import json
import logging
from pathlib import Path

import faiss
import numpy as np

from ..config import encoder_dir
from ..ingestion.pipeline import ChunkRecord
from .encoders import Encoder

logger = logging.getLogger(__name__)

# Filas por bloque de codificacion. Solo fija cada cuanto se graba el
# checkpoint; el batch real lo maneja sentence-transformers.
BLOQUE_EMBEDDING = 4096


def _huella(records: list[ChunkRecord]) -> str:
    """Hash del texto ya codificado, para invalidar el cache si el texto
    cambio sin cambiar el chunk_id."""
    h = hashlib.blake2b(digest_size=16)
    for r in records:
        h.update(r.texto.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def encode_records(
    records: list[ChunkRecord], encoder: Encoder, cache_path: Path | None = None
) -> np.ndarray:
    """Codifica los chunks, opcionalmente con checkpoint reanudable.

    Codificar el corpus completo son horas de CPU: sin checkpoint, cualquier
    interrupcion obliga a empezar de cero. Con `cache_path` los vectores se
    escriben a un .npy por bloques y una corrida nueva retoma donde iba.
    """
    # encode_passages (no encode) para que los encoders que lo requieran
    # apliquen su prefijo de pasaje -- ver src/embedding/encoders.py.
    if cache_path is None:
        return encoder.encode_passages([r.texto for r in records])

    forma = (len(records), encoder.dim)
    progreso_path = cache_path.with_suffix(".progreso")
    hechas = 0
    if cache_path.exists() and progreso_path.exists():
        embeddings = np.lib.format.open_memmap(cache_path, mode="r+")
        # La forma sola no basta para validar el cache: detecta que cambio el
        # numero de chunks o el encoder, pero NO que los mismos N chunks vengan
        # en otro orden (el checkpoint de extraccion se anexa por documento, y
        # basta reprocesar uno para reordenarlos). En ese caso las filas ya
        # codificadas corresponderian a otros textos y el indice saldria
        # corrupto en silencio, que es el peor final posible tras horas de CPU.
        # Por eso el .progreso guarda tambien el chunk_id de la ultima fila.
        # Y tampoco basta el chunk_id: al reparar los guiones de fin de linea
        # (ver src/cleaning/clean.py) cambio el TEXTO de 38.423 chunks sin
        # cambiar ni su cantidad ni su chunk_id, asi que el cache viejo se
        # habria dado por bueno y el indice habria quedado con los vectores
        # del texto roto. Por eso el .progreso guarda un hash del texto.
        guardado = progreso_path.read_text().splitlines()
        candidatas = int(guardado[0]) if guardado else 0
        chunk_id_esperado = guardado[1] if len(guardado) > 1 else None
        huella_esperada = guardado[2] if len(guardado) > 2 else None
        alineado = (
            embeddings.shape == forma
            and 0 < candidatas <= len(records)
            and chunk_id_esperado == records[candidatas - 1].chunk_id
            and huella_esperada == _huella(records[:candidatas])
        )
        if alineado:
            hechas = candidatas
            logger.info("reanudando codificacion desde la fila %d de %d", hechas, forma[0])
        else:
            # Cambio el corpus, el orden o el encoder: el cache no sirve.
            del embeddings
            hechas = 0

    if hechas == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype="float32", shape=forma
        )

    for inicio in range(hechas, forma[0], BLOQUE_EMBEDDING):
        fin = min(inicio + BLOQUE_EMBEDDING, forma[0])
        embeddings[inicio:fin] = encoder.encode_passages(
            [r.texto for r in records[inicio:fin]]
        )
        embeddings.flush()
        progreso_path.write_text(f"{fin}\n{records[fin - 1].chunk_id}\n{_huella(records[:fin])}")
        logger.info("codificados %d/%d chunks (%s)", fin, forma[0], encoder.name)

    return np.asarray(embeddings)


def build_faiss_index(
    records: list[ChunkRecord], encoder: Encoder, cache_path: Path | None = None
) -> faiss.Index:
    if not records:
        raise ValueError("no hay chunks para indexar")

    embeddings = encode_records(records, encoder, cache_path)
    index = faiss.IndexFlatIP(encoder.dim)  # producto interno = coseno (vectores normalizados)
    index.add(embeddings)
    return index


def persist_index(
    index: faiss.Index,
    records: list[ChunkRecord],
    encoder_name: str,
    out_dir: Path | None = None,
) -> Path:
    if index.ntotal != len(records):
        raise ValueError(
            f"indice y metadata desalineados: index.ntotal={index.ntotal} vs "
            f"len(records)={len(records)}"
        )

    out_dir = out_dir or encoder_dir(encoder_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))

    metadata_path = out_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_metadata_dict(), ensure_ascii=False) + "\n")

    return out_dir


def build_and_persist(
    records: list[ChunkRecord],
    encoder: Encoder,
    out_dir: Path | None = None,
    cache_path: Path | None = None,
) -> Path:
    index = build_faiss_index(records, encoder, cache_path)
    return persist_index(index, records, encoder.name, out_dir)


def load_index(encoder_name: str, index_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    index_dir = index_dir or encoder_dir(encoder_name)

    index = faiss.read_index(str(index_dir / "index.faiss"))

    metadata: list[dict] = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    # Filas metadata-only (documentos sin texto, marcadas `en_indice: false`)
    # no tienen vector: la alineacion se verifica contra las que si lo tienen,
    # y esas tienen que ser exactamente las primeras `ntotal` (el parche las
    # anade al final; una intercalada romperia el mapeo id->metadata).
    con_vector = [m for m in metadata if m.get("en_indice") is not False]
    if index.ntotal != len(con_vector):
        raise ValueError(
            f"indice desalineado al cargar: index.ntotal={index.ntotal} vs "
            f"lineas de metadata={len(metadata)} ({index_dir})"
        )
    if metadata[: index.ntotal] != con_vector:
        raise ValueError(
            f"indice desalineado al cargar: filas sin vector intercaladas "
            f"entre las primeras {index.ntotal} ({index_dir})"
        )

    return index, metadata
