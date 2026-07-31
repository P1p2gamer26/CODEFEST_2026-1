"""Construccion y persistencia del indice FAISS (sec. 5 y 6, pasos 4-7).

Invariante critico: el orden de `records` (lista de ChunkRecord) debe ser
exactamente el orden en que se insertan en `index.add()`, porque FAISS solo
guarda vectores + un id interno entero -- `metadata.jsonl` debe tener una
linea por vector, en ese mismo orden (sec. 5.3), para que el id interno de
FAISS devuelto por una busqueda mapee a la linea correcta del metadata.
"""

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
        if embeddings.shape == forma:
            hechas = int(progreso_path.read_text())
            logger.info("reanudando codificacion desde la fila %d de %d", hechas, forma[0])
        else:
            # Cambio el corpus o el encoder: el cache no sirve.
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
        progreso_path.write_text(str(fin))
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

    if index.ntotal != len(metadata):
        raise ValueError(
            f"indice desalineado al cargar: index.ntotal={index.ntotal} vs "
            f"lineas de metadata={len(metadata)} ({index_dir})"
        )

    return index, metadata
