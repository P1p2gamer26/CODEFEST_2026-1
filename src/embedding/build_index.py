"""Construccion y persistencia del indice FAISS (sec. 5 y 6, pasos 4-7).

Invariante critico: el orden de `records` (lista de ChunkRecord) debe ser
exactamente el orden en que se insertan en `index.add()`, porque FAISS solo
guarda vectores + un id interno entero -- `metadata.jsonl` debe tener una
linea por vector, en ese mismo orden (sec. 5.3), para que el id interno de
FAISS devuelto por una busqueda mapee a la linea correcta del metadata.
"""

import json
from pathlib import Path

import faiss

from ..config import encoder_dir
from ..ingestion.pipeline import ChunkRecord
from .encoders import Encoder


def build_faiss_index(records: list[ChunkRecord], encoder: Encoder) -> faiss.Index:
    if not records:
        raise ValueError("no hay chunks para indexar")

    embeddings = encoder.encode([r.texto for r in records])
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
    records: list[ChunkRecord], encoder: Encoder, out_dir: Path | None = None
) -> Path:
    index = build_faiss_index(records, encoder)
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
