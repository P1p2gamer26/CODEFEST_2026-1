"""Verifica el invariante critico de la seccion 5.3: metadata.jsonl debe
tener exactamente una linea por vector del indice FAISS, en el mismo orden."""

import faiss

from src.embedding.build_index import build_and_persist, load_index
from src.embedding.encoders import HashingFakeEncoder
from src.ingestion.pipeline import ChunkRecord

SAMPLE_TEXTS = [
    "Los satelites en orbita baja enfrentan riesgos crecientes de colision.",
    "La inteligencia artificial se aplica cada vez mas en logistica militar.",
    "El desarrollo territorial en America Latina muestra brechas persistentes.",
    "Space debris mitigation requires international cooperation.",
    "AI governance frameworks are still under discussion among allies.",
]


def _make_records() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            doc_id=f"doc{i:02d}",
            chunk_id=f"doc{i:02d}-c0000",
            fuente=f"fuente_{i}.pdf",
            formato="pdf",
            fenomeno=(i % 3) + 1,
            posicion=0,
            num_tokens=len(text.split()),
            texto=text,
        )
        for i, text in enumerate(SAMPLE_TEXTS)
    ]


def test_index_and_metadata_have_same_length_and_order(tmp_path):
    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()

    out_dir = build_and_persist(records, encoder, out_dir=tmp_path / "encoder_test")

    assert (out_dir / "index.faiss").exists()
    assert (out_dir / "metadata.jsonl").exists()

    index, metadata = load_index(encoder.name, index_dir=out_dir)
    assert index.ntotal == len(records)
    assert len(metadata) == len(records)
    for record, meta in zip(records, metadata):
        assert meta["chunk_id"] == record.chunk_id
        assert meta["texto"] == record.texto


def test_self_search_returns_near_perfect_score(tmp_path):
    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    out_dir = build_and_persist(records, encoder, out_dir=tmp_path / "encoder_test")
    index, metadata = load_index(encoder.name, index_dir=out_dir)

    target = records[2]
    query_vec = encoder.encode_one(target.texto)
    scores, ids = index.search(query_vec.reshape(1, -1), 1)

    assert metadata[ids[0][0]]["chunk_id"] == target.chunk_id
    assert scores[0][0] > 0.999


def test_persist_raises_on_length_mismatch(tmp_path):
    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    index = faiss.IndexFlatIP(encoder.dim)
    index.add(encoder.encode([r.texto for r in records]))

    from src.embedding.build_index import persist_index

    try:
        persist_index(index, records[:-1], encoder.name, out_dir=tmp_path / "encoder_bad")
        assert False, "se esperaba ValueError por desalineacion"
    except ValueError:
        pass
