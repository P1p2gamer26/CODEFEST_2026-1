"""Verifica el invariante critico de la seccion 5.3: metadata.jsonl debe
tener exactamente una linea por vector del indice FAISS, en el mismo orden."""

import json

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


def test_load_index_admite_filas_metadata_only_al_final(tmp_path):
    """Documentos sin texto (imagenes del manifest) se conservan con una fila
    `en_indice: false` al final. La alineacion se verifica contra las filas con
    vector, y estas tienen que ser las primeras `ntotal`."""
    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    out_dir = build_and_persist(records, encoder, out_dir=tmp_path / "encoder_test")

    with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "doc_id": "img01", "chunk_id": "img01-c0000",
                    "fuente": "foto.jpg", "formato": "jpg", "fenomeno": 2,
                    "posicion": 0, "num_tokens": 0, "texto": "", "en_indice": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    index, metadata = load_index(encoder.name, index_dir=out_dir)
    assert index.ntotal == len(records)
    assert len(metadata) == len(records) + 1
    assert metadata[-1]["chunk_id"] == "img01-c0000"


def test_load_index_rechaza_filas_metadata_only_intercaladas(tmp_path):
    """Una fila sin vector antes de una con vector rompe el mapeo id->metadata
    y tiene que abortar, no arriesgar la cascada."""
    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    out_dir = build_and_persist(records, encoder, out_dir=tmp_path / "encoder_test")

    lineas = (out_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    fila_extra = json.dumps(
        {
            "doc_id": "img01", "chunk_id": "img01-c0000",
            "fuente": "foto.jpg", "formato": "jpg", "fenomeno": 2,
            "posicion": 0, "num_tokens": 0, "texto": "", "en_indice": False,
        },
        ensure_ascii=False,
    )
    # Intercalada en el medio, no al final.
    lineas.insert(2, fila_extra)
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(lineas) + "\n", encoding="utf-8", newline="\n"
    )

    try:
        load_index(encoder.name, index_dir=out_dir)
        assert False, "se esperaba ValueError por fila sin vector intercalada"
    except ValueError:
        pass


def test_checkpoint_de_codificacion_reanuda_si_los_chunks_no_cambiaron(tmp_path):
    """El .npy se graba por bloques para poder retomar una corrida de horas."""
    from src.embedding.build_index import encode_records

    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    cache = tmp_path / "emb.npy"

    primera = encode_records(records, encoder, cache_path=cache)
    assert cache.with_suffix(".progreso").exists()

    segunda = encode_records(records, encoder, cache_path=cache)  # ya completo
    assert (primera == segunda).all()


def test_checkpoint_se_descarta_si_los_chunks_vienen_en_otro_orden(tmp_path):
    """La forma del .npy no detecta un reordenamiento: los mismos N chunks en
    otro orden dan la misma forma, y las filas ya codificadas corresponderian a
    otros textos. El indice saldria corrupto en silencio despues de horas de
    CPU, asi que el checkpoint guarda tambien el chunk_id de la ultima fila."""
    from src.embedding.build_index import encode_records

    encoder = HashingFakeEncoder(name="test-encoder")
    records = _make_records()
    cache = tmp_path / "emb.npy"

    encode_records(records, encoder, cache_path=cache)

    revueltos = list(reversed(records))
    vectores = encode_records(revueltos, encoder, cache_path=cache)

    # Si el cache viejo se hubiera reutilizado, la fila 0 tendria el vector del
    # primer record original en vez del que ahora ocupa esa posicion.
    esperado = encoder.encode_passages([revueltos[0].texto])[0]
    assert (vectores[0] == esperado).all()
