"""Cascada de dos encoders. La logica no es trivial (mezcla de scores + lectura
del vector por fila en OTRO indice) y se rompe en silencio: si la fila apunta
al chunk equivocado los resultados siguen siendo plausibles."""

import faiss
import numpy as np
import pytest

from src.retrieval.rerank import rerank_por_segundo_encoder, verificar_alineacion
from src.retrieval.search import Hit


def _hit(chunk_id, doc_id, score, fila):
    return Hit(
        rank=0, score=score, chunk_id=chunk_id, doc_id=doc_id,
        fuente=f"{doc_id}.pdf", texto="texto", formato="pdf",
        fenomeno=1, idioma="es", fila=fila,
    )


def _indice(vectores: list[list[float]]) -> faiss.Index:
    arr = np.array(vectores, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


def test_peso_cero_no_altera_el_orden():
    hits = [_hit("c1", "A", 0.9, 0), _hit("c2", "B", 0.5, 1), _hit("c3", "C", 0.3, 2)]
    index = _indice([[1, 0], [0, 1], [1, 1]])

    salida = rerank_por_segundo_encoder(hits, index, np.array([1.0, 0.0]), peso=0.0)

    assert [h.chunk_id for h in salida] == ["c1", "c2", "c3"]
    assert [h.rank for h in salida] == [1, 2, 3]


def test_el_secundario_puede_remontar_a_un_candidato():
    """El caso que justifica toda la cascada: un chunk que el primario dejo
    tercero sube porque el secundario lo puntua muy alto."""
    hits = [_hit("c1", "A", 0.60, 0), _hit("c2", "B", 0.55, 1), _hit("c3", "C", 0.50, 2)]
    # En el espacio del secundario, la consulta [1,0] solo apunta a la fila 2.
    index = _indice([[0, 1], [0, 1], [1, 0]])

    salida = rerank_por_segundo_encoder(hits, index, np.array([1.0, 0.0]), peso=1.0)

    assert salida[0].chunk_id == "c3"
    assert salida[0].score == pytest.approx(1.50)


def test_un_peso_pequeno_no_alcanza_para_remontar():
    """El peso gradua cuanta autoridad se le da al secundario; con 0,1 la
    ventaja de 0,10 del primario sobrevive."""
    hits = [_hit("c1", "A", 0.60, 0), _hit("c3", "C", 0.50, 2)]
    index = _indice([[0, 1], [0, 1], [1, 0]])

    salida = rerank_por_segundo_encoder(hits, index, np.array([1.0, 0.0]), peso=0.1)

    assert salida[0].chunk_id == "c1"


def test_los_hits_sin_fila_se_conservan_sin_repuntuar():
    """Los hits del grafo no vienen de FAISS y no tienen fila; descartarlos
    seria perder candidatos."""
    hits = [_hit("c1", "A", 0.9, -1), _hit("c2", "B", 0.1, 0)]
    index = _indice([[1, 0], [0, 1]])

    salida = rerank_por_segundo_encoder(hits, index, np.array([1.0, 0.0]), peso=1.0)

    assert {h.chunk_id for h in salida} == {"c1", "c2"}
    assert next(h for h in salida if h.chunk_id == "c1").score == pytest.approx(0.9)


def test_verificar_alineacion_detecta_indices_incompatibles():
    a = [{"chunk_id": "D1-c0"}, {"chunk_id": "D1-c1"}]

    with pytest.raises(ValueError, match="fila 1"):
        verificar_alineacion(a, [{"chunk_id": "D1-c0"}, {"chunk_id": "OTRO"}])

    with pytest.raises(ValueError, match="distinto numero"):
        verificar_alineacion(a, [{"chunk_id": "D1-c0"}])


def test_search_expone_la_fila_del_indice(tmp_path):
    """La cascada depende de que `fila` sea la fila real de FAISS."""
    from src.embedding.build_index import build_and_persist, load_index
    from src.embedding.encoders import HashingFakeEncoder
    from src.ingestion.pipeline import ChunkRecord
    from src.retrieval.search import search

    encoder = HashingFakeEncoder(name="test-encoder")
    records = [
        ChunkRecord(doc_id=f"d{i}", chunk_id=f"d{i}-c0", fuente="f.pdf", formato="pdf",
                    fenomeno=1, posicion=0, num_tokens=5, texto=f"texto numero {i}")
        for i in range(5)
    ]
    out = build_and_persist(records, encoder, out_dir=tmp_path / "enc")
    index, metadata = load_index(encoder.name, index_dir=out)

    hits = search("texto numero 3", encoder, index, metadata, k=5)

    assert all(metadata[h.fila]["chunk_id"] == h.chunk_id for h in hits)
