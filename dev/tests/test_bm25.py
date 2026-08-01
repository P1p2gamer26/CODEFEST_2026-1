"""BM25 tiene que premiar el termino raro, no el frecuente. Es la parte que
puede romperse en silencio: un idf mal calculado o una normalizacion por
longitud invertida siguen devolviendo resultados, solo que peores.
"""

from src.retrieval.bm25 import BM25, tokenize


def _meta(i: int, texto: str) -> dict:
    return {
        "chunk_id": f"DOC-{i}::0",
        "doc_id": f"DOC-{i}",
        "fuente": f"doc{i}.pdf",
        "texto": texto,
        "formato": "pdf",
        "fenomeno": 1,
        "idioma": "es",
    }


RELLENO = "las fuerzas armadas participan en el ejercicio conjunto de la region"


def test_termino_raro_gana():
    metadata = [_meta(i, RELLENO) for i in range(20)]
    metadata.append(_meta(99, "informe sobre amenazas NBQR en el ejercicio conjunto"))
    bm25 = BM25(metadata)

    hits = bm25.search("amenazas NBQR", k=3)

    assert hits[0].doc_id == "DOC-99"
    assert hits[0].rank == 1


def test_termino_en_todos_no_discrimina():
    """Un termino presente en todos los documentos tiene idf ~0: no debe
    inventarse un ganador con puntuacion alta."""
    metadata = [_meta(i, RELLENO) for i in range(20)]
    bm25 = BM25(metadata)

    hits = bm25.search("ejercicio", k=5)

    assert len(hits) == 5
    assert all(h.score < 0.5 for h in hits)


def test_documento_corto_gana_a_largo_con_igual_frecuencia():
    """Normalizacion por longitud: la misma aparicion pesa mas en un chunk
    corto que en uno diluido."""
    metadata = [_meta(i, RELLENO) for i in range(10)]
    metadata.append(_meta(50, "NBQR"))
    metadata.append(_meta(51, "NBQR " + RELLENO * 10))
    bm25 = BM25(metadata)

    hits = bm25.search("NBQR", k=2)

    assert [h.doc_id for h in hits] == ["DOC-50", "DOC-51"]


def test_consulta_sin_terminos_conocidos():
    bm25 = BM25([_meta(0, RELLENO)])
    assert bm25.search("xyzzy", k=3) == []


def test_tokenize_baja_y_separa():
    assert tokenize("Fuerzas, Armadas del PAIS-2024") == ["fuerzas", "armadas", "del", "pais", "2024"]
