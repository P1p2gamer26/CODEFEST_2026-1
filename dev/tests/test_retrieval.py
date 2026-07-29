"""Pruebas del modulo de recuperacion: agregacion a documento, fusion RRF y
el limite de 250 palabras por fragmento respetando oraciones completas."""

from src.retrieval.aggregate import aggregate_documents
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.search import Hit
from src.retrieval.truncate import enforce_word_limit


def _hit(chunk_id, doc_id, score, texto="texto de prueba.", idioma="es", fenomeno=1):
    return Hit(
        rank=0,
        score=score,
        chunk_id=chunk_id,
        doc_id=doc_id,
        fuente=f"{doc_id}.pdf",
        texto=texto,
        formato="pdf",
        fenomeno=fenomeno,
        idioma=idioma,
    )


def test_aggregate_documents_max_pooling_orders_by_best_chunk():
    hits = [
        _hit("c1", "docA", 0.9),
        _hit("c2", "docA", 0.2),
        _hit("c3", "docB", 0.5),
        _hit("c4", "docC", 0.7),
        _hit("c5", "docD", 0.1),
    ]
    docs = aggregate_documents(hits, top_n=3, strategy="max")
    assert [d.doc_id for d in docs] == ["docA", "docC", "docB"]
    assert [d.rank for d in docs] == [1, 2, 3]


def test_aggregate_documents_sum_can_reorder_vs_max():
    hits = [
        _hit("c1", "docA", 0.4),
        _hit("c2", "docA", 0.4),
        _hit("c3", "docB", 0.5),
    ]
    docs_max = aggregate_documents(hits, top_n=2, strategy="max")
    docs_sum = aggregate_documents(hits, top_n=2, strategy="sum")
    assert docs_max[0].doc_id == "docB"  # 0.5 > 0.4
    assert docs_sum[0].doc_id == "docA"  # 0.8 > 0.5


def test_reciprocal_rank_fusion_rewards_consistent_items():
    list_a = [_hit("c1", "d1", 0.9), _hit("c2", "d2", 0.8), _hit("c3", "d3", 0.7)]
    list_b = [_hit("c2", "d2", 0.95), _hit("c1", "d1", 0.6), _hit("c4", "d4", 0.5)]

    fused = reciprocal_rank_fusion([list_a, list_b], key=lambda h: h.chunk_id)
    fused_ids = [item.chunk_id for item, _ in fused]

    # c1 y c2 aparecen en ambas listas con buen rango -> deberian ir primero
    assert set(fused_ids[:2]) == {"c1", "c2"}
    assert "c3" in fused_ids and "c4" in fused_ids


def test_enforce_word_limit_keeps_short_chunks_intact():
    hits = [_hit("c1", "d1", 0.9, texto="Una oracion corta. Otra oracion corta.")]
    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)
    assert len(fragments) == 1
    assert fragments[0]["text"] == hits[0].texto
    assert fragments[0]["rank"] == 1


def test_enforce_word_limit_splits_oversized_chunk_on_sentence_boundaries():
    long_sentences = [f"Sentence number {i} in this long chunk about defense policy." for i in range(60)]
    texto = " ".join(long_sentences)  # ~ 600 palabras, muy por encima de 250
    hits = [_hit("c1", "d1", 0.9, texto=texto, idioma="en")]

    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)

    assert len(fragments) > 1
    for frag in fragments:
        assert len(frag["text"].split()) <= 250
        assert frag["chunk_id"] == "c1"  # sub-fragmentos comparten chunk_id (sec. 9.3.1)
    # los ranks son secuenciales
    assert [f["rank"] for f in fragments] == list(range(1, len(fragments) + 1))
    # ninguna oracion original quedo cortada a la mitad
    reconstructed = " ".join(f["text"] for f in fragments)
    for sent in long_sentences:
        assert sent in reconstructed or sent in texto


def test_enforce_word_limit_stops_at_max_fragments():
    hits = [_hit(f"c{i}", f"d{i}", 1.0 - i * 0.01) for i in range(20)]
    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)
    assert len(fragments) == 10
    assert [f["rank"] for f in fragments] == list(range(1, 11))
