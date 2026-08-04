"""Pruebas del modulo de recuperacion: agregacion a documento, fusion RRF y
el limite de 250 palabras por fragmento respetando oraciones completas."""

from src.retrieval.aggregate import aggregate_documents, filtrar_por_fenomeno_dominante
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.search import Hit
from src.retrieval.truncate import enforce_word_limit, ordenar_para_fragmentos


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


def test_filtro_fenomeno_descarta_el_minoritario():
    hits = [_hit("c1", "docA", 0.9, fenomeno=2), _hit("c2", "docB", 0.8, fenomeno=2),
            _hit("c3", "docC", 0.1, fenomeno=3)]
    assert [h.doc_id for h in filtrar_por_fenomeno_dominante(hits, umbral=0.5)] == ["docA", "docB"]


def test_filtro_fenomeno_se_abstiene_si_el_pool_esta_repartido():
    # Ningun fenomeno llega al 90% del score: devuelve el pool intacto.
    hits = [_hit("c1", "docA", 0.5, fenomeno=1), _hit("c2", "docB", 0.5, fenomeno=2)]
    assert filtrar_por_fenomeno_dominante(hits, umbral=0.9) == hits


def test_filtro_fenomeno_pondera_por_score_no_por_conteo():
    # Tres fragmentos flojos del fenomeno 3 no le ganan a uno excelente del 2.
    hits = [_hit("c1", "docA", 0.95, fenomeno=2)] + [
        _hit(f"c{i}", "docB", 0.2, fenomeno=3) for i in range(2, 5)
    ]
    assert [h.doc_id for h in filtrar_por_fenomeno_dominante(hits)] == ["docA"]


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


def test_aggregate_documents_topm_acota_la_inundacion():
    """topM existe para el caso que diagnostico_ceros.py encontro: un documento
    con muchos chunks mediocres desplaza del top-3 a uno con un chunk excelente.
    top2 corta esa cola; sum no."""
    hits = [_hit("a", "docA", 0.9)] + [_hit(f"b{i}", "docB", 0.2) for i in range(10)]

    assert aggregate_documents(hits, top_n=1, strategy="sum")[0].doc_id == "docB"  # 2.0 > 0.9
    assert aggregate_documents(hits, top_n=1, strategy="top2")[0].doc_id == "docA"  # 0.9 > 0.4


def test_aggregate_documents_top1_equivale_a_max():
    hits = [_hit("c1", "docA", 0.4), _hit("c2", "docA", 0.4), _hit("c3", "docB", 0.5)]
    assert [d.doc_id for d in aggregate_documents(hits, top_n=2, strategy="top1")] == [
        d.doc_id for d in aggregate_documents(hits, top_n=2, strategy="max")
    ]


def test_reciprocal_rank_fusion_rewards_consistent_items():
    list_a = [_hit("c1", "d1", 0.9), _hit("c2", "d2", 0.8), _hit("c3", "d3", 0.7)]
    list_b = [_hit("c2", "d2", 0.95), _hit("c1", "d1", 0.6), _hit("c4", "d4", 0.5)]

    fused = reciprocal_rank_fusion([list_a, list_b], key=lambda h: h.chunk_id)
    fused_ids = [item.chunk_id for item, _ in fused]

    # c1 y c2 aparecen en ambas listas con buen rango -> deberian ir primero
    assert set(fused_ids[:2]) == {"c1", "c2"}
    assert "c3" in fused_ids and "c4" in fused_ids


def test_ordenar_para_fragmentos_pone_primero_los_del_top3():
    """Las dos mitades de la respuesta tienen que coincidir: entregar un
    fragmento de un documento que uno mismo dejo fuera del top-3 es casi
    seguro un cero en NDCG@10. Medido: solo el 31% de los fragmentos
    entregados venia de los 3 documentos declarados."""
    hits = [
        _hit("c1", "otro1", 0.99),
        _hit("c2", "top1", 0.50),
        _hit("c3", "otro2", 0.98),
        _hit("c4", "top2", 0.40),
    ]
    ordenado = ordenar_para_fragmentos(hits, doc_ids_prioritarios=["top1", "top2"])
    assert [h.chunk_id for h in ordenado] == ["c2", "c4", "c1", "c3"]


def test_ordenar_para_fragmentos_no_descarta_nada():
    """Reordena, NO filtra. Si el top-3 no tiene suficientes chunks, los de
    mas abajo completan igual: la sec. 1.4 exige EXACTAMENTE 10 fragmentos y
    este cambio no puede romper eso."""
    hits = [_hit("c1", "otro", 0.9), _hit("c2", "top", 0.8)]
    ordenado = ordenar_para_fragmentos(hits, doc_ids_prioritarios=["top"])
    assert len(ordenado) == 2
    assert {h.chunk_id for h in ordenado} == {"c1", "c2"}


def test_ordenar_para_fragmentos_manda_al_final_los_idiomas_ilegibles():
    """45 de los 500 fragmentos entregados salian en coreano, ruso, arabe,
    chino o aleman. El evaluador de ADL no los lee, asi que son ceros
    seguros; se posponen, no se descartan (sec. 8.7, filtro por metadata)."""
    hits = [_hit("c1", "d", 0.9, idioma="ko"), _hit("c2", "d", 0.5, idioma="es")]
    ordenado = ordenar_para_fragmentos(hits, priorizar_idioma=True)
    assert [h.chunk_id for h in ordenado] == ["c2", "c1"]
    # ...pero si TODOS son ilegibles se entregan igual, no se pierde el cupo
    solo_ko = [_hit("c1", "d", 0.9, idioma="ko")]
    assert len(ordenar_para_fragmentos(solo_ko, priorizar_idioma=True)) == 1


def test_ordenar_para_fragmentos_el_documento_manda_sobre_el_idioma():
    """El orden de los criterios importa: la alineacion con el top-3 es la
    ganancia medida (+0.114 de NDCG), el idioma es un desempate dentro del
    grupo. Un chunk legible de un documento descartado NO debe adelantar a
    uno ilegible del top-3."""
    hits = [_hit("c1", "otro", 0.9, idioma="es"), _hit("c2", "top", 0.8, idioma="ko")]
    ordenado = ordenar_para_fragmentos(hits, doc_ids_prioritarios=["top"], priorizar_idioma=True)
    assert [h.chunk_id for h in ordenado] == ["c2", "c1"]


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


def test_enforce_word_limit_corta_oracion_unica_gigante():
    """Texto de OCR sin puntuacion: el splitter devuelve UNA sola oracion de
    mas de 250 palabras. Antes se emitia entera y violaba la sec. 9.3.2."""
    texto = " ".join(f"palabra{i}" for i in range(400))  # sin un solo punto
    hits = [_hit("c1", "d1", 0.9, texto=texto, idioma="es")]

    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)

    assert len(fragments) == 2
    for frag in fragments:
        assert len(frag["text"].split()) <= 250
    # no se pierde ni se duplica texto
    assert " ".join(f["text"] for f in fragments) == texto


def test_enforce_word_limit_descarta_texto_duplicado():
    """El corpus trae el mismo informe bajo dos doc_id, asi que dos chunks
    distintos pueden tener texto identico. Repetirlo no suma en NDCG@10 y
    ademas desplaza a un candidato que si podria sumar."""
    hits = [
        _hit("c1", "d1", 0.9, texto="El mismo parrafo repetido."),
        _hit("c2", "d2", 0.8, texto="  EL MISMO   parrafo REPETIDO.  "),  # solo cambia formato
        _hit("c3", "d3", 0.7, texto="Un parrafo distinto."),
    ]

    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)

    assert len(fragments) == 2
    assert [f["doc_id"] for f in fragments] == ["d1", "d3"]  # se conserva el mejor rankeado
    assert [f["rank"] for f in fragments] == [1, 2]


def test_enforce_word_limit_rellena_el_cupo_tras_descartar_duplicados():
    """Descartar un duplicado no puede dejar la lista corta: hay que tomar el
    siguiente candidato del pool, que es justamente la ganancia buscada."""
    hits = [_hit("c1", "d1", 0.9, texto="Repetido.")]
    hits += [_hit("c2", "d2", 0.8, texto="Repetido.")]  # duplicado, se descarta
    hits += [_hit(f"c{i}", f"d{i}", 0.5, texto=f"Fragmento distinto numero {i}.") for i in range(3, 14)]

    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)

    assert len(fragments) == 10  # el cupo se completa igual
    assert len({f["text"].lower() for f in fragments}) == 10


def test_enforce_word_limit_stops_at_max_fragments():
    # Textos distintos a proposito: con el texto por defecto todos serian el
    # mismo fragmento y el dedup los colapsaria, tapando lo que prueba el caso.
    hits = [_hit(f"c{i}", f"d{i}", 1.0 - i * 0.01, texto=f"Fragmento numero {i}.") for i in range(20)]
    fragments = enforce_word_limit(hits, max_fragments=10, max_words=250)
    assert len(fragments) == 10
    assert [f["rank"] for f in fragments] == list(range(1, 11))
