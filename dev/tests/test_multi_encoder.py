"""Multi-encoder (sec. 4.4) y fusion RRF de varios indices (sec. 8.4).

La invariante que protegen estas pruebas: todos los indices deben compartir
exactamente los mismos chunks. RRF fusiona por `chunk_id` y la metadata se
lee de un solo `metadata.jsonl`, asi que si cada encoder fragmentara por su
cuenta (con su propio tokenizer) los chunk_id colisionarian apuntando a
textos distintos y el resultado seria silenciosamente incorrecto.
"""

from src.embedding.encoders import HashingFakeEncoder, get_encoder
from src.retrieval.fusion import rebuild_hits_from_fusion, reciprocal_rank_fusion
from src.retrieval.search import Hit


def _hit(chunk_id, texto):
    return Hit(
        rank=0,
        score=0.0,
        chunk_id=chunk_id,
        doc_id="docA",
        fuente="docA.pdf",
        texto=texto,
        formato="pdf",
        fenomeno=1,
        idioma="es",
    )


def _meta(chunk_id, texto):
    return {
        "chunk_id": chunk_id,
        "doc_id": "docA",
        "fuente": "docA.pdf",
        "texto": texto,
        "formato": "pdf",
        "fenomeno": 1,
        "idioma": "es",
    }


def _tiny_records():
    from src.ingestion.pipeline import ChunkRecord

    textos = [
        "Los satelites en orbita baja enfrentan riesgos de colision.",
        "La cooperacion internacional mitiga la congestion orbital.",
        "La inteligencia artificial se aplica en logistica militar.",
    ]
    return [
        ChunkRecord(
            doc_id="docA",
            chunk_id=f"docA-c{i:04d}",
            fuente="docA.pdf",
            formato="pdf",
            fenomeno=1,
            posicion=i,
            num_tokens=len(t.split()),
            texto=t,
            idioma="es",
        )
        for i, t in enumerate(textos)
    ]


def test_both_encoder_indices_share_the_same_chunk_ids(tmp_path):
    """La invariante de la que depende toda la fusion: dos indices
    construidos en la misma corrida (fragmentando una sola vez) tienen los
    mismos chunk_id, en el mismo orden. Si esto se rompe, generador.py
    aborta -- ver la validacion al cargar los indices."""
    from src.embedding.build_index import build_and_persist, load_index

    records = _tiny_records()
    dir_a = tmp_path / "encoder_a"
    dir_b = tmp_path / "encoder_b"

    # Mismos records, dos encoders distintos: exactamente lo que hace
    # scripts/build_corpus_index.py al recibir varios --encoder-name.
    build_and_persist(records, HashingFakeEncoder(name="encoder-a"), out_dir=dir_a)
    build_and_persist(records, HashingFakeEncoder(name="encoder-b"), out_dir=dir_b)

    _, meta_a = load_index("encoder-a", index_dir=dir_a)
    _, meta_b = load_index("encoder-b", index_dir=dir_b)

    ids_a = [m["chunk_id"] for m in meta_a]
    ids_b = [m["chunk_id"] for m in meta_b]
    assert ids_a == ids_b, "los indices deben compartir los mismos chunk_id en el mismo orden"
    assert len(set(ids_a)) == len(ids_a), "los chunk_id deben ser unicos"
    # Y el texto de cada chunk_id debe coincidir entre ambos indices.
    for ma, mb in zip(meta_a, meta_b):
        assert ma["texto"] == mb["texto"]


def test_different_tokenizers_would_desync_chunk_ids(tmp_path):
    """Demuestra POR QUE hay que fragmentar una sola vez: con presupuestos de
    tokens distintos, el mismo documento produce cantidades distintas de
    chunks, y los chunk_id (doc_id + posicion) dejan de referirse al mismo
    texto entre indices."""
    from src.chunking.chunker import chunk_document

    texto = " ".join(
        f"Esta es la oracion numero {i} del documento de prueba." for i in range(40)
    )
    pocos = chunk_document(texto, formato="pdf", lang="es", count_tokens=lambda t: len(t.split()))
    muchos = chunk_document(
        texto, formato="pdf", lang="es", count_tokens=lambda t: len(t.split()) * 4
    )

    assert len(pocos) != len(muchos), (
        "distinto conteo de tokens deberia dar distinta fragmentacion; "
        "por eso el chunking no puede depender del encoder cuando hay varios"
    )


def test_two_fake_encoders_produce_different_rankings():
    """Si dos encoders dieran siempre el mismo ranking, fusionarlos no
    aportaria nada -- y las pruebas de fusion no probarian nada."""
    a = HashingFakeEncoder(name="encoder-a")
    b = HashingFakeEncoder(name="encoder-b")
    vec_a = a.encode_one("riesgos de colision orbital")
    vec_b = b.encode_one("riesgos de colision orbital")
    assert vec_a.shape == vec_b.shape
    assert not (vec_a == vec_b).all()


def test_fusion_rebuilds_text_from_metadata_not_from_the_item():
    """El texto devuelto debe corresponder al chunk_id, siempre. Se relee de
    la metadata en vez de confiar en el item representativo de la fusion,
    porque ese item pudo venir de otra lista."""
    metadata_by_chunk_id = {
        "c1": _meta("c1", "texto correcto de c1"),
        "c2": _meta("c2", "texto correcto de c2"),
    }
    # La lista del "segundo encoder" trae el texto equivocado a proposito.
    lista_1 = [_hit("c1", "texto correcto de c1")]
    lista_2 = [_hit("c2", "TEXTO CORRUPTO"), _hit("c1", "TEXTO CORRUPTO")]

    fused = reciprocal_rank_fusion([lista_1, lista_2], key=lambda h: h.chunk_id)
    hits = rebuild_hits_from_fusion(fused, metadata_by_chunk_id, limit=10)

    assert {h.chunk_id for h in hits} == {"c1", "c2"}
    for h in hits:
        assert h.texto == metadata_by_chunk_id[h.chunk_id]["texto"]
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_fusion_drops_items_without_known_metadata():
    """Un chunk_id que el grafo reporte pero que no este en el indice no debe
    llegar a resultados.jsonl con texto vacio: se descarta."""
    metadata_by_chunk_id = {"c1": _meta("c1", "texto de c1")}
    fused = reciprocal_rank_fusion(
        [[_hit("c1", "texto de c1")], [_hit("fantasma", "no indexado")]],
        key=lambda h: h.chunk_id,
    )
    hits = rebuild_hits_from_fusion(fused, metadata_by_chunk_id, limit=10)
    assert [h.chunk_id for h in hits] == ["c1"]


def test_fusion_respects_limit():
    metadata_by_chunk_id = {f"c{i}": _meta(f"c{i}", f"texto {i}") for i in range(10)}
    fused = reciprocal_rank_fusion(
        [[_hit(f"c{i}", f"texto {i}") for i in range(10)]],
        key=lambda h: h.chunk_id,
    )
    assert len(rebuild_hits_from_fusion(fused, metadata_by_chunk_id, limit=3)) == 3


def test_encoder_prefixes_are_applied_when_declared():
    """Los encoders de la familia E5 exigen prefijos distintos para consulta y
    pasaje; omitirlos degrada la calidad en silencio."""

    class _SpyEncoder(HashingFakeEncoder):
        def __init__(self):
            super().__init__(name="spy")
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "
            self.seen: list[str] = []

        def encode(self, texts):
            self.seen.extend(texts)
            return super().encode(texts)

        def encode_passages(self, texts):
            return self.encode([self.passage_prefix + t for t in texts])

        def encode_query(self, text):
            return self.encode_one(self.query_prefix + text)

    enc = _SpyEncoder()
    enc.encode_passages(["fragmento indexado"])
    enc.encode_query("consulta del usuario")

    assert "passage: fragmento indexado" in enc.seen
    assert "query: consulta del usuario" in enc.seen


def test_default_encoder_has_no_prefixes():
    """El encoder primario no usa prefijos: la interfaz asimetrica no debe
    alterar su comportamiento."""
    enc = get_encoder(name="cualquiera", use_fake=True)
    texto = "un fragmento"
    assert (enc.encode_passages([texto])[0] == enc.encode([texto])[0]).all()
    assert (enc.encode_query(texto) == enc.encode_one(texto)).all()
