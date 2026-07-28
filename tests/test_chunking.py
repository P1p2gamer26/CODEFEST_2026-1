"""Verifica el requisito obligatorio de completitud linguistica (sec. 3.3):
ningun chunk puede contener una oracion cortada a la mitad."""

from src.chunking import chunk_document, split_sentences
from src.chunking.chunker import split_into_sections

EN_PARAGRAPH = (
    "Artificial intelligence is reshaping defense planning. Analysts now rely "
    "on machine learning models to process satellite imagery. These models "
    "reduce the time needed to review reconnaissance footage. However, bias "
    "in training data remains a serious concern for operational reliability. "
    "Independent testing protocols are increasingly required before "
    "deployment. Allied nations are discussing shared governance standards "
    "for these systems. Workforce shortages remain the binding constraint on "
    "faster adoption across smaller defense ministries."
)

ES_PARAGRAPH = (
    "La congestion de la orbita baja terrestre ha aumentado de forma "
    "sostenida. Las grandes constelaciones de satelites explican buena parte "
    "de ese crecimiento. Cada colision genera miles de fragmentos nuevos. "
    "Estos fragmentos permanecen en orbita durante decadas. La cooperacion "
    "internacional es clave para mitigar este riesgo. Las directrices de "
    "desintegracion post-mision no se cumplen de forma pareja entre todos los "
    "operadores."
)


def _assert_chunks_are_sentence_aligned(texto: str, lang: str, formato: str = "html") -> list:
    reference_sentences = split_sentences(texto, lang)
    chunks = chunk_document(texto, formato=formato, lang=lang, token_budget=25, overlap_sentences=1)

    assert chunks, "se esperaban chunks para un parrafo no vacio"

    joined = " ".join(reference_sentences)
    for chunk in chunks:
        # cada chunk debe ser exactamente la union de un tramo contiguo de
        # oraciones completas (con espacio simple como separador, igual que
        # el propio empaquetador) -- si el chunk cortara una oracion a
        # mitad, esta busqueda de subcadena fallaria.
        assert chunk.texto in joined, f"chunk no alineado a oraciones completas: {chunk.texto!r}"
        # cada oracion completa de referencia debe aparecer intacta o estar
        # totalmente ausente del chunk, nunca truncada.
        for sent in reference_sentences:
            if sent in chunk.texto:
                continue
            # si una porcion parcial de la oracion aparece, es un corte ilegal
            assert sent[:-5] not in chunk.texto or len(sent) < 5, (
                f"posible oracion cortada en chunk: {sent!r} vs {chunk.texto!r}"
            )
    return chunks


def test_no_sentence_is_split_across_chunks_en():
    chunks = _assert_chunks_are_sentence_aligned(EN_PARAGRAPH, "en")
    assert len(chunks) > 1, "el presupuesto de tokens pequeno deberia forzar varios chunks"


def test_no_sentence_is_split_across_chunks_es():
    chunks = _assert_chunks_are_sentence_aligned(ES_PARAGRAPH, "es")
    assert len(chunks) > 1


def test_positions_are_sequential_from_zero():
    chunks = chunk_document(EN_PARAGRAPH, formato="html", lang="en", token_budget=25, overlap_sentences=1)
    posiciones = [c.posicion for c in chunks]
    assert posiciones == list(range(len(chunks)))


def test_overlap_repeats_last_sentence_of_previous_chunk():
    chunks = chunk_document(EN_PARAGRAPH, formato="html", lang="en", token_budget=25, overlap_sentences=1)
    assert len(chunks) > 1
    reference_sentences = split_sentences(EN_PARAGRAPH, "en")
    for prev, curr in zip(chunks, chunks[1:]):
        prev_sentences = [s for s in reference_sentences if s in prev.texto]
        assert prev_sentences, "no se pudo mapear el chunk previo a oraciones de referencia"
        last_sentence_of_prev = prev_sentences[-1]
        assert curr.texto.startswith(last_sentence_of_prev), (
            "el chunk siguiente deberia empezar con la oracion de solape"
        )


def test_structural_split_respects_markdown_headers():
    texto = (
        "# Titulo\n\n"
        "## Introduccion\n\n"
        f"{EN_PARAGRAPH}\n\n"
        "## Conclusiones\n\n"
        f"{ES_PARAGRAPH}\n"
    )
    sections = split_into_sections(texto)
    headings = [h for h, _ in sections]
    assert "Introduccion" in headings
    assert "Conclusiones" in headings


def test_empty_document_produces_no_chunks():
    assert chunk_document("", formato="html", lang="en") == []
    assert chunk_document("   \n\n  ", formato="pdf", lang="es") == []


def test_tabular_format_yields_one_chunk_per_row():
    texto = "col_a: 1 | col_b: x\n\ncol_a: 2 | col_b: y\n\ncol_a: 3 | col_b: z"
    chunks = chunk_document(texto, formato="csv", lang=None)
    assert len(chunks) == 3
    assert [c.posicion for c in chunks] == [0, 1, 2]
    assert chunks[0].texto == "col_a: 1 | col_b: x"


def test_single_oversized_sentence_becomes_its_own_chunk():
    long_sentence = "This is a single sentence that by itself already exceeds the tiny token budget we set for this test."
    chunks = chunk_document(long_sentence, formato="html", lang="en", token_budget=3, overlap_sentences=0)
    assert len(chunks) == 1
    assert chunks[0].texto == long_sentence


def test_oversized_sentence_after_overlap_does_not_hang():
    """Regresion: si el overlap de la oracion anterior + la siguiente oracion
    ya excede el presupuesto, el empaquetador debia entrar en un loop
    infinito (nunca avanzaba `i`, seguia vaciando el mismo overlap para
    siempre). Reproducido con oraciones largas de informes tecnicos reales
    (ver corpus_real_ejemplo/); aqui se fuerza el mismo escenario con un
    presupuesto pequeno para que el test corra rapido y sin red."""
    texto = (
        "First sentence sets up context. "
        "Second sentence is deliberately long enough that even alone with the "
        "one-sentence overlap from the previous chunk it still cannot possibly "
        "fit inside the tiny token budget configured for this regression test."
    )
    chunks = chunk_document(texto, formato="html", lang="en", token_budget=5, overlap_sentences=1)
    assert len(chunks) >= 1
    assert sum(len(c.texto.split()) for c in chunks) >= len(texto.split())
