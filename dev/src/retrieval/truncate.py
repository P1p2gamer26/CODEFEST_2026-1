"""Aplica el limite de 250 palabras por fragmento (sec. 9.2.1). Si un chunk
recuperado supera el limite, se divide en sub-fragmentos que respetan
limites oracionales completos (nunca se corta una oracion) y comparten el
mismo `chunk_id` de origen, cada uno con su propio `rank` en la lista final
(sec. 9.2.1 y 9.3.1). Simplificacion documentada: no se implementa la fusion
opcional de chunks cortos adyacentes del mismo documento (el reglamento la
describe como posible, no obligatoria); ver limitaciones en
informe_tecnico.pdf.
"""

from ..chunking.sentence_split import split_sentences
from ..config import MAX_FRAGMENT_WORDS, N_FRAGMENTS_PER_QUERY
from .search import Hit


def _split_oversized(hit: Hit, max_words: int) -> list[dict]:
    sentences = split_sentences(hit.texto, hit.idioma)
    if not sentences:
        sentences = [hit.texto]

    # Una sola "oracion" puede superar por si sola el limite: pasa con texto
    # de OCR o de PDF mal extraido, donde no queda puntuacion aprovechable y
    # el splitter devuelve un bloque entero. Ahi el corte por palabras es la
    # unica salida; sin esto el fragmento sale con mas de 250 palabras y la
    # sec. 9.3.2 lo penaliza o lo descarta.
    trozos: list[str] = []
    for sent in sentences:
        palabras = sent.split()
        if len(palabras) > max_words:
            trozos += [
                " ".join(palabras[i : i + max_words])
                for i in range(0, len(palabras), max_words)
            ]
        else:
            trozos.append(sent)
    sentences = trozos

    sub_fragments: list[dict] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current and current_words + sent_words > max_words:
            sub_fragments.append(
                {"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": " ".join(current)}
            )
            current, current_words = [], 0
        current.append(sent)
        current_words += sent_words

    if current:
        sub_fragments.append(
            {"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": " ".join(current)}
        )
    return sub_fragments


def enforce_word_limit(
    hits: list[Hit],
    max_fragments: int = N_FRAGMENTS_PER_QUERY,
    max_words: int = MAX_FRAGMENT_WORDS,
) -> list[dict]:
    fragments: list[dict] = []

    for hit in hits:
        if len(fragments) >= max_fragments:
            break

        n_words = len(hit.texto.split())
        if n_words <= max_words:
            fragments.append({"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": hit.texto})
        else:
            for sub in _split_oversized(hit, max_words):
                if len(fragments) >= max_fragments:
                    break
                fragments.append(sub)

    fragments = fragments[:max_fragments]
    for i, frag in enumerate(fragments, start=1):
        frag["rank"] = i
    return fragments
