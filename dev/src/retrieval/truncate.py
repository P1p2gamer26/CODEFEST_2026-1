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


def _clave_dedup(texto: str) -> str:
    """Normaliza solo espacios y mayusculas: la comparacion sigue siendo de
    texto exacto. Deliberadamente NO se hace similitud difusa, que podria
    descartar fragmentos distintos que comparten un parrafo."""
    return " ".join(texto.split()).lower()


def enforce_word_limit(
    hits: list[Hit],
    max_fragments: int = N_FRAGMENTS_PER_QUERY,
    max_words: int = MAX_FRAGMENT_WORDS,
) -> list[dict]:
    """Los 10 fragmentos de la consulta, sin repetir texto.

    El corpus de ADL trae documentos duplicados (el mismo informe de CSIS bajo
    dos doc_id, series que reeditan capitulos enteros), asi que dos chunks
    distintos pueden tener texto identico. Entregar el mismo texto dos veces
    no puede sumar en NDCG@10 -- el ranking ideal no lo contiene -- y ademas
    desplaza fuera del top-10 a un candidato que si podria sumar. Medido sobre
    las 50 consultas oficiales: 17 de 500 fragmentos eran duplicados exactos."""
    fragments: list[dict] = []
    vistos: set[str] = set()

    for hit in hits:
        if len(fragments) >= max_fragments:
            break

        n_words = len(hit.texto.split())
        if n_words <= max_words:
            candidatos = [{"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": hit.texto}]
        else:
            candidatos = _split_oversized(hit, max_words)

        for sub in candidatos:
            if len(fragments) >= max_fragments:
                break
            clave = _clave_dedup(sub["text"])
            if clave in vistos:
                continue
            vistos.add(clave)
            fragments.append(sub)

    fragments = fragments[:max_fragments]
    for i, frag in enumerate(fragments, start=1):
        frag["rank"] = i
    return fragments
