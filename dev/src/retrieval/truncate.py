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
from .calidad_chunk import fraccion_aparato
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


# Idiomas que el evaluador de ADL puede leer. El corpus trae informes de
# SIPRI y SWF traducidos al coreano, ruso, arabe, chino y aleman, y sus chunks
# compiten por los mismos 10 cupos: 45 de los 500 fragmentos entregados salian
# en un idioma ilegible, y q006 gastaba 5 de sus 10 cupos asi.
IDIOMAS_LEGIBLES = frozenset({"es", "en", "pt"})

# A partir de que fraccion de aparato bibliografico se considera que el
# fragmento no le dice nada al evaluador. Barrido sobre los 500 fragmentos de
# la entrega, modelando que ADL puntua 0 un fragmento asi: 0.50 y 0.60 dan
# exactamente el mismo resultado (ningun fragmento cae entre ambos), 0.75
# degrada 3 menos y gana un poco menos, 0.90 se queda corto. Se toma 0.60 por
# ser el mas conservador de los dos empatados.
UMBRAL_APARATO = 0.60


def ordenar_para_fragmentos(
    hits: list[Hit],
    doc_ids_prioritarios: list[str] | None = None,
    priorizar_idioma: bool = True,
    degradar_aparato: bool = True,
) -> list[Hit]:
    """Reordena los hits ANTES de armar los fragmentos. No filtra nada.

    Dos criterios, en este orden:

    1. **Alineacion con los documentos entregados.** `build_result_object`
       calculaba las dos mitades de la respuesta por caminos independientes:
       los documentos por agregacion del pool, los fragmentos por score crudo
       de chunk. Resultado medido sobre las 41 consultas anotadas a mano:
       **solo el 31% de los fragmentos venia de los 3 documentos que la propia
       respuesta declaraba mas relevantes**, con una mediana de 9 documentos
       distintos entre los 10 fragmentos. La respuesta se contradecia a si
       misma, y un fragmento de un documento que uno mismo dejo fuera del
       top-3 es casi seguro un cero en NDCG@10.
    2. **Idioma legible**, dentro de cada grupo (ver IDIOMAS_LEGIBLES). Es un
       post-filtro por metadata, que la sec. 8.7 autoriza; no interviene
       ningun modelo generativo.
    3. **No ser aparato bibliografico.** La sec. 10.2.1 dice que la relevancia
       del fragmento se juzga sobre su campo `text`, asi que un chunk que es
       una lista de notas al pie puntua 0 aunque venga de un documento
       relevante. Medido sobre la entrega actual: 11 de 500 fragmentos son
       bibliografia, dos de ellos en rank 2 y 3.

       EFECTO PEQUENO Y MEDIDO: +0.003 de NDCG@10 modelando que el evaluador
       les da 0. Se aplica porque no puede perder (solo mueve casos claros al
       fondo de los mismos 10) y cuesta cero en CPU, no porque mueva la aguja.
       El proxy binario de eval_mini no puede ver esta mejora -- le da 1 a la
       bibliografia de un documento relevante --, por eso se justifica contra
       el reglamento y se mide con `ndcg_penalizado`.

    Reordena, NO descarta: si el top-3 no tiene 10 chunks, los de mas abajo
    completan igual, y si todos los chunks de una consulta fueran ilegibles se
    entregan esos. Asi el esquema de la sec. 1.4 (exactamente 10 fragmentos)
    no puede romperse por este cambio.

    `sorted` es estable, asi que el ultimo criterio de desempate es el orden
    de entrada, o sea el score. No hace falta pasarlo explicito.
    """
    if not doc_ids_prioritarios and not priorizar_idioma and not degradar_aparato:
        return hits
    top = set(doc_ids_prioritarios or ())

    def clave(hit: Hit) -> tuple[int, int, int]:
        fuera_del_top = 1 if (top and hit.doc_id not in top) else 0
        ilegible = 1 if (priorizar_idioma and hit.idioma not in IDIOMAS_LEGIBLES) else 0
        aparato = (
            1
            if (degradar_aparato and fraccion_aparato(hit.texto) >= UMBRAL_APARATO)
            else 0
        )
        return (fuera_del_top, ilegible, aparato)

    return sorted(hits, key=clave)


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
