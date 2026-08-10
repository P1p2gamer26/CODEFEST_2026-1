"""Aplica el limite de 250 palabras por fragmento (sec. 9.2.1). Si un chunk
recuperado supera el limite, se divide en sub-fragmentos que respetan
limites oracionales completos (nunca se corta una oracion) y comparten el
mismo `chunk_id` de origen, cada uno con su propio `rank` en la lista final
(sec. 9.2.1 y 9.3.1). Simplificacion documentada: no se implementa la fusion
opcional de chunks cortos adyacentes del mismo documento (el reglamento la
describe como posible, no obligatoria); ver limitaciones en
informe_tecnico.pdf.
"""

import re

from ..chunking.sentence_split import split_sentences
from ..config import MAX_FRAGMENT_WORDS, N_FRAGMENTS_PER_QUERY
from .calidad_chunk import fraccion_aparato
from .glosario import _normalizar as _sin_acentos
from .search import Hit

# E23. Palabras vacias de espanol y de ingles: sin ellas, "sobre", "entre" o
# "which" harian que casi cualquier pasaje "cubra" la consulta y el criterio
# seria inerte. No es una lista exhaustiva ni pretende serlo -- se fijo antes
# de medir y no se retoco despues, que es lo que la vuelve honesta.
STOPWORDS = frozenset("""
para como cual cuales cuanto cuantos donde desde entre sobre segun hacia hasta
ante bajo cabe contra durante mediante salvo tras esta este estos estas esos
esas aquel aquella ellos ellas nosotros vosotros pero sino aunque porque
cuando mientras siempre nunca tambien tanto tanta menos mismo misma otro otra
otros otras cada todo toda todos todas algun alguna algunos algunas ningun
ninguna mucho mucha muchos muchas poco poca pocos pocas
that this these those with from have been they their there where which what
when while about into over under between during through after before more
most such than then them your ours will would could should because being
""".split())


def tokens_de(texto: str) -> set[str]:
    """Palabras de contenido: sin acentos, de 4 letras o mas, sin vacias.

    El umbral de 4 caracteres deja fuera las siglas de 2-3 letras, pero esas
    ya llegan por la via del encoder; lo que este criterio busca es si el
    pasaje menciona el OBJETO de la consulta, no afinar la coincidencia.
    """
    return {w for w in re.findall(r"[a-z0-9]{4,}", _sin_acentos(texto)) if w not in STOPWORDS}


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
    tokens_consulta: frozenset[str] | None = None,
) -> list[Hit]:
    """Reordena los hits ANTES de armar los fragmentos. No filtra nada.

    Cuatro criterios, en este orden:

    0. **Idioma legible, POR ENCIMA de la alineacion** (E22). Ver el punto 2:
       el orden entre estos dos no es cosmetico y se midio.
    1. **Alineacion con los documentos entregados.** `build_result_object`
       calculaba las dos mitades de la respuesta por caminos independientes:
       los documentos por agregacion del pool, los fragmentos por score crudo
       de chunk. Resultado medido sobre las 41 consultas anotadas a mano:
       **solo el 31% de los fragmentos venia de los 3 documentos que la propia
       respuesta declaraba mas relevantes**, con una mediana de 9 documentos
       distintos entre los 10 fragmentos. La respuesta se contradecia a si
       misma, y un fragmento de un documento que uno mismo dejo fuera del
       top-3 es casi seguro un cero en NDCG@10.
    2. **Idioma legible** (ver IDIOMAS_LEGIBLES). Es un post-filtro por
       metadata, que la sec. 8.7 autoriza; no interviene ningun modelo
       generativo.

       **E22 (9 ago 2026): este criterio va ANTES que la alineacion, y el
       cambio se midio.** Con la alineacion primero, un fragmento en coreano
       del documento nº 1 le ganaba a uno legible del documento nº 4. Para un
       evaluador que lee espanol/ingles el primero vale **r=0 con certeza**;
       el segundo tiene probabilidad no nula de valer mas. Los 19 fragmentos
       ilegibles de la entrega salian de solo DOS documentos (`F3-SIPRI-002`
       y `F3-SIPRI-100`, traducciones al coreano 100% ilegibles), y en las 5
       consultas afectadas habia documentos del top-3 completamente legibles
       de donde sacar el reemplazo. Resultado: **ilegibles 19 -> 0**, NDCG@10
       **+0.007 [+0.001, +0.015]** (3 victorias, 0 derrotas), F1@3 sin mover
       una milesima, y la alineacion baja de 499/500 a 480/500 -- exactamente
       los 19 intercambiados y ni uno mas.

       Se pre-registro la prediccion de que "el proxy va a mostrar una
       perdida" y **no se cumplio**: los documentos coreanos resultaron ser
       los irrelevantes en 4 de las 5 consultas. Queda anotado porque la
       prediccion era nuestra y el dato la corrigio.

    4. **Cobertura lexica de la consulta** (E23), como ultimo desempate. La
       sec. 10.2.1 juzga la relevancia sobre el campo `text` y con escala
       graduada: un pasaje del documento correcto que no menciona el objeto de
       la consulta en ninguno de sus dos idiomas entro por el TEMA del
       documento, no por responder. Medido: **175 de 500 fragmentos entregados
       no contenian ni una palabra de contenido de la consulta expandida**, y
       bajan a 138. NDCG penalizado **+0.009 [-0.001, +0.021]**.

       Se probo tambien la version GRADUADA (ordenar por numero de palabras en
       comun) y **se descarto**: misma cobertura final, efecto **-0.001**, y
       mueve 355 de 500 fragmentos. Mucha rotacion para nada; la binaria mueve
       159.

       Salvedad que va junto al numero: **la evidencia es enteramente de las
       50** y su IC cruza el cero. Pasa por el criterio de dano, no por
       demostrar ganancia.
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
    toks = tokens_consulta or frozenset()

    def clave(hit: Hit) -> tuple[int, int, int, int]:
        fuera_del_top = 1 if (top and hit.doc_id not in top) else 0
        ilegible = 1 if (priorizar_idioma and hit.idioma not in IDIOMAS_LEGIBLES) else 0
        aparato = (
            1
            if (degradar_aparato and fraccion_aparato(hit.texto) >= UMBRAL_APARATO)
            else 0
        )
        sin_cobertura = 0 if (toks and (tokens_de(hit.texto) & toks)) else 1
        # E22: `ilegible` va PRIMERO, antes que `fuera_del_top`.
        return (ilegible, fuera_del_top, aparato, sin_cobertura)

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
