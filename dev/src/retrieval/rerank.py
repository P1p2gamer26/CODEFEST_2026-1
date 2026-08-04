"""Cascada de dos encoders: uno recupera, el otro re-puntua (sec. 4.4).

Por que cascada y no fusion simetrica. La fusion RRF de dos listas premia el
ACUERDO entre ellas, y medido sobre las 50 consultas MiniLM y e5 comparten
apenas el 11,3% de los documentos del top-3 y el 6,2% de los fragmentos del
top-10. Con ese desacuerdo, RRF no fusiona: intercala la lista buena con la
mala y el resultado queda a medio camino (0,306 -> 0,268).

Lo que e5 hace mal es el RECALL -- su propio top-60 rara vez contiene el
documento correcto. Puntuar candidatos que el primer encoder ya encontro es un
trabajo distinto y mas facil. La cascada conserva el recall del primario y usa
al secundario solo donde puede aportar.

El truco que lo hace barato es el invariante del chunking unico: los dos
indices se construyen sobre los MISMOS records en el MISMO orden, asi que el
vector del chunk de la fila `i` se lee del otro indice con `reconstruct(i)`,
sin volver a codificar ni un pasaje. El costo por consulta es una sola
vectorizacion de la consulta.
"""

import faiss
import numpy as np

from .search import Hit


def rerank_por_segundo_encoder(
    hits: list[Hit],
    index_secundario: faiss.Index,
    vector_consulta: np.ndarray,
    peso: float,
) -> list[Hit]:
    """Reordena `hits` mezclando su score con el del encoder secundario.

        score = cos_primario + peso * cos_secundario

    Los dos terminos son cosenos (ambos encoders normalizan, ver
    embedding/encoders.py), asi que estan en la misma escala y sumarlos es
    legitimo -- a diferencia de mezclar con BM25, donde no lo era.

    `peso = 0` devuelve el mismo orden de entrada: es la forma de comprobar que
    la cascada no altera nada por si misma.

    Los hits sin fila conocida (`fila < 0`: vienen del grafo o de una fusion
    previa, no de FAISS) conservan su score primario sin re-puntuar, en vez de
    descartarse: perder candidatos seria peor que no mejorarlos.
    """
    if not hits:
        return []

    consulta = np.asarray(vector_consulta, dtype="float32").reshape(-1)

    puntuados: list[tuple[float, Hit]] = []
    for hit in hits:
        score = hit.score
        if peso and hit.fila >= 0:
            vector = index_secundario.reconstruct(hit.fila)
            score += peso * float(np.dot(consulta, vector))
        puntuados.append((score, hit))

    # sorted es estable, asi que ante empates se conserva el orden del primario.
    puntuados.sort(key=lambda par: par[0], reverse=True)

    reordenados = []
    for rank, (score, hit) in enumerate(puntuados, start=1):
        reordenados.append(
            Hit(
                rank=rank,
                score=score,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                fuente=hit.fuente,
                texto=hit.texto,
                formato=hit.formato,
                fenomeno=hit.fenomeno,
                idioma=hit.idioma,
                fila=hit.fila,
            )
        )
    return reordenados


def verificar_alineacion(metadata_a: list[dict], metadata_b: list[dict]) -> None:
    """Aborta si los dos indices no describen los mismos chunks en el mismo
    orden. Sin esto la cascada re-puntuaria el chunk equivocado en silencio,
    que es el peor final posible: resultados plausibles y mal fundados."""
    if len(metadata_a) != len(metadata_b):
        raise ValueError(
            f"los indices tienen distinto numero de chunks: {len(metadata_a)} vs {len(metadata_b)}. "
            "Se construyeron con chunkings distintos y no son comparables por fila."
        )
    for i, (a, b) in enumerate(zip(metadata_a, metadata_b)):
        if a["chunk_id"] != b["chunk_id"]:
            raise ValueError(
                f"los indices divergen en la fila {i}: {a['chunk_id']!r} vs {b['chunk_id']!r}. "
                "El chunking debe hacerse UNA sola vez para todos los encoders."
            )
