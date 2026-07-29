"""Reciprocal Rank Fusion (RRF), sec. 8.4 y ecuacion (7). Combina varias
listas ya ordenadas (multiples encoders, o vector+grafo con el grafo tratado
como un "indice" adicional, sec. 8.5) sin usar puntuaciones absolutas ni
ningun modelo generativo -- solo la posicion (rank) de cada item en cada
lista.

RRF se elige sobre CombSUM/CombMNZ porque es robusto a que dos encoders
produzcan puntuaciones en escalas distintas: solo mira posiciones. Eso lo
hace la opcion natural cuando se fusionan espacios vectoriales diferentes.

INVARIANTE: fusionar por `chunk_id` solo es correcto si todos los indices
comparten los mismos chunks. Por eso `scripts/build_corpus_index.py`
fragmenta UNA sola vez y reutiliza esos records para todos los encoders.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

from ..config import RRF_K0

if TYPE_CHECKING:
    from .search import Hit

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    key: Callable[[T], str],
    k0: int = RRF_K0,
) -> list[tuple[T, float]]:
    """Cada lista en `ranked_lists` debe venir ya ordenada de mayor a menor
    relevancia (rank 1 = primer elemento). `key` extrae el identificador
    (p. ej. chunk_id) usado para reconocer el mismo item entre listas.

    Devuelve pares (item, score_rrf) ordenados de mayor a menor score_rrf;
    el item devuelto es su primera aparicion entre las listas fusionadas.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    representative: dict[str, T] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_key = key(item)
            rrf_scores[item_key] += 1.0 / (k0 + rank)
            representative.setdefault(item_key, item)

    ordered_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)
    return [(representative[k_], rrf_scores[k_]) for k_ in ordered_keys]


def rebuild_hits_from_fusion(
    fused: Sequence[tuple[object, float]],
    metadata_by_chunk_id: dict[str, dict],
    limit: int,
) -> list["Hit"]:
    """Reconstruye `Hit`s a partir del resultado de `reciprocal_rank_fusion`.

    Los items fusionados pueden venir de fuentes distintas (otro encoder, el
    grafo), asi que el texto y la metadata se releen SIEMPRE desde
    `metadata_by_chunk_id` en vez de confiar en el item representativo: es lo
    que garantiza que el texto devuelto corresponda de verdad al chunk_id
    reportado. Los items sin metadata conocida se descartan.
    """
    from .search import Hit  # import local: evita un ciclo search <-> fusion

    hits: list[Hit] = []
    for item, score in fused:
        if len(hits) >= limit:
            break
        meta = metadata_by_chunk_id.get(getattr(item, "chunk_id", None))
        if meta is None:
            continue
        hits.append(
            Hit(
                rank=len(hits) + 1,
                score=score,
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                fuente=meta["fuente"],
                texto=meta["texto"],
                formato=meta["formato"],
                fenomeno=meta.get("fenomeno"),
                idioma=meta.get("idioma"),
            )
        )
    return hits
