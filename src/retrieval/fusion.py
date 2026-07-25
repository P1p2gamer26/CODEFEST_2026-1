"""Reciprocal Rank Fusion (RRF), sec. 8.4 y ecuacion (7). Combina varias
listas ya ordenadas (multiples encoders, o vector+grafo con el grafo tratado
como un "indice" adicional, sec. 8.5) sin usar puntuaciones absolutas ni
ningun modelo generativo -- solo la posicion (rank) de cada item en cada
lista."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TypeVar

from ..config import RRF_K0

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
