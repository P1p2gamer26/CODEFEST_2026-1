"""Modulo de recuperacion: consulta -> vector -> busqueda FAISS -> metadata
(sec. 8.1). Ninguna etapa usa un modelo generativo; solo vectores,
puntuaciones de similitud coseno y metadata (sec. 8.3).
"""

from dataclasses import dataclass

import faiss

from ..config import OVERFETCH_FACTOR
from ..embedding.encoders import Encoder


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    fuente: str
    texto: str
    formato: str
    fenomeno: int | None
    idioma: str | None


def search(
    query: str,
    encoder: Encoder,
    index: faiss.Index,
    metadata: list[dict],
    k: int = 10,
    fenomeno: int | None = None,
    formato: str | None = None,
    idioma: str | None = None,
    min_score: float | None = None,
    overfetch_factor: int = OVERFETCH_FACTOR,
) -> list[Hit]:
    """Busca los `k` fragmentos mas relevantes para `query`.

    Post-filtros (sec. 8.7) operan directamente sobre metadata (`fenomeno`,
    `formato`, `idioma`) o sobre el score de similitud coseno (`min_score`),
    nunca via un modelo generativo. Se sobre-recupera `k * overfetch_factor`
    candidatos de FAISS antes de filtrar, ya que los filtros pueden
    descartar algunos de los top-k crudos.
    """
    # encode_query (no encode_one) para que los encoders que lo requieran
    # apliquen su prefijo de consulta -- ver src/embedding/encoders.py.
    query_vec = encoder.encode_query(query).reshape(1, -1)
    fetch_k = min(index.ntotal, max(k * overfetch_factor, k))
    scores, ids = index.search(query_vec, fetch_k)

    hits: list[Hit] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        if fenomeno is not None and meta.get("fenomeno") != fenomeno:
            continue
        if formato is not None and meta.get("formato") != formato:
            continue
        if idioma is not None and meta.get("idioma") != idioma:
            continue
        if min_score is not None and score < min_score:
            continue

        hits.append(
            Hit(
                rank=0,  # se asigna abajo, tras aplicar todos los filtros
                score=float(score),
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                fuente=meta["fuente"],
                texto=meta["texto"],
                formato=meta["formato"],
                fenomeno=meta.get("fenomeno"),
                idioma=meta.get("idioma"),
            )
        )
        if len(hits) >= k:
            break

    for i, hit in enumerate(hits, start=1):
        hit.rank = i
    return hits
