"""Agregacion de fragmentos al nivel de documento (sec. 8.6). Opera solo
sobre las puntuaciones numericas de FAISS, sin intervencion de modelos
generativos."""

from collections import defaultdict
from dataclasses import dataclass

from .search import Hit

AggregationStrategy = str  # "max" | "sum" | "mean"


@dataclass
class DocumentHit:
    rank: int
    doc_id: str
    score: float


def aggregate_documents(
    hits: list[Hit], top_n: int = 3, strategy: AggregationStrategy = "max"
) -> list[DocumentHit]:
    scores_by_doc: dict[str, list[float]] = defaultdict(list)
    for hit in hits:
        scores_by_doc[hit.doc_id].append(hit.score)

    if strategy == "max":
        agg_scores = {doc_id: max(scores) for doc_id, scores in scores_by_doc.items()}
    elif strategy == "sum":
        agg_scores = {doc_id: sum(scores) for doc_id, scores in scores_by_doc.items()}
    elif strategy == "mean":
        agg_scores = {doc_id: sum(scores) / len(scores) for doc_id, scores in scores_by_doc.items()}
    else:
        raise ValueError(f"estrategia de agregacion desconocida: {strategy}")

    ranked = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        DocumentHit(rank=i, doc_id=doc_id, score=score)
        for i, (doc_id, score) in enumerate(ranked, start=1)
    ]
