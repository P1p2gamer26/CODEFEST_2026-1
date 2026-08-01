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
    hits: list[Hit], top_n: int = 3, strategy: AggregationStrategy = "sum"
) -> list[DocumentHit]:
    """Por defecto "sum", no "max": un documento relevante para una consulta
    suele tener VARIOS pasajes relevantes, mientras que "max" premia al que
    tiene un unico chunk afortunado.

    La eleccion se apoya en ese argumento, NO en la medicion. Sobre dev/eval/
    (41 consultas) sum promedia 0.306 y max 0.226, pero contando por consulta
    el reparto es 16-8 con 17 empates: prueba de signos p=0.15, y p=0.69 sobre
    las 10 consultas de anotacion independiente. Sum gana en ambas muestras y
    no pierde en ninguna, pero NO esta demostrado. Verificable con
    scripts/eval_mini.py --comparar-con."""
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
