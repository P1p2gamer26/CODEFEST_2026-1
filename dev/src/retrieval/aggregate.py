"""Agregacion de fragmentos al nivel de documento (sec. 8.6). Opera solo
sobre las puntuaciones numericas de FAISS, sin intervencion de modelos
generativos."""

from collections import defaultdict
from dataclasses import dataclass

from .search import Hit

AggregationStrategy = str  # "max" | "sum" | "mean" | "topM" (M entero: top2, top3...)


@dataclass
class DocumentHit:
    rank: int
    doc_id: str
    score: float


def filtrar_por_fenomeno_dominante(
    hits: list[Hit], umbral: float = 0.0
) -> list[Hit]:
    """Descarta del pool los fragmentos que no son del fenomeno dominante.

    La sec. 8.7 permite post-filtrar por metadata, y el fenomeno es la senal
    mas barata que hay: **37 de las 41 consultas anotadas tienen TODOS sus
    documentos relevantes en un solo fenomeno**, y sin filtrar se gastan 17 de
    123 cupos (14%) en documentos del fenomeno equivocado.

    El fenomeno se decide por VOTO PONDERADO POR SCORE del propio pool, no por
    conteo: un fragmento en rank 1 debe pesar mas que uno en rank 60.

    `umbral` es la fraccion del score total que el fenomeno ganador debe
    superar para que se aplique el filtro. Por debajo de eso **se abstiene** y
    devuelve el pool intacto: con voto simple el filtro ganaba 4 consultas
    pero perdia 2, y las que perdia son justamente aquellas donde el pool esta
    repartido y el ganador no es de fiar. `umbral=0` filtra siempre (voto
    simple); `umbral=1` no filtra nunca.
    """
    if not hits:
        return hits

    peso: dict[int, float] = defaultdict(float)
    for hit in hits:
        if hit.fenomeno is not None:
            peso[hit.fenomeno] += max(hit.score, 0.0)
    if not peso:
        return hits

    total = sum(peso.values())
    ganador, mejor = max(peso.items(), key=lambda kv: kv[1])
    if total <= 0 or mejor / total < umbral:
        return hits

    filtrados = [h for h in hits if h.fenomeno == ganador]
    return filtrados or hits


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
    elif strategy.startswith("top") and strategy[3:].isdigit():
        # "sum" no tiene tope, asi que un documento con 30 chunks mediocres en el
        # pool le gana a uno con un solo chunk en rank 1. Medido con
        # scripts/diagnostico_ceros.py: 15 de las 17 consultas con F1=0 tienen el
        # documento correcto DENTRO del pool, varias con un chunk en rank 1-3.
        # topM suma solo los M mejores chunks del documento: conserva el premio a
        # tener varios pasajes relevantes (top1 == max) sin premiar el inundar.
        m = int(strategy[3:])
        agg_scores = {
            doc_id: sum(sorted(scores, reverse=True)[:m]) for doc_id, scores in scores_by_doc.items()
        }
    else:
        raise ValueError(f"estrategia de agregacion desconocida: {strategy}")

    ranked = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        DocumentHit(rank=i, doc_id=doc_id, score=score)
        for i, (doc_id, score) in enumerate(ranked, start=1)
    ]
