"""Vinculacion del grafo de conocimiento con la recuperacion vectorial
(sec. 8.5): NER sobre la consulta -> nodos coincidentes -> vecinos de primer
orden -> pool de candidatos rankeado por evidencia, listo para fusionarse
con los resultados de FAISS via Reciprocal Rank Fusion
(src/retrieval/fusion.py), tratando el grafo como un indice adicional."""

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from .ner import extract_entities


@dataclass
class GraphHit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str


def desempatar_con_grafo(hits: list, graph_hits: list[GraphHit]) -> list:
    """Integra el grafo en la recuperacion SIN desplazar candidatos (sec. 8.5).

    En vez de fusionar los vecinos del grafo como una lista mas (medido sobre
    las 50 oficiales: F1@3 0.455 -> 0.358, NDCG@10 0.516 -> 0.390; 11-0 de
    perdidas), el grafo entra como clave secundaria de orden: reordena los
    hits que el pool vectorial ya trae, rompiendo SOLO empates exactos de
    score por la evidencia de primer orden. El conjunto del pool no cambia,
    asi que los documentos que entran al top-3 solo pueden variar donde el
    recuperador los tenia exactamente empatados (el caso de q005/q026).

    `sorted` es estable: dos hits con el mismo score y la misma evidencia
    conservan su orden original.
    """
    if not graph_hits:
        return hits
    evidencia = {gh.chunk_id: gh.score for gh in graph_hits}
    return sorted(
        hits,
        key=lambda h: (h.score, evidencia.get(h.chunk_id, 0.0)),
        reverse=True,
    )


def _normalize(text: str) -> str:
    return text.strip().lower()


def graph_search(
    query: str, graph: nx.MultiDiGraph, lang: str | None = None, k: int = 10
) -> list[GraphHit]:
    query_entities = {_normalize(e.text) for e in extract_entities(query, lang)}
    if not query_entities:
        return []

    node_by_normalized = {_normalize(n): n for n in graph.nodes}
    matched_nodes = [node_by_normalized[e] for e in query_entities if e in node_by_normalized]
    if not matched_nodes:
        return []

    evidence_count: dict[tuple[str, str], int] = defaultdict(int)
    for node in matched_nodes:
        for _, _, data in graph.out_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1
        for _, _, data in graph.in_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1

    ranked = sorted(evidence_count.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        GraphHit(rank=i, score=float(count), doc_id=doc_id, chunk_id=chunk_id)
        for i, ((doc_id, chunk_id), count) in enumerate(ranked, start=1)
    ]
