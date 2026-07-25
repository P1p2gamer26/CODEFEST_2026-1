"""Construccion e integracion del grafo de conocimiento (sec. 7.2, paso 3;
sec. 7.3). Cada arista conserva `doc_id` y `chunk_id` de origen para poder
rastrear la evidencia textual de cada relacion."""

from pathlib import Path

import networkx as nx

from ..ingestion.pipeline import ChunkRecord
from .relations import extract_triples


def build_knowledge_graph(records: list[ChunkRecord]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for record in records:
        for subject, relation, obj in extract_triples(record.texto, record.idioma):
            if not subject or not obj:
                continue
            graph.add_node(subject, label=subject)
            graph.add_node(obj, label=obj)
            graph.add_edge(
                subject,
                obj,
                relation=relation,
                doc_id=record.doc_id,
                chunk_id=record.chunk_id,
                weight=1,
            )

    return graph


def export_graphml(graph: nx.MultiDiGraph, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GraphML solo admite atributos escalares (str/int/float/bool); todos los
    # atributos usados aqui (relation, doc_id, chunk_id, weight, label) ya lo son.
    nx.write_graphml(graph, out_path)
