"""Pruebas del grafo de conocimiento (bonus): construccion, roundtrip
GraphML y vinculacion con la recuperacion (sec. 7 y 8.5)."""

import networkx as nx

from src.graph.build_graph import export_graphml
from src.graph.graph_retrieval import graph_search

FIXTURE_TRIPLES = [
    ("OTAN", "coopera_con", "Union Europea", "doc1", "doc1-c0000"),
    ("Union Europea", "financia", "investigacion en inteligencia artificial", "doc1", "doc1-c0001"),
    ("CEPAL", "publica", "informe de desarrollo territorial", "doc2", "doc2-c0000"),
]


def _build_fixture_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for subj, rel, obj, doc_id, chunk_id in FIXTURE_TRIPLES:
        graph.add_node(subj, label=subj)
        graph.add_node(obj, label=obj)
        graph.add_edge(subj, obj, relation=rel, doc_id=doc_id, chunk_id=chunk_id, weight=1)
    return graph


def test_graphml_roundtrip_preserves_nodes_and_edges(tmp_path):
    graph = _build_fixture_graph()
    out_path = tmp_path / "grafo.graphml"

    export_graphml(graph, out_path)
    assert out_path.exists()

    reloaded = nx.read_graphml(out_path)
    assert reloaded.number_of_nodes() == graph.number_of_nodes()
    assert reloaded.number_of_edges() == graph.number_of_edges()
    assert set(reloaded.nodes) == set(graph.nodes)


def test_graph_search_finds_first_order_neighbors_evidence():
    graph = _build_fixture_graph()
    hits = graph_search("Que hace la OTAN junto con la Union Europea?", graph, lang="es")

    assert hits, "se esperaba al menos un chunk candidato del grafo"
    chunk_ids = {h.chunk_id for h in hits}
    assert "doc1-c0000" in chunk_ids


def test_graph_search_returns_empty_when_no_entities_match():
    graph = _build_fixture_graph()
    hits = graph_search("una consulta totalmente ajena sin entidades conocidas", graph, lang="es")
    assert hits == []
