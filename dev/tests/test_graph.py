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


def test_graph_ignora_formatos_sin_narrativa():
    """NER sobre una fila de CSV o los atributos de una tesela produce basura:
    en el corpus real las entidades mas frecuentes eran FALSO, VERDADEIRO y el
    nombre de una capa de mapa."""
    from src.graph.build_graph import build_knowledge_graph
    from src.ingestion.pipeline import ChunkRecord

    def _rec(formato, texto):
        return ChunkRecord(
            doc_id="d1", chunk_id="d1-c0", fuente="f.csv", formato=formato,
            fenomeno=1, posicion=0, num_tokens=10, texto=texto, idioma="es",
        )

    tabular = [_rec("csv", "estado: VERDADERO, activo: FALSO, region: Bogota")]
    assert build_knowledge_graph(tabular).number_of_nodes() == 0
    assert build_knowledge_graph([_rec("pbf", "capa: au_seg_marq, valor: 1")]).number_of_nodes() == 0


def test_graph_limpia_caracteres_no_validos_en_xml():
    """El texto de OCR trae bytes de control que hacen abortar a
    nx.write_graphml al final de horas de NER."""
    from src.graph.build_graph import _limpiar

    assert _limpiar("Naciones\x00Unidas") == "NacionesUnidas"
    assert _limpiar("Corte\nSuprema  de\tJusticia") == "Corte Suprema de Justicia"
    assert _limpiar("\x0b\x1f") == ""


def _grafo_de_prueba():
    from src.graph.build_graph import build_knowledge_graph
    from src.ingestion.pipeline import ChunkRecord

    textos = [
        "La OTAN firmo un acuerdo con Colombia en Bruselas.",
        "Naciones Unidas presento un informe sobre Colombia.",
    ]
    return build_knowledge_graph([
        ChunkRecord(doc_id=f"D{i}", chunk_id=f"D{i}-c0", fuente="f.pdf", formato="pdf",
                    fenomeno=1, posicion=0, num_tokens=20, texto=t, idioma="es")
        for i, t in enumerate(textos)
    ])


def test_export_graphml_con_estilo_es_releible(tmp_path):
    """El export con layout escribe el XML a mano (formato yEd); si se rompe,
    networkx deja de poder releerlo y el bonus se entrega corrupto."""
    import networkx as nx
    from src.graph.build_graph import export_graphml

    g = _grafo_de_prueba()
    out = tmp_path / "g.graphml"
    export_graphml(g, out)

    releido = nx.read_graphml(out)
    assert releido.number_of_nodes() == g.number_of_nodes()
    assert all("x" in d and "tipo" in d for _, d in releido.nodes(data=True))


def test_export_graphml_omite_el_layout_si_el_grafo_es_grande(tmp_path):
    """El layout compara todas las parejas de nodos: con los 224.101 del corpus
    real no terminaria nunca. Por encima del tope se exporta plano."""
    import networkx as nx
    from src.graph.build_graph import export_graphml

    g = _grafo_de_prueba()
    out = tmp_path / "g.graphml"
    export_graphml(g, out, max_nodos_layout=1)

    releido = nx.read_graphml(out)
    assert releido.number_of_nodes() == g.number_of_nodes()  # sigue completo
    assert not any("x" in d for _, d in releido.nodes(data=True))  # pero sin layout
