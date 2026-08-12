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


# -- desempatar_con_grafo: integracion no-desplazante (sec. 8.5) -------------


def _hit(chunk_id, score):
    from src.retrieval.search import Hit

    return Hit(rank=1, score=score, chunk_id=chunk_id, doc_id=chunk_id.split("-c")[0],
               fuente="f.pdf", texto="x", formato="pdf", fenomeno=1, idioma="es")


def test_desempate_no_desplaza_quienes_no_estan_empatados():
    """El grafo solo puede reordenar empates exactos de score: con scores
    distintos el orden queda intacto aunque la evidencia favorezca a otro."""
    from src.graph.graph_retrieval import GraphHit, desempatar_con_grafo

    hits = [_hit("d1-c0", 0.9), _hit("d2-c0", 0.8), _hit("d3-c0", 0.7)]
    graph_hits = [GraphHit(rank=1, score=3.0, chunk_id="d3-c0", doc_id="d3")]
    ordenado = desempatar_con_grafo(hits, graph_hits)
    assert [h.chunk_id for h in ordenado] == ["d1-c0", "d2-c0", "d3-c0"]


def test_desempate_rompe_empates_exactos_por_evidencia():
    from src.graph.graph_retrieval import GraphHit, desempatar_con_grafo

    hits = [_hit("d1-c0", 0.5), _hit("d2-c0", 0.5), _hit("d3-c0", 0.5)]
    graph_hits = [
        GraphHit(rank=1, score=5.0, chunk_id="d2-c0", doc_id="d2"),
        GraphHit(rank=2, score=2.0, chunk_id="d1-c0", doc_id="d1"),
    ]
    ordenado = desempatar_con_grafo(hits, graph_hits)
    assert [h.chunk_id for h in ordenado] == ["d2-c0", "d1-c0", "d3-c0"]


def test_desempate_sin_evidencia_es_inerte_y_estable():
    """El grafo no puede desplazar a nadie: sin evidencia (o con evidencia
    cero) el orden de entrada se conserva porque `sorted` es estable."""
    from src.graph.graph_retrieval import GraphHit, desempatar_con_grafo

    hits = [_hit("d1-c0", 0.5), _hit("d2-c0", 0.5)]
    assert [h.chunk_id for h in desempatar_con_grafo(hits, [])] == ["d1-c0", "d2-c0"]

    graph_hits = [GraphHit(rank=1, score=0.0, chunk_id="d2-c0", doc_id="d2")]
    assert [h.chunk_id for h in desempatar_con_grafo(hits, graph_hits)] == ["d1-c0", "d2-c0"]


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


def test_export_graphml_escapa_comillas_en_nombres_de_entidad(tmp_path):
    """saxutils.escape no toca la comilla doble, y los nombres de entidad van
    dentro de atributos XML entrecomillados. El grafo del corpus real tiene 160
    nodos con comilla en el nombre: sin escaparla el GraphML sale corrupto y ni
    networkx lo relee."""
    import networkx as nx
    from src.graph.build_graph import export_graphml

    g = nx.MultiDiGraph()
    g.add_node('Operacion "Libertad"', label='Operacion "Libertad"', tipo="ORG")
    g.add_node("ONU", label="ONU", tipo="ORG")
    g.add_edge('Operacion "Libertad"', "ONU", relation='dijo "algo"',
               doc_id="D1", chunk_id="D1-c0", weight=1)

    out = tmp_path / "g.graphml"
    export_graphml(g, out)

    releido = nx.read_graphml(out)
    assert 'Operacion "Libertad"' in releido.nodes


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
