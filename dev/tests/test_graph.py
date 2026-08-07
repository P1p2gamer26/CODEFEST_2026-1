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


def test_limpiar_entidad_descarta_parentesis_desbalanceados():
    """Cuando el span de spaCy no coincide con un parentesis del texto original
    queda un caracter colgando (ej. "Instituto Kroc) Objetivo"). Se recorta el
    parentesis colgante del borde y se descarta lo que sigue desbalanceado."""
    from src.graph.ner import _limpiar_entidad

    assert _limpiar_entidad("Instituto Kroc) Objetivo") is None
    assert _limpiar_entidad("Naciones Unidas)") == "Naciones Unidas"
    assert _limpiar_entidad("(OTAN") == "OTAN"
    assert _limpiar_entidad("(ONU)") == "ONU"
    # El ')' de "Cooperacion (ONU)" cierra un parentesis interno: no se toca.
    assert _limpiar_entidad("Cooperacion (ONU)") == "Cooperacion (ONU)"
    assert _limpiar_entidad("(") is None


def test_limpiar_entidad_acota_la_longitud():
    """La etiqueta MISC del esquema CoNLL ocasionalmente captura clausulas
    completas: se admiten entidades de hasta 6 palabras / 60 caracteres en vez
    de descartar la etiqueta MISC por completo."""
    from src.graph.ner import _limpiar_entidad

    assert _limpiar_entidad("a" * 61) is None
    assert _limpiar_entidad("a" * 60) == "a" * 60
    assert _limpiar_entidad("una " * 7) is None
    assert _limpiar_entidad("una " * 6) == "una una una una una una"


def test_build_knowledge_graph_agrega_peso_por_chunk():
    """La misma relacion (sujeto, relacion, objeto) en varios chunks se agrega
    en una sola arista con weight = numero de chunks y chunk_ids con la lista
    completa de evidencia, conservando la trazabilidad puntual de la sec. 7.2."""
    from src.graph.build_graph import build_knowledge_graph
    from src.ingestion.pipeline import ChunkRecord

    def _rec(i, texto):
        return ChunkRecord(doc_id=f"D{i}", chunk_id=f"D{i}-c0", fuente="f.pdf", formato="pdf",
                           fenomeno=1, posicion=0, num_tokens=20, texto=texto, idioma="es")

    textos = [
        "La OTAN coopera con Colombia y firmo un acuerdo en Bruselas.",
        "La OTAN coopera con Colombia para fortalecer su presencia.",
    ]
    graph = build_knowledge_graph([_rec(i, t) for i, t in enumerate(textos)])

    aristas = [d for u, v, d in graph.edges(data=True) if (u, d["relation"], v) == ("OTAN", "cooperar", "Colombia")]
    assert len(aristas) == 1, "la tripleta repetida debe agregarse en una sola arista"
    assert aristas[0]["weight"] == 2
    assert aristas[0]["chunk_ids"] == "D0-c0;D1-c0"
    assert aristas[0]["doc_id"] == "D0"
    assert aristas[0]["chunk_id"] == "D0-c0"
    assert graph.nodes["OTAN"]["tipo"] in {"ORG", "MISC"}


def test_export_graphml_escribe_chunk_ids_y_peso(tmp_path):
    """El esquema de aristas del GraphML con estilo expone weight (numero de
    chunks de respaldo) y chunk_ids (evidencia completa), y se mantiene
    releible por networkx."""
    import networkx as nx
    from src.graph.build_graph import export_graphml

    g = nx.MultiDiGraph()
    g.add_node("OTAN", label="OTAN", tipo="ORG")
    g.add_node("Colombia", label="Colombia", tipo="LOC")
    g.add_edge("OTAN", "Colombia", relation="cooperar", doc_id="D0", chunk_id="D0-c0",
               chunk_ids="D0-c0;D1-c0", weight=2)

    out = tmp_path / "g.graphml"
    export_graphml(g, out)

    releido = nx.read_graphml(out)
    u, v, d = next(iter(releido.edges(data=True)))
    assert d["relation"] == "cooperar"
    assert d["weight"] == 2
    assert d["chunk_ids"] == "D0-c0;D1-c0"


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
