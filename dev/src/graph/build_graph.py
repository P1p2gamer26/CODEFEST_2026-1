"""Construccion e integracion del grafo de conocimiento (sec. 7.2, paso 3;
sec. 7.3). Cada arista conserva `doc_id` y `chunk_id` de origen para poder
rastrear la evidencia textual de cada relacion.

`export_graphml()` escribe un GraphML con layout calculado (spring layout
determinista, seed fija) y en el formato yEd: posiciones, tamanos, colores por
tipo de entidad, etiquetas en nodos y relaciones en aristas. Asi el archivo se
ve bien (sin nodos superpuestos, con las relaciones legibles) tanto en yEd
como en Gephi, y `networkx.read_graphml` lo sigue leyendo igual (extrae x/y
del yfiles geometry de forma nativa)."""

from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import networkx as nx

from ..ingestion.pipeline import ChunkRecord
from .relations import extract_triples

# Etiquetas NER de spaCy (CoNLL en es/pt, OntoNotes en en) agrupadas en
# categorias legibles para colorear el grafo y que se entienda que tipo de
# entidad es cada nodo.
CATEGORIA_POR_TIPO = {
    "PER": "Persona",
    "PERSON": "Persona",
    "ORG": "Organizacion",
    "ORGANIZATION": "Organizacion",
    "LOC": "Lugar",
    "LOCATION": "Lugar",
    "GPE": "Lugar",
    "FAC": "Lugar",
}
COLOR_POR_CATEGORIA = {
    "Organizacion": "#4C78A8",
    "Persona": "#F58518",
    "Lugar": "#54A24B",
}
COLOR_MISC = "#B279A2"
COLOR_NODO = "#2F2F2F"
COLOR_ARISTA = "#8A8A8A"

ALTO_NODO = 34.0


def _tamano_nodo(graph: nx.Graph, nodo: str) -> tuple[float, float]:
    """Ancho/alto aproximados del nodo en pantalla, segun su etiqueta."""
    label = str(graph.nodes[nodo].get("label", nodo))
    ancho = min(240.0, max(70.0, 9.0 * len(label) + 34.0))
    return ancho, ALTO_NODO


def _layout_sin_solapes(
    graph: nx.Graph,
    seed: int,
    k: float = 1.8,
    iterations: int = 500,
    max_pasadas: int = 200,
) -> dict[str, tuple[float, float]]:
    """Spring layout determinista + pasada de separacion de rectangulos.

    El spring layout por si solo no garantiza que nodos de distinto tamano no
    se superpongan (asume nodos puntuales). Despues de calcularlo se empujan
    aparte las parejas cuyo rectangulo se solapa, separando por el eje con
    menor solapamiento para alterar lo menos posible el dibujo original.
    """
    pos = {n: [x, y] for n, (x, y) in nx.spring_layout(graph, seed=seed, k=k, iterations=iterations).items()}
    nodos = list(pos)

    def recta(n: str) -> tuple[float, float, float, float]:
        ancho, alto = _tamano_nodo(graph, n)
        return (pos[n][0] - ancho / 2, pos[n][1] - alto / 2, pos[n][0] + ancho / 2, pos[n][1] + alto / 2)

    for _ in range(max_pasadas):
        hubo_movimiento = False
        for i in range(len(nodos)):
            for j in range(i + 1, len(nodos)):
                a, b = nodos[i], nodos[j]
                ax0, ay0, ax1, ay1 = recta(a)
                bx0, by0, bx1, by1 = recta(b)
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > 0 and oy > 0:
                    hubo_movimiento = True
                    if ox <= oy:
                        direccion = 1.0 if pos[a][0] <= pos[b][0] else -1.0
                        pos[a][0] -= direccion * (ox / 2 + 1.0)
                        pos[b][0] += direccion * (ox / 2 + 1.0)
                    else:
                        direccion = 1.0 if pos[a][1] <= pos[b][1] else -1.0
                        pos[a][1] -= direccion * (oy / 2 + 1.0)
                        pos[b][1] += direccion * (oy / 2 + 1.0)
        if not hubo_movimiento:
            break
    return {n: (x, y) for n, (x, y) in pos.items()}


def build_knowledge_graph(records: list[ChunkRecord]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    tipo_counts: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        for subject, relation, obj, subj_tipo, obj_tipo in extract_triples(
            record.texto, record.idioma
        ):
            if not subject or not obj:
                continue
            graph.add_node(subject, label=subject)
            graph.add_node(obj, label=obj)
            tipo_counts[subject][subj_tipo] += 1
            tipo_counts[obj][obj_tipo] += 1
            graph.add_edge(
                subject,
                obj,
                relation=relation,
                doc_id=record.doc_id,
                chunk_id=record.chunk_id,
                weight=1,
            )

    for node in graph.nodes:
        # Si una entidad aparece con varias etiquetas NER, se usa la mas
        # frecuente (ej: "OTAN" como ORG en unas oraciones y MISC en otras).
        contador = tipo_counts[node]
        graph.nodes[node]["tipo"] = contador.most_common(1)[0][0] if contador else "MISC"

    return graph


def export_graphml(
    graph: nx.MultiDiGraph,
    out_path: Path,
    layout_seed: int = 42,
) -> None:
    """Escribe el grafo en GraphML con layout y estilo para visualizarlo.

    El layout es un spring layout determinista (seed fija -> reproducible):
    separa los nodos conectados entre si y evita que se superpongan. Se
    escriben a la vez las posiciones como atributos genericos x/y (los lee
    Gephi y networkx) y como geometria yfiles (los usa yEd), ademas de
    tamanos, color por tipo de entidad y etiquetas de relacion en cada arista.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if graph.number_of_nodes() == 0:
        nx.write_graphml(graph, out_path)
        return

    pos = _layout_sin_solapes(graph, layout_seed)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    escala = 2000.0 / span

    def posicion(nodo: str) -> tuple[float, float]:
        x, y = pos[nodo]
        return ((x - min(xs)) * escala + 40.0, (y - min(ys)) * escala + 40.0)

    def nodo_xml(nodo: str) -> str:
        label = str(graph.nodes[nodo].get("label", nodo))
        tipo = str(graph.nodes[nodo].get("tipo", "MISC"))
        grado = graph.degree(nodo)
        categoria = CATEGORIA_POR_TIPO.get(tipo, "Misc")
        color = COLOR_POR_CATEGORIA.get(categoria, COLOR_MISC)
        ancho, alto = _tamano_nodo(graph, nodo)
        cx, cy = posicion(nodo)
        x = cx - ancho / 2
        y = cy - alto / 2
        label_esc = _xml_escape(label)
        return (
            f'  <node id="{_xml_escape(nodo)}">\n'
            f'    <data key="d0">{label_esc}</data>\n'
            f'    <data key="d1">{_xml_escape(tipo)}</data>\n'
            f'    <data key="d2">{grado}</data>\n'
            f'    <data key="d5">\n'
            f'      <y:ShapeNode>\n'
            f'        <y:Geometry height="{alto:.1f}" width="{ancho:.1f}" '
            f'x="{x:.1f}" y="{y:.1f}"/>\n'
            f'        <y:Fill color="{color}" transparent="false"/>\n'
            f'        <y:BorderStyle color="{COLOR_NODO}" type="line" width="1.0"/>\n'
            f'        <y:NodeLabel alignment="center" autoSizePolicy="content" '
            f'fontFamily="Dialog" fontSize="11" fontStyle="plain" '
            f'hasBackgroundColor="false" hasLineColor="false" modelName="internal" '
            f'modelPosition="c" textColor="#FFFFFF" visible="true">{label_esc}</y:NodeLabel>\n'
            f'        <y:Shape type="rectangle"/>\n'
            f'      </y:ShapeNode>\n'
            f'    </data>\n'
            f'    <data key="d3">{cx:.1f}</data>\n'
            f'    <data key="d4">{cy:.1f}</data>\n'
            f'  </node>'
        )

    def arista_xml(u: str, v: str, eid: str, data: dict) -> str:
        relation = str(data.get("relation", ""))
        doc_id = str(data.get("doc_id", ""))
        chunk_id = str(data.get("chunk_id", ""))
        return (
            f'  <edge id="{_xml_escape(eid)}" source="{_xml_escape(u)}" '
            f'target="{_xml_escape(v)}" directed="true">\n'
            f'    <data key="d6">{_xml_escape(relation)}</data>\n'
            f'    <data key="d7">{_xml_escape(doc_id)}</data>\n'
            f'    <data key="d8">{_xml_escape(chunk_id)}</data>\n'
            f'    <data key="d9">1</data>\n'
            f'    <data key="d10">\n'
            f'      <y:PolyLineEdge>\n'
            f'        <y:LineStyle color="{COLOR_ARISTA}" type="line" width="1.0"/>\n'
            f'        <y:Arrows source="none" target="standard"/>\n'
            f'        <y:EdgeLabel alignment="center" backgroundColor="#F2F2F2" '
            f'fontFamily="Dialog" fontSize="9" fontStyle="plain" '
            f'hasLineColor="false" modelName="centered" textColor="#333333" '
            f'visible="true">{_xml_escape(relation)}</y:EdgeLabel>\n'
            f'        <y:BendStyle smoothed="true"/>\n'
            f'      </y:PolyLineEdge>\n'
            f'    </data>\n'
            f'  </edge>'
        )

    bloques = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"',
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '         xmlns:y="http://www.yworks.com/xml/graphml"',
        '         xmlns:yed="http://www.yworks.com/xml/yed/3"',
        '         xmlns:yfiles="http://www.yworks.com/xml/yfiles/3.0"',
        '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns',
        '                             http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">',
        '  <key id="d0" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="d1" for="node" attr.name="tipo" attr.type="string"/>',
        '  <key id="d2" for="node" attr.name="degree" attr.type="int"/>',
        '  <key id="d3" for="node" attr.name="x" attr.type="double"/>',
        '  <key id="d4" for="node" attr.name="y" attr.type="double"/>',
        '  <key id="d5" for="node" yfiles.type="nodegraphics"/>',
        '  <key id="d6" for="edge" attr.name="relation" attr.type="string"/>',
        '  <key id="d7" for="edge" attr.name="doc_id" attr.type="string"/>',
        '  <key id="d8" for="edge" attr.name="chunk_id" attr.type="string"/>',
        '  <key id="d9" for="edge" attr.name="weight" attr.type="int"/>',
        '  <key id="d10" for="edge" yfiles.type="edgegraphics"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    for i, nodo in enumerate(graph.nodes):
        bloques.append(nodo_xml(nodo))

    eid = 0
    for u, v, data in graph.edges(data=True):
        bloques.append(arista_xml(u, v, f"e{eid}", data))
        eid += 1

    bloques.extend(["  </graph>", "</graphml>", ""])

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(bloques))
