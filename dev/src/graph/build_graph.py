"""Construccion e integracion del grafo de conocimiento (sec. 7.2, paso 3;
sec. 7.3). Cada arista conserva `doc_id` y `chunk_id` de origen (primera
aparicion) para poder rastrear la evidencia textual de cada relacion; si la
misma tripleta (sujeto, relacion, objeto) aparece en varios chunks, se agrega
en una sola arista con `weight` = numero de chunks que la respaldan y
`chunk_ids` con la lista completa de evidencia.

`export_graphml()` escribe un GraphML con layout calculado (spring layout
determinista, seed fija) y en el formato yEd: posiciones, tamanos, colores por
tipo de entidad, etiquetas en nodos y relaciones en aristas. Asi el archivo se
ve bien (sin nodos superpuestos, con las relaciones legibles) tanto en yEd
como en Gephi, y `networkx.read_graphml` lo sigue leyendo igual (extrae x/y
del yfiles geometry de forma nativa).

El layout tiene un tope de tamano: es cuadratico en el numero de nodos y el
grafo del corpus real tiene 224.101, o sea 2,5e10 parejas por pasada. Por
encima de MAX_NODOS_LAYOUT se exporta sin estilo, que es lo unico viable a esa
escala (y de todas formas ningun visor abre un grafo asi; para mirarlo se usa
scripts/ver_grafo.py, que dibuja el subgrafo de los nodos mas conectados)."""

import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import networkx as nx

from ..ingestion.pipeline import ChunkRecord
from .relations import extract_triples

# XML 1.0 no admite caracteres de control salvo tab, LF y CR (sec. 2.2 de la
# norma), y GraphML es XML. El texto del corpus real -- sobre todo el que sale
# de OCR y de PDF mal extraido -- trae bytes NUL y de control que se cuelan en
# los nombres de entidad, y la escritura aborta con ValueError al final de la
# construccion, despues de horas de NER. Se limpian al construir el grafo, no
# al exportarlo, para que el grafo en memoria ya sea exportable.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￾￿]")

# El NER busca entidades nombradas en lenguaje natural. Aplicado a una fila de
# CSV o a los atributos de una tesela vectorial produce basura: medido sobre el
# corpus real, las entidades mas frecuentes del grafo eran "FALSO",
# "VERDADEIRO" (valores booleanos de columnas) y "au_seg_marq" (el nombre de la
# capa de los mapas de Amazon Underworld). Son el 17,7% de los chunks y no
# aportan conocimiento, solo ruido, tamano y tiempo de NER.
FORMATOS_SIN_NARRATIVA = frozenset({"csv", "xlsx", "pbf"})

# Por encima de esto, export_graphml omite layout y estilo (ver el docstring
# del modulo). El tope estaba en 2.000, que en la practica es inalcanzable:
# medido (1 ago 2026, CPU) 100 nodos -> 1,9 s, 400 -> 32 s, 1.000 -> 233 s, o
# sea que 2.000 pasan de los 15 min. 400 es lo mas alto que sigue siendo
# interactivo, y muy por encima del --top 40 que usa scripts/ver_grafo.py.
MAX_NODOS_LAYOUT = 400

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


def _attr(texto: str) -> str:
    """Escapa un valor que va DENTRO de un atributo XML entrecomillado.

    `saxutils.escape` solo cubre & < >, no la comilla doble, asi que una entidad
    como 'Operacion "Libertad"' cerraba el atributo y producia un GraphML que ni
    networkx podia releer. No es hipotetico: el grafo del corpus real tiene 160
    nodos con comilla doble en el nombre (texto de OCR y de PDF mal extraido).
    Para el contenido de un elemento basta `_xml_escape`.
    """
    return _xml_escape(texto, {'"': "&quot;"})


def _limpiar(texto: str) -> str:
    """Deja el texto en algo que GraphML acepte, y colapsa los espacios: un
    nombre de entidad con saltos de linea es valido en XML pero ilegible."""
    return " ".join(_CONTROL.sub("", texto).split())


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

    OJO: la separacion compara todas las parejas, o sea O(n^2) por pasada. Solo
    se llama por debajo de MAX_NODOS_LAYOUT.
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
    # Evidencia por tripleta: misma (sujeto, relacion, objeto) repetida en
    # varios chunks -> una sola arista con weight = numero de chunks que la
    # respaldan y chunk_ids con la lista completa de evidencia (unida con ';',
    # porque GraphML solo admite atributos escalares). doc_id/chunk_id apuntan
    # a la primera aparicion para la trazabilidad puntual de la sec. 7.2.
    evidencia: dict[tuple[str, str, str], dict] = {}

    for record in records:
        if record.formato in FORMATOS_SIN_NARRATIVA:
            continue
        for subject, relation, obj, subj_tipo, obj_tipo in extract_triples(
            record.texto, record.idioma
        ):
            subject, relation, obj = _limpiar(subject), _limpiar(relation), _limpiar(obj)
            if not subject or not obj:
                continue
            graph.add_node(subject, label=subject)
            graph.add_node(obj, label=obj)
            tipo_counts[subject][subj_tipo] += 1
            tipo_counts[obj][obj_tipo] += 1
            clave = (subject, relation, obj)
            if clave not in evidencia:
                evidencia[clave] = {
                    "doc_id": record.doc_id,
                    "chunk_id": record.chunk_id,
                    "chunk_ids": [],
                }
            if record.chunk_id not in evidencia[clave]["chunk_ids"]:
                evidencia[clave]["chunk_ids"].append(record.chunk_id)

    for (subject, relation, obj), datos in evidencia.items():
        chunk_ids = datos["chunk_ids"]
        graph.add_edge(
            subject,
            obj,
            relation=relation,
            doc_id=datos["doc_id"],
            chunk_id=datos["chunk_id"],
            chunk_ids=";".join(chunk_ids),
            weight=len(chunk_ids),
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
    max_nodos_layout: int = MAX_NODOS_LAYOUT,
) -> None:
    """Escribe el grafo en GraphML con layout y estilo para visualizarlo.

    El layout es un spring layout determinista (seed fija -> reproducible):
    separa los nodos conectados entre si y evita que se superpongan. Se
    escriben a la vez las posiciones como atributos genericos x/y (los lee
    Gephi y networkx) y como geometria yfiles (los usa yEd), ademas de
    tamanos, color por tipo de entidad y etiquetas de relacion en cada arista.

    Por encima de `max_nodos_layout` se escribe el GraphML plano: calcular el
    layout es cuadratico y con el grafo del corpus real (224.101 nodos) no
    terminaria nunca.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if graph.number_of_nodes() == 0 or graph.number_of_nodes() > max_nodos_layout:
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
            f'  <node id="{_attr(nodo)}">\n'
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
        chunk_ids = str(data.get("chunk_ids", chunk_id))
        weight = str(data.get("weight", 1))
        return (
            f'  <edge id="{_attr(eid)}" source="{_attr(u)}" '
            f'target="{_attr(v)}" directed="true">\n'
            f'    <data key="d6">{_xml_escape(relation)}</data>\n'
            f'    <data key="d7">{_xml_escape(doc_id)}</data>\n'
            f'    <data key="d8">{_xml_escape(chunk_id)}</data>\n'
            f'    <data key="d11">{_xml_escape(chunk_ids)}</data>\n'
            f'    <data key="d9">{weight}</data>\n'
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
        '  <key id="d11" for="edge" attr.name="chunk_ids" attr.type="string"/>',
        '  <key id="d10" for="edge" yfiles.type="edgegraphics"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    for nodo in graph.nodes:
        bloques.append(nodo_xml(nodo))

    eid = 0
    for u, v, data in graph.edges(data=True):
        bloques.append(arista_xml(u, v, f"e{eid}", data))
        eid += 1

    bloques.extend(["  </graph>", "</graphml>", ""])

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(bloques))
