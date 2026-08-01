"""Construccion e integracion del grafo de conocimiento (sec. 7.2, paso 3;
sec. 7.3). Cada arista conserva `doc_id` y `chunk_id` de origen para poder
rastrear la evidencia textual de cada relacion."""

import re
from pathlib import Path

import networkx as nx

from ..ingestion.pipeline import ChunkRecord
from .relations import extract_triples

# XML 1.0 no admite caracteres de control salvo tab, LF y CR (sec. 2.2 de la
# norma), y GraphML es XML. El texto del corpus real -- sobre todo el que sale
# de OCR y de PDF mal extraido -- trae bytes NUL y de control que se cuelan en
# los nombres de entidad, y nx.write_graphml aborta con ValueError al final de
# la construccion, despues de horas de NER. Se limpian al construir el grafo,
# no al exportarlo, para que el grafo en memoria ya sea exportable.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f￾￿]")


def _limpiar(texto: str) -> str:
    """Deja el texto en algo que GraphML acepte, y colapsa los espacios: un
    nombre de entidad con saltos de linea es valido en XML pero ilegible."""
    return " ".join(_CONTROL.sub("", texto).split())


# El NER busca entidades nombradas en lenguaje natural. Aplicado a una fila de
# CSV o a los atributos de una tesela vectorial produce basura: medido sobre el
# corpus real, las entidades mas frecuentes del grafo eran "FALSO",
# "VERDADEIRO" (valores booleanos de columnas) y "au_seg_marq" (el nombre de la
# capa de los mapas de Amazon Underworld). Son el 17,7% de los chunks y no
# aportan conocimiento, solo ruido, tamano y tiempo de NER.
FORMATOS_SIN_NARRATIVA = frozenset({"csv", "xlsx", "pbf"})


def build_knowledge_graph(records: list[ChunkRecord]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for record in records:
        if record.formato in FORMATOS_SIN_NARRATIVA:
            continue
        for subject, relation, obj in extract_triples(record.texto, record.idioma):
            subject, relation, obj = _limpiar(subject), _limpiar(relation), _limpiar(obj)
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
