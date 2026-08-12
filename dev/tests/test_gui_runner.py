"""Paridad entre el flujo interactivo y Entrega/generador.py."""

from src.gui import runner
from src.retrieval.search import Hit


class _Encoder:
    def count_tokens(self, text):
        return len(text.split())


class _Generador:
    def __init__(self):
        self.texto_consulta = None

    def expandir_consulta(self, text):
        return f"{text} EQUIVALENCIA"

    def build_result_object(self, query_id, hits, texto_consulta=None):
        self.texto_consulta = texto_consulta
        return {"query_id": query_id, "documents": [], "fragments": []}


def _hit():
    return Hit(
        rank=1,
        score=0.8,
        chunk_id="c1",
        doc_id="d1",
        fuente="d1.pdf",
        texto="texto",
        formato="pdf",
        fenomeno=1,
        idioma="es",
        fila=0,
    )


def test_answer_one_separa_texto_vectorial_y_texto_del_grafo(monkeypatch):
    vistos = {}
    generador = _Generador()

    monkeypatch.setattr(runner, "search", lambda *args, **kwargs: [_hit()])

    def graph_search(query, graph, lang=None, k=10):
        vistos["grafo"] = query
        return []

    monkeypatch.setattr("src.graph.graph_retrieval.graph_search", graph_search)

    resultado, tokens, score = runner._answer_one(
        "consulta original",
        "q001",
        _Encoder(),
        object(),
        [],
        {},
        object(),
        generador,
        200,
    )

    assert resultado["query_id"] == "q001"
    assert tokens == 2
    assert score == 0.8
    assert vistos["grafo"] == "consulta original"
    assert generador.texto_consulta == "consulta original EQUIVALENCIA"
