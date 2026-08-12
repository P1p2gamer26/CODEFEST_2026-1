"""El panel de metricas de la GUI puntua contra el mini ground truth y separa
las etiquetas humanas de las del panel de agentes (que no son comparables)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

tk = pytest.importorskip("tkinter")


def test_gui_arranca_con_la_configuracion_evaluada_sin_grafo():
    import gui_app

    assert gui_app.GUI_USE_GRAPH is False


def _resultado(query_id, doc_ids):
    return {
        "query_id": query_id,
        "documents": [{"doc_id": d} for d in doc_ids],
        "fragments": [{"doc_id": d} for d in doc_ids],
    }


def test_panel_separa_humanas_de_agente():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("sin display para Tk")
    root.withdraw()
    try:
        import gui_app

        panel = gui_app.MetricsPanel(root)
        panel.gt = {
            "q001": ({"D1", "D2"}, "humano"),
            "q002": ({"D9"}, "agente"),
        }
        panel.registrar(_resultado("q001", ["D1", "D2", "X"]))
        panel.registrar(_resultado("q002", ["X", "Y", "Z"]))
        panel.registrar(_resultado("chat", ["X"]))  # sin anotar: no cuenta

        assert len(panel.acumulado["humano"]) == 1
        assert len(panel.acumulado["agente"]) == 1
        assert panel.acumulado["humano"][0][0] == pytest.approx(0.8)  # P=2/3, R=1
        assert panel.acumulado["agente"][0][0] == 0.0
        assert "2 consultas" in panel.lbl_total.cget("text")
        assert "F1@3 0.400" in panel.lbl_total.cget("text")

        panel.reiniciar()
        assert panel.acumulado == {}
        assert panel.lbl_total.cget("text") == "total: --"
    finally:
        root.destroy()
