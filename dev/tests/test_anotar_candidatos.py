"""El ida y vuelta de la anotacion manual: candidatos.md marcado a mano ->
ground_truth_mini.jsonl. Si el parseo de las casillas se rompe, --recolectar
devolveria un ground truth vacio o parcial sin dar error, y las decisiones de
diseno se tomarian sobre una medicion silenciosamente incorrecta."""

import json

from scripts import anotar_candidatos


def _escribir_candidatos(tmp_path, contenido):
    ruta = tmp_path / "candidatos.md"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def test_recolectar_toma_solo_los_marcados(tmp_path, monkeypatch, capsys):
    candidatos = _escribir_candidatos(
        tmp_path,
        "## q001\n\n"
        "- [x] `F1-CSET-076` &mdash; a.pdf\n      extracto...\n\n"
        "- [ ] `F1-CSET-019` &mdash; b.pdf\n      extracto...\n\n"
        "- [X] `F1-ILIA-009` &mdash; c.pdf\n      extracto...\n\n"
        "## q002\n\n"
        "- [ ] `F2-SWF-124` &mdash; d.pdf\n      extracto...\n",
    )
    gt = tmp_path / "gt.jsonl"
    monkeypatch.setattr(anotar_candidatos, "CANDIDATOS_PATH", candidatos)
    monkeypatch.setattr(anotar_candidatos, "GROUND_TRUTH_PATH", gt)

    anotar_candidatos.recolectar(None)

    filas = [json.loads(l) for l in gt.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(filas) == 1  # q002 no tiene marcas -> se omite
    assert filas[0]["query_id"] == "q001"
    # mayuscula y minuscula valen; los no marcados no entran
    assert filas[0]["docs_relevantes"] == ["F1-CSET-076", "F1-ILIA-009"]


def test_recolectar_preserva_lo_ya_anotado_a_mano(tmp_path, monkeypatch):
    candidatos = _escribir_candidatos(
        tmp_path, "## q002\n\n- [x] `F2-SWF-124` &mdash; d.pdf\n      extracto...\n"
    )
    gt = tmp_path / "gt.jsonl"
    gt.write_text(
        json.dumps({"query_id": "q005", "docs_relevantes": ["F1-ATLCOUNCIL-165"]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(anotar_candidatos, "CANDIDATOS_PATH", candidatos)
    monkeypatch.setattr(anotar_candidatos, "GROUND_TRUTH_PATH", gt)

    anotar_candidatos.recolectar(None)

    filas = {json.loads(l)["query_id"] for l in gt.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert filas == {"q002", "q005"}  # la anotacion manual previa no se pierde
