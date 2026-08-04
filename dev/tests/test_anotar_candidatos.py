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

    filas = {f["query_id"]: f for f in (json.loads(l) for l in gt.read_text(encoding="utf-8").splitlines() if l.strip())}
    # mayuscula y minuscula valen; los no marcados no entran
    assert filas["q001"]["docs_relevantes"] == ["F1-CSET-076", "F1-ILIA-009"]
    # q002 se vio y no tenia nada relevante: entra con la lista VACIA, no se
    # omite. Omitirla la hacia indistinguible de una consulta sin anotar, que
    # es como las 9 consultas con cero marcas quedaron figurando como
    # "pendientes de anotar" cuando en realidad ya se habian revisado.
    assert filas["q002"]["docs_relevantes"] == []


def test_recolectar_lee_la_procedencia_del_pool(tmp_path, monkeypatch):
    """El campo `pool` viaja en el .md, no en un flag de --recolectar.

    eval_mini --sin-pooling separa las consultas anotadas de forma
    independiente de las sesgadas por el pool de un encoder. Si la procedencia
    dependiera de que el anotador repita los mismos flags horas despues, se
    perderia en silencio y la comparacion entre encoders quedaria sesgada sin
    que nada avisara.
    """
    candidatos = _escribir_candidatos(
        tmp_path,
        "<!-- pool: minilm+bm25 -->\n\n## q003\n\n- [x] `F1-CSET-001` &mdash; a.pdf\n      x...\n",
    )
    gt = tmp_path / "gt.jsonl"
    monkeypatch.setattr(anotar_candidatos, "CANDIDATOS_PATH", candidatos)
    monkeypatch.setattr(anotar_candidatos, "GROUND_TRUTH_PATH", gt)

    anotar_candidatos.recolectar(None)

    fila = json.loads(gt.read_text(encoding="utf-8").strip())
    assert fila["pool"] == "minilm+bm25"


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
