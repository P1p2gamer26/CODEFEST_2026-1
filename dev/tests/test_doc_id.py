"""doc_id (sec. 2.3): debe ser estable frente a renombres/reubicaciones y
distinto para contenidos distintos.

Ademas, cuando ADL entrega su propio doc_id (clave real de emparejamiento
con el ground truth, aclarado en la Q&A final), ese debe ganarle al hash."""

import json

import pytest

from src.ingestion.doc_id import (
    DOC_ID_LENGTH,
    compute_doc_id,
    load_doc_id_manifest,
    resolve_doc_id,
)


def test_doc_id_is_stable_across_renaming(tmp_path):
    original = tmp_path / "informe_original.txt"
    original.write_text("Contenido de prueba sobre seguridad espacial.", encoding="utf-8")

    renamed = tmp_path / "subcarpeta" / "informe_renombrado.txt"
    renamed.parent.mkdir()
    renamed.write_text("Contenido de prueba sobre seguridad espacial.", encoding="utf-8")

    assert compute_doc_id(original) == compute_doc_id(renamed)


def test_doc_id_differs_for_different_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("Texto A", encoding="utf-8")
    b.write_text("Texto B", encoding="utf-8")

    assert compute_doc_id(a) != compute_doc_id(b)


def test_doc_id_has_expected_length(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("cualquier contenido", encoding="utf-8")
    assert len(compute_doc_id(path)) == DOC_ID_LENGTH


def _write_doc(tmp_path, name="informe.pdf"):
    path = tmp_path / name
    path.write_text("contenido cualquiera", encoding="utf-8")
    return path


def test_resolve_doc_id_prefers_adl_manifest_over_content_hash(tmp_path):
    path = _write_doc(tmp_path)
    assert resolve_doc_id(path, {"informe.pdf": "DOC-042"}) == "DOC-042"


def test_resolve_doc_id_falls_back_to_hash_without_manifest(tmp_path):
    path = _write_doc(tmp_path)
    assert resolve_doc_id(path, None) == compute_doc_id(path)
    # Archivo ausente del manifest: tampoco puede quedarse sin doc_id.
    assert resolve_doc_id(path, {"otro.pdf": "DOC-999"}) == compute_doc_id(path)


@pytest.mark.parametrize(
    "filename, content",
    [
        ("manifest.json", json.dumps({"informe.pdf": "DOC-042"})),
        ("manifest.json", json.dumps([{"fuente": "informe.pdf", "doc_id": "DOC-042"}])),
        ("manifest.jsonl", json.dumps({"archivo": "informe.pdf", "id": "DOC-042"})),
        ("manifest.csv", "fuente,doc_id\ninforme.pdf,DOC-042\n"),
    ],
)
def test_load_doc_id_manifest_accepts_the_likely_formats(tmp_path, filename, content):
    manifest_path = tmp_path / filename
    manifest_path.write_text(content, encoding="utf-8")
    assert load_doc_id_manifest(manifest_path) == {"informe.pdf": "DOC-042"}


def test_load_doc_id_manifest_indexes_by_filename_not_full_path(tmp_path):
    """El manifest de ADL puede traer rutas; el corpus se reubica en disco."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"fuente": "fenomeno_1/pdf/informe.pdf", "doc_id": "DOC-042"}]),
        encoding="utf-8",
    )
    assert load_doc_id_manifest(manifest_path) == {"informe.pdf": "DOC-042"}


def test_load_doc_id_manifest_fails_loudly_on_unknown_field_names(tmp_path):
    """Mejor romper en la construccion del indice que entregar 50 consultas
    emparejadas contra doc_id equivocados."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("columna_rara,otra\ninforme.pdf,DOC-042\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no produjo ninguna entrada"):
        load_doc_id_manifest(manifest_path)
