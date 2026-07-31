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


def _manifest(tmp_path, filas):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(f) for f in filas), encoding="utf-8"
    )
    return load_doc_id_manifest(manifest_path)


def test_resolve_doc_id_prefers_adl_manifest_over_content_hash(tmp_path):
    path = _write_doc(tmp_path)
    manifest = _manifest(tmp_path, [{"nombre": "informe.pdf", "doc_id": "DOC-042"}])
    assert resolve_doc_id(path, manifest) == "DOC-042"


def test_resolve_doc_id_falls_back_to_hash_without_manifest(tmp_path):
    path = _write_doc(tmp_path)
    assert resolve_doc_id(path, None) == compute_doc_id(path)
    # Archivo ausente del manifest: tampoco puede quedarse sin doc_id.
    manifest = _manifest(tmp_path, [{"nombre": "otro.pdf", "doc_id": "DOC-999"}])
    assert resolve_doc_id(path, manifest) == compute_doc_id(path)


def test_resolve_doc_id_desambigua_nombres_repetidos_por_ruta(tmp_path):
    """El inventario real de ADL trae 59 nombres de archivo que aparecen en
    dos carpetas con DOC_ID distintos (CSET Reports vs. Translation, tiles por
    nivel de zoom). Indexar solo por nombre asignaria el doc_id equivocado y
    esos documentos no puntuarian en F1@3."""
    corpus = tmp_path / "corpus"
    for carpeta in ("Reports", "Translation"):
        (corpus / carpeta).mkdir(parents=True)
        (corpus / carpeta / "cset.pdf").write_text(carpeta, encoding="utf-8")

    manifest = _manifest(
        tmp_path,
        [
            {"ruta": "Reports/cset.pdf", "nombre": "cset.pdf", "doc_id": "F1-CSET-001"},
            {"ruta": "Translation/cset.pdf", "nombre": "cset.pdf", "doc_id": "F1-CSET-002"},
        ],
    )
    assert manifest.nombres_ambiguos == {"cset.pdf"}

    resolver = lambda c: resolve_doc_id(corpus / c / "cset.pdf", manifest, corpus_dir=corpus)
    assert resolver("Reports") == "F1-CSET-001"
    assert resolver("Translation") == "F1-CSET-002"


def test_resolve_doc_id_por_nombre_cuando_no_hay_corpus_dir(tmp_path):
    """Sin ruta relativa disponible, un nombre inequivoco sigue resolviendo."""
    path = _write_doc(tmp_path)
    manifest = _manifest(
        tmp_path, [{"ruta": "F2/pdf/informe.pdf", "nombre": "informe.pdf", "doc_id": "DOC-042"}]
    )
    assert resolve_doc_id(path, manifest) == "DOC-042"


@pytest.mark.parametrize(
    "filename, content",
    [
        ("manifest.json", json.dumps({"informe.pdf": "DOC-042"})),
        ("manifest.json", json.dumps([{"fuente": "informe.pdf", "doc_id": "DOC-042"}])),
        ("manifest.jsonl", json.dumps({"archivo": "informe.pdf", "id": "DOC-042"})),
        ("manifest.csv", "fuente,doc_id\ninforme.pdf,DOC-042\n"),
        ("manifest.csv", "ruta,nombre,doc_id\nF1/pdf/informe.pdf,informe.pdf,DOC-042\n"),
    ],
)
def test_load_doc_id_manifest_accepts_the_likely_formats(tmp_path, filename, content):
    manifest_path = tmp_path / filename
    manifest_path.write_text(content, encoding="utf-8")
    manifest = load_doc_id_manifest(manifest_path)
    assert manifest.lookup(nombre="informe.pdf") == "DOC-042"


def test_load_doc_id_manifest_fails_loudly_on_unknown_field_names(tmp_path):
    """Mejor romper en la construccion del indice que entregar 50 consultas
    emparejadas contra doc_id equivocados."""
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("columna_rara,otra\ninforme.pdf,DOC-042\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no produjo ninguna entrada"):
        load_doc_id_manifest(manifest_path)
