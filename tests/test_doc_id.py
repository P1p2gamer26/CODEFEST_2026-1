"""doc_id (sec. 2.3): debe ser estable frente a renombres/reubicaciones y
distinto para contenidos distintos."""

from src.ingestion.doc_id import DOC_ID_LENGTH, compute_doc_id


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
