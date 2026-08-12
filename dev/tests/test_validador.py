"""El validador cierra cuatro huecos frente al PDF y la Q&A de ADL.

Los cuatro son fallos que costarian la evaluacion y que la version anterior
no veia: tipos de la Tabla 1, sintaxis posterior a Python 3.9 en el
generador, oraciones cortadas (sec. 3.3) y documentos del manifest que no
llegaron al indice.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validar_entrega import (  # noqa: E402
    validar_cobertura_corpus,
    validar_completitud_linguistica,
    validar_python39,
    validar_tipos_metadata,
)


def _fila(**cambios):
    fila = {
        "doc_id": "F1-AIINDEX-001",
        "chunk_id": "F1-AIINDEX-001-c0000",
        "fuente": "x.pdf",
        "formato": "pdf",
        "fenomeno": 1,
        "posicion": 0,
        "num_tokens": 10,
        "texto": "hola.",
    }
    fila.update(cambios)
    return fila


def test_tipos_metadata_acepta_registro_valido():
    assert validar_tipos_metadata([_fila()]) == []


def test_tipos_metadata_rechaza_fenomeno_como_cadena():
    """Tabla 1, sec. 3.4: fenomeno/posicion/num_tokens son enteros."""
    errores = validar_tipos_metadata([_fila(fenomeno="1")])
    assert any("fenomeno" in e for e in errores)


def test_tipos_metadata_rechaza_fenomeno_fuera_de_rango():
    errores = validar_tipos_metadata([_fila(fenomeno=4)])
    assert any("fuera de" in e for e in errores)


def test_tipos_metadata_rechaza_bool_como_entero():
    """bool es subclase de int en Python: isinstance solo no alcanza."""
    errores = validar_tipos_metadata([_fila(posicion=True)])
    assert any("posicion" in e for e in errores)


def test_completitud_reporta_fragmento_cortado():
    """Sec. 3.3: ningun fragmento con oraciones cortadas."""
    avisos = validar_completitud_linguistica(
        [{"query_id": "q001",
          "fragments": [{"rank": 1, "text": "esto se corta a media fra"}]}]
    )
    assert len(avisos) == 1
    assert "q001" in avisos[0]


def test_completitud_acepta_fragmento_terminado():
    avisos = validar_completitud_linguistica(
        [{"query_id": "q001",
          "fragments": [{"rank": 1, "text": "esto termina bien."}]}]
    )
    assert avisos == []


def test_python39_rechaza_union_pep604(tmp_path):
    """PEP 604 (X | None) es 3.10+; ADL evalua con >= 3.9.5 y el import
    revienta antes de leer una consulta."""
    script = tmp_path / "generador.py"
    script.write_text("def f(x: int | None) -> None: ...\n", encoding="utf-8")
    assert validar_python39(script) != []


def test_python39_acepta_el_generador_con_future(tmp_path):
    script = tmp_path / "generador.py"
    script.write_text(
        "from __future__ import annotations\n"
        "def f(x: int | None) -> None: ...\n",
        encoding="utf-8",
    )
    assert validar_python39(script) == []


def test_cobertura_reporta_los_doc_id_que_faltan(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "ruta,nombre,doc_id\n"
        "a/x.pdf,x.pdf,F1-A-001\n"
        "a/y.pdf,y.pdf,F1-A-002\n",
        encoding="utf-8",
    )
    avisos = validar_cobertura_corpus({"F1-A-001"}, manifest)
    assert len(avisos) == 1
    assert "F1-A-002" in avisos[0]


def test_cobertura_callada_cuando_esta_todo(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("ruta,nombre,doc_id\na/x.pdf,x.pdf,F1-A-001\n",
                        encoding="utf-8")
    assert validar_cobertura_corpus({"F1-A-001"}, manifest) == []


def test_la_entrega_real_pasa_los_tipos():
    """Puerta sobre el artefacto que se entrega, no sobre un fixture."""
    meta = (Path(__file__).resolve().parents[2] / "Entrega" / "base_vectorial"
            / "encoder_paraphrase-multilingual-MiniLM-L12-v2" / "metadata.jsonl")
    if not meta.is_file():
        import pytest

        pytest.skip("base_vectorial no rehidratada")
    filas = [json.loads(l) for l in meta.open(encoding="utf-8")]
    assert validar_tipos_metadata(filas) == []
