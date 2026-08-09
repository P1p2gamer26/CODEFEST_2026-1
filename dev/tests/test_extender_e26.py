"""E26: extender un fragmento con el chunk contiguo no puede duplicar texto.

Es EL fallo que mato a la version anterior de esta idea (concatenar el vecino
entero): el chunker solapa una oracion entre chunks consecutivos y el parrafo
salia dos veces. El caso de abajo es el real de `F2-UNOOSA-030`, donde el
segmentador parte la oracion solapada distinto a cada lado del borde
('23.' + 'Las directrices...' contra '23. Las directrices...' entera), asi que
un dedup por IGUALDAD de oracion no lo ve y hace falta la contencion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from barrido_extender_e26 import extender, vecinos_de  # noqa: E402


def test_vecinos_de():
    assert vecinos_de("F1-X-001-c0007") == ["F1-X-001-c0008", "F1-X-001-c0006"]
    assert vecinos_de("F1-X-001-c0000") == ["F1-X-001-c0001"]  # no hay -1
    assert vecinos_de("sin-sufijo") == []


def test_no_duplica_la_oracion_solapada_aunque_el_splitter_la_parta_distinto():
    frag = {
        "chunk_id": "F2-X-030-c0238",
        "text": "El resultado de esos intercambios podria presentarse a la Comision. "
                "23. Las directrices reflejan un entendimiento comun sobre dificultades.",
    }
    vecino = ("23. Las directrices reflejan un entendimiento comun sobre dificultades. "
              "Se alienta a los Estados a que promuevan la cooperacion internacional.")
    out = extender(frag, "es", {"F2-X-030-c0239": vecino}, max_words=250)
    assert out["text"].lower().count("las directrices reflejan") == 1
    assert "se alienta a los estados" in out["text"].lower()


def test_respeta_el_tope_de_palabras():
    frag = {"chunk_id": "F2-X-030-c0010", "text": "uno dos tres."}
    vecino = " ".join(f"palabra{i} palabra{i}b palabra{i}c." for i in range(200))
    out = extender(frag, "es", {"F2-X-030-c0011": vecino}, max_words=20)
    assert len(out["text"].split()) <= 20
    assert len(out["text"].split()) > 3  # algo se anadio


def test_no_toca_el_fragmento_que_ya_llena_el_presupuesto():
    frag = {"chunk_id": "F2-X-030-c0010", "text": "uno dos tres cuatro cinco."}
    out = extender(frag, "es", {"F2-X-030-c0011": "seis siete."}, max_words=5)
    assert out == frag


def test_sin_vecino_devuelve_el_fragmento_intacto():
    frag = {"chunk_id": "F2-X-030-c0010", "text": "uno dos tres."}
    assert extender(frag, "es", {}, max_words=250) == frag
