"""E13: deduplicar fragmentos casi identicos entre los 10 entregados."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from barrido_dedup_e13 import dedup, solapamiento


def test_solapamiento_identico_es_uno():
    t = "el corpus tiene setecientos sesenta pdf y novecientos json en total"
    assert solapamiento(t, t) == 1.0


def test_solapamiento_disjunto_es_cero():
    assert solapamiento("alfa beta gamma delta epsilon", "uno dos tres cuatro cinco") == 0.0


def test_solapamiento_parcial_entre_cero_y_uno():
    a = "space debris in low earth orbit is a growing problem for operators"
    b = "is a growing problem for operators of commercial satellite constellations"
    assert 0.0 < solapamiento(a, b) < 1.0


def test_dedup_reemplaza_el_duplicado_y_devuelve_diez():
    frags = [{"chunk_id": f"c{i}", "text": f"texto unico numero {i} alfa beta gamma delta"}
             for i in range(10)]
    frags[7]["text"] = frags[3]["text"]          # duplicado exacto del rank 4
    reserva = [{"chunk_id": "r0", "text": "reemplazo distinto omega psi chi phi upsilon"}]
    out = dedup(frags, umbral=0.8, reserva=reserva)
    assert len(out) == 10
    assert [f["chunk_id"] for f in out].count("c7") == 0
    assert out[7]["chunk_id"] == "r0"


def test_dedup_sin_reserva_conserva_el_duplicado():
    """Sec. 9.2 exige 10 fragmentos: sin reemplazo disponible NO se borra."""
    frags = [{"chunk_id": f"c{i}", "text": "mismo texto alfa beta gamma delta epsilon"}
             for i in range(10)]
    out = dedup(frags, umbral=0.8, reserva=[])
    assert len(out) == 10


def test_dedup_conserva_el_orden_de_los_no_tocados():
    frags = [{"chunk_id": f"c{i}", "text": f"texto unico numero {i} alfa beta gamma delta"}
             for i in range(10)]
    out = dedup(frags, umbral=0.8, reserva=[])
    assert [f["chunk_id"] for f in out] == [f["chunk_id"] for f in frags]
