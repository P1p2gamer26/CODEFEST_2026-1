"""Pruebas de la logica que decide entre configuraciones.

Es la que dice si una diferencia de F1@3 es real o ruido de muestreo, y de
ella salieron decisiones de la entrega (agregacion por suma, un solo encoder,
recuperacion sin grafo). Un error silencioso aqui corrompe todas.
"""

from scripts.eval_mini import (
    f1,
    ndcg,
    ndcg_anotado,
    ndcg_penalizado,
    techo_f1,
    veredicto_signos,
)


def test_f1_usa_el_denominador_de_recall_de_la_especificacion():
    """Sec. 10.2.1: R@3 = aciertos / min(|D*|, 3), no / |D*|. Con 5 documentos
    relevantes y los 3 devueltos correctos, el F1 vale 1 y no 0.75."""
    p, r, valor = f1(["a", "b", "c"], {"a", "b", "c", "d", "e"})
    assert (p, r) == (1.0, 1.0)
    assert valor == 1.0


def test_f1_sin_aciertos_es_cero():
    assert f1(["x", "y", "z"], {"a"}) == (0.0, 0.0, 0.0)


def test_veredicto_sin_diferencias():
    assert "exactamente lo mismo" in veredicto_signos("A", 0, "B", 0)


def test_veredicto_no_concluyente_con_reparto_parejo():
    """16-8 sobre 24 da p=0.15: es el caso real de sum vs max, que se entrego
    igual pero documentando que NO estaba demostrado."""
    salida = veredicto_signos("sum", 16, "max", 8)
    assert "p = 0.152" in salida
    assert "NO concluyente" in salida


def test_veredicto_concluyente_con_reparto_desigual():
    """20-4 sobre 24 da p=0.002. El umbral ad-hoc anterior no solo se
    equivocaba de criterio: cuando el reparto era significativo no imprimia
    nada, y el silencio se leia como 'sin diferencia'."""
    salida = veredicto_signos("A", 20, "B", 4)
    assert "gana de forma consistente" in salida
    assert "A gana" in salida


def test_veredicto_nombra_al_ganador_correcto_cuando_gana_el_segundo():
    salida = veredicto_signos("A", 4, "B", 20)
    assert "B gana de forma consistente" in salida


def test_techo_f1_no_es_uno():
    """El F1@3 no puede tender a 1 y hay que tenerlo a la vista.

    La sec. 10.2.2 fija P@3 = aciertos/3 con los 3 cupos siempre llenos: una
    consulta con un solo documento relevante topa en 0.50, una con dos en 0.80.
    Sobre la distribucion real del ground truth propio (5 consultas con 1
    documento, 11 con 2, 34 con 3 o mas) el techo es 0.906.
    """
    gt = (
        [{"docs_relevantes": ["a"]}] * 5
        + [{"docs_relevantes": ["a", "b"]}] * 11
        + [{"docs_relevantes": ["a", "b", "c"]}] * 34
    )
    assert round(techo_f1(gt, 3), 3) == 0.906
    assert round(techo_f1([{"docs_relevantes": ["a"]}], 3), 2) == 0.50
    assert round(techo_f1([{"docs_relevantes": ["a", "b"]}], 3), 2) == 0.80


def test_ndcg_penalizado_hunde_la_bibliografia_y_respeta_la_prosa():
    """La sec. 10.2.1 juzga el campo `text`: un fragmento de notas al pie de un
    documento relevante vale 1 para el proxy binario y ~0 para el evaluador."""
    rel = {"D1"}
    biblio = {
        "doc_id": "D1",
        "text": (
            "[1047] “What Are the Risks from Artificial Intelligence?,” MIT AI "
            "Risk Initiative, accessed September 2025, https://airisk.mit.edu/."
        ),
    }
    prosa = {
        "doc_id": "D1",
        "text": (
            "Conduct cybersecurity threat assessments to identify potential "
            "threat actors [359, 360], leveraging cyber threat intelligence [361]."
        ),
    }
    assert ndcg([biblio], rel, k=1) == 1.0
    assert ndcg_penalizado([biblio], rel, k=1) < 0.1
    # La prosa no se toca: el penalizado solo descuenta aparato bibliografico.
    assert ndcg_penalizado([prosa], rel, k=1) == ndcg([prosa], rel, k=1) == 1.0


def test_ndcg_penalizado_nunca_supera_al_binario():
    """Es una cota inferior por construccion: la ganancia se multiplica por un
    factor en [0, 1]. Si esta invariante se rompe, el reporte de 'cuanto miente
    el proxy' cambia de signo y deja de significar nada."""
    rel = {"D1"}
    frags = [
        {"doc_id": "D1", "text": "Texto util sobre capacidades espaciales y defensa."},
        {"doc_id": "D2", "text": "https://ejemplo.org/x"},
        {"doc_id": "D1", "text": "12 Autor, “Titulo Citado Largo,” Editorial, March 4, 2020."},
    ]
    assert ndcg_penalizado(frags, rel) <= ndcg(frags, rel)


def test_ndcg_anotado_distingue_responde_de_solo_mencionar():
    """Es la distincion que el proxy binario NO puede hacer y que decide las
    mejoras sobre fragmentos: mismo documento relevante, pasajes de valor
    distinto. Con nota 1 ("del tema pero no responde") tiene que dar la mitad
    que con nota 2."""
    frags = [{"chunk_id": "a", "doc_id": "D1", "text": "x"}]
    responde = ndcg_anotado(frags, {"a": 2})
    menciona = ndcg_anotado(frags, {"a": 1})
    assert responde == ndcg(frags, {"D1"}), "nota 2 equivale al proxy binario"
    assert abs(menciona - responde / 2) < 1e-9


def test_ndcg_anotado_cuenta_como_cero_lo_que_quedo_sin_nota():
    """La consigna del .md dice que dejar la casilla vacia equivale a 0; si esto
    cambiara, una anotacion a medias inflaria la metrica en silencio."""
    frags = [{"chunk_id": "a", "doc_id": "D1", "text": "x"}]
    assert ndcg_anotado(frags, {}) == 0.0
    assert ndcg_anotado(frags, {"otro": 2}) == 0.0
