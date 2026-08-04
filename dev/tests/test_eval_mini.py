"""Pruebas de la logica que decide entre configuraciones.

Es la que dice si una diferencia de F1@3 es real o ruido de muestreo, y de
ella salieron decisiones de la entrega (agregacion por suma, un solo encoder,
recuperacion sin grafo). Un error silencioso aqui corrompe todas.
"""

from scripts.eval_mini import f1, veredicto_signos


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
