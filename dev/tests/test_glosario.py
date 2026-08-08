"""Pruebas del glosario bilingue de expansion de consulta.

Lo que estas pruebas protegen no es un formato sino una propiedad: **una
consulta sin terminos del glosario tiene que salir identica**. Si eso se
rompe, cambia el vector de las 40 consultas que el glosario no toca y con el
la entrega entera, en silencio.
"""

from src.retrieval.glosario import (
    GLOSARIO,
    expandir_consulta,
    terminos_expandidos,
)

# Enunciados reales de consultas_50_oficiales.jsonl.
Q001 = (
    "¿Cómo está transformando la inteligencia artificial la capacidad de los "
    "Estados para prevenir, detectar y contrarrestar amenazas NBQR?"
)
Q021 = (
    "¿Qué implicaciones militares tienen las maniobras de proximidad y "
    "encuentro (RPO) realizadas por satélites?"
)
Q049 = (
    "¿De qué manera el accionar de grupos armados interfiere y obstaculiza el "
    "desarrollo de los procesos de restitución de tierras?"
)


def test_una_consulta_sin_terminos_sale_identica():
    """La propiedad que sostiene todo lo demas: efecto nulo donde no aplica."""
    assert expandir_consulta(Q049) == Q049
    assert terminos_expandidos(Q049) == []


def test_nbqr_trae_la_forma_que_el_corpus_si_usa():
    """NBQR no aparece ni una vez en los 128.526 chunks; CBRN aparece en 66."""
    assert "CBRN" in expandir_consulta(Q001)
    assert Q001 in expandir_consulta(Q001), "la consulta original se conserva"


def test_los_acentos_no_impiden_el_emparejamiento():
    """La tabla va sin acentos y la consulta viene con ellos."""
    assert terminos_expandidos("armas de energía dirigida contra satélites")
    assert terminos_expandidos("desechos orbitales") == terminos_expandidos(
        "DESECHOS ORBITALES"
    )


def test_no_expande_lo_que_la_consulta_ya_nombra_como_el_corpus():
    """q021 ya trae la sigla RPO, que el corpus usa 871 veces.

    La entrada se quito justamente por eso: expandirla costaba F1 0.33 -> 0.00.
    El criterio de entrada a la tabla es que la CONSULTA no tenga ningun
    puente al vocabulario del corpus, no solo que el termino espanol sea raro.
    """
    assert terminos_expandidos(Q021) == []


def test_la_expansion_es_determinista():
    """Dos corridas tienen que producir el mismo texto: si no, el vector de la
    consulta cambia entre corridas y la entrega deja de ser reproducible."""
    for _ in range(3):
        assert expandir_consulta(Q001) == expandir_consulta(Q001)


def test_no_hay_terminos_duplicados_en_la_tabla():
    claves = [k for k, _ in GLOSARIO]
    assert len(claves) == len(set(claves))


def test_una_consulta_con_dos_terminos_los_trae_los_dos_sin_repetir():
    texto = "pruebas antisatélite y sus desechos orbitales"
    extras = terminos_expandidos(texto)
    assert len(extras) == 2
    assert len(extras) == len(set(extras))
