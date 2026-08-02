"""El guion de fin de linea de los PDF (U+FFFE) partia palabras en el 30% de
los chunks del corpus real. Ver src/cleaning/clean.py."""

from src.cleaning.clean import clean_text

CORTE = "￾"  # U+FFFE, tal como lo entrega pypdfium2


def test_une_palabra_partida():
    assert clean_text(f"satel{CORTE}lites") == "satellites"
    assert clean_text(f"weap{CORTE}ons") == "weapons"
    assert clean_text("inteli­gencia") == "inteligencia"


def test_conserva_el_guion_de_los_compuestos():
    assert clean_text(f"AI{CORTE}related") == "AI-related"
    assert clean_text(f"COVID{CORTE}19") == "COVID-19"
    assert clean_text(f"Human{CORTE}Centered") == "Human-Centered"


def test_el_mapa_del_corpus_manda_sobre_la_heuristica():
    # Sin mapa la heuristica une; con evidencia del corpus pone el guion.
    assert clean_text(f"open{CORTE}source") == "opensource"
    assert clean_text(f"open{CORTE}source", {("open", "source"): "-"}) == "open-source"


def test_guion_suelto_se_borra_sin_pegar_palabras():
    assert clean_text(f"uno {CORTE} dos") == "uno dos"


def test_no_toca_el_texto_sano():
    sano = "Los detritos orbitales en la orbita baja terrestre.\n\nSegundo parrafo."
    assert clean_text(sano) == sano
