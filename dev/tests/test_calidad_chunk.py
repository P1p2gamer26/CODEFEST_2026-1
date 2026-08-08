"""Pruebas del detector de aparato bibliografico.

Los casos son texto REAL de Entrega/resultados.jsonl, no inventado: el detector
se diseno mirando esos 500 fragmentos y estas pruebas fijan los aciertos y los
limites conocidos para que un retoque de las expresiones regulares no los
rompa en silencio.
"""

from src.retrieval.calidad_chunk import calidad, fraccion_aparato, segmentar

# F1-CSET-098-c0636, rank 2 de su consulta: un cupo del top-10 gastado en la
# lista de referencias.
BIBLIOGRAFIA = (
    "[1047] “What Are the Risks from Artificial Intelligence?,” MIT AI Risk "
    "Initiative, accessed September 2025, https://airisk.mit.edu/. "
    "[1048] “Vulnerability Scanning Tools and Services,” UK National Cyber "
    "Security Centre, January 19, 2021, "
    "www.ncsc.gov.uk/guidance/vulnerability-scanning-tools-and-services."
)

# F1-CSET-098-c0070: 22% de tokens con digito y aun asi es contenido util. Es
# el contraejemplo que descarta usar densidad numerica como senal.
PROSA_MUY_CITADA = (
    "Conduct cybersecurity threat assessments to identify potential threat "
    "actors [359, 360], leveraging cyber threat intelligence [361]. Conduct "
    "threat modeling to identify potential vulnerabilities and risks "
    "[362, 363, 364, 365], including those specific to AI systems [366]."
)

# F2-SWF-078-c0242: dos oraciones de prosa y despues el bloque de notas. El
# chunker corta por ventana de tokens y no respeta la frontera cuerpo/notas.
MIXTO = (
    "The Ukraine conflict and the subsequent sanctions placed on the Russian "
    "Federation brought to light several Russian industrial and technological "
    "deficiencies in its space program such as the hardening and "
    "miniaturization of electronics.222 Despite these challenges, Russian "
    "President Vladimir Putin recently announced a number of initiatives "
    "suggesting that Russia intends to aggressively address its shortfalls in "
    "space.223 219 “Рогозин предупредил о необратимых последствиях "
    "размещения оружия США в космосе [Rogozin warned about the irreversible "
    "consequences of placing U.S. weapons in space],” VPK, March 14, 2018, "
    "https://vpk-news.ru/news/41695; Vladimir Kozin, “Pentagon Rushes Into "
    "Space,” Red Star, 2017, No. 2 37,” accessed March 11, 2018, "
    "https://dlib.eastview.com/search/pub/doc?art=64&id=48594676."
)


def test_bibliografia_pura_es_casi_todo_aparato():
    assert fraccion_aparato(BIBLIOGRAFIA) > 0.9


def test_prosa_con_muchas_citas_numericas_no_se_marca():
    """El caso que descarta la densidad de digitos como senal.

    Si esta prueba empieza a fallar, el detector se volvio agresivo y va a
    hundir contenido bueno: la sec. 9.3.2 exige 10 fragmentos, asi que
    degradar uno util cuesta un cupo real.
    """
    assert fraccion_aparato(PROSA_MUY_CITADA) == 0.0
    assert calidad(PROSA_MUY_CITADA) == 1.0


def test_chunk_mixto_queda_en_el_medio():
    """Ni 0 ni 1: es lo que justifica puntuar por segmento y no por chunk."""
    fraccion = fraccion_aparato(MIXTO)
    assert 0.2 < fraccion < 0.8


def test_la_llamada_pegada_al_punto_corta_igual():
    """'...electronics.222 Despite...' no tiene blanco tras el punto.

    Sin la regla que corta ahi, el segmento de prosa se traga el bloque de
    notas que le sigue, hereda su URL y el chunk entero queda marcado como
    bibliografia (medido: 0.92 en vez de 0.44).
    """
    segmentos = segmentar(MIXTO)
    assert any(s.startswith("Despite these challenges") for s in segmentos)


def test_calidad_es_el_complemento():
    for texto in (BIBLIOGRAFIA, PROSA_MUY_CITADA, MIXTO):
        assert abs(calidad(texto) + fraccion_aparato(texto) - 1.0) < 1e-9


def test_texto_vacio_no_aporta_nada():
    assert fraccion_aparato("") == 1.0
    assert calidad("   ") == 0.0
