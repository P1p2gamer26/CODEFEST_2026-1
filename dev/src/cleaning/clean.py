"""Limpieza basica y conservadora del texto extraido (sec. 2.2).

Deliberadamente NO se hace lowercasing, stemming ni remocion de puntuacion:
la evaluacion del reto compara el campo `text` de los fragmentos de forma
textual contra el ground truth, asi que cualquier normalizacion agresiva del
contenido perjudicaria la puntuacion en vez de ayudar.
"""

import re
import unicodedata

# Guion de fin de linea de los PDF: pypdfium2 lo entrega como U+FFFE (el
# glifo de guion sin mapeo Unicode en la fuente incrustada), y tambien
# aparece el soft hyphen real U+00AD. Sin repararlo, el 30% de los chunks
# del corpus real llevan palabras partidas ("satel...lites", "weap...ons") y
# el tokenizer del encoder fragmenta justo los terminos de dominio que
# discriminan la consulta. Medido: 38.397 de 128.526 chunks, 613 documentos.
_GUION_CORTE = "￾­"
_CORTE_RE = re.compile(f"(\\w*)[{_GUION_CORTE}](\\w*)")

_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x00, 0x20) if chr(c) not in "\n\t") + chr(0x7F) + "]"
)
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def _unir(izq: str, der: str, mapa: dict[tuple[str, str], str] | None) -> str:
    """Decide si un corte de linea era una palabra partida ("satellites") o
    un guion legitimo ("AI-related")."""
    if not izq or not der:
        return izq + der  # guion suelto o al borde de la linea: sobra
    if mapa is not None:
        decidido = mapa.get((izq.lower(), der.lower()))
        if decidido is not None:
            return izq + decidido + der
    # Sin evidencia del corpus: mayuscula/digito de un lado delata un
    # compuesto real ("AI-related", "COVID-19", "US-based"); minuscula+
    # minuscula es de lejos el caso mas comun y casi siempre una palabra
    # partida. Una sigla en mayusculas a la izquierda cuenta como compuesto
    # salvo que sea larga (un titulo TODO EN MAYUSCULAS si se parte).
    sigla_izq = izq.isupper() and len(izq) <= 4
    if der[0].isupper() or der[0].isdigit() or izq[-1].isdigit() or sigla_izq:
        return f"{izq}-{der}"
    return izq + der


def reparar_cortes(texto: str, mapa: dict[tuple[str, str], str] | None = None) -> str:
    """Rehace las palabras partidas por el guion de fin de linea del PDF.

    `mapa` es el desempate empirico que construye scripts/reparar_guiones.py
    contando en el propio corpus la forma unida contra la forma con guion
    ("open-source" gana, "satellites" gana). Sin mapa cae en la heuristica.
    """
    return _CORTE_RE.sub(lambda m: _unir(m.group(1), m.group(2), mapa), texto)


def clean_text(texto: str, mapa_guiones: dict[tuple[str, str], str] | None = None) -> str:
    if not texto:
        return ""

    texto = unicodedata.normalize("NFC", texto)
    texto = reparar_cortes(texto, mapa_guiones)
    texto = _CONTROL_CHARS_RE.sub("", texto)
    texto = _SPACES_RE.sub(" ", texto)
    texto = _TRAILING_SPACE_RE.sub("\n", texto)
    texto = _BLANK_LINES_RE.sub("\n\n", texto)
    return texto.strip()
