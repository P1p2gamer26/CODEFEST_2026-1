"""Limpieza basica y conservadora del texto extraido (sec. 2.2).

Deliberadamente NO se hace lowercasing, stemming ni remocion de puntuacion:
la evaluacion del reto compara el campo `text` de los fragmentos de forma
textual contra el ground truth, asi que cualquier normalizacion agresiva del
contenido perjudicaria la puntuacion en vez de ayudar.
"""

import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x00, 0x20) if chr(c) not in "\n\t") + chr(0x7F) + "]"
)
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def clean_text(texto: str) -> str:
    if not texto:
        return ""

    texto = unicodedata.normalize("NFC", texto)
    texto = _CONTROL_CHARS_RE.sub("", texto)
    texto = _SPACES_RE.sub(" ", texto)
    texto = _TRAILING_SPACE_RE.sub("\n", texto)
    texto = _BLANK_LINES_RE.sub("\n\n", texto)
    return texto.strip()
