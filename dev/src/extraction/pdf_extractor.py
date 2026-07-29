"""Extraccion de texto desde PDF, preservando el orden de lectura por pagina.

Tablas e imagenes decorativas se omiten si no aportan texto legible (sec. 2.1
de la especificacion). No se hace deteccion de layout multi-columna avanzada:
para PDFs de una o dos columnas simples (el caso tipico de informes de
observatorios/think tanks) el orden top-to-bottom, left-to-right de
pdfplumber es suficiente; layouts mas complejos son una limitacion conocida
(ver informe_tecnico.pdf).
"""

from collections import Counter
from pathlib import Path

import pdfplumber

from .base import RawDocument

# Umbral: una linea que se repite (identica) en al menos esta fraccion de
# paginas se considera boilerplate (cabecera/pie de pagina/numeracion) y se
# descarta (sec. 2.2 de la especificacion). Solo se aplica con >=3 paginas
# para no penalizar documentos cortos donde una frase corta podria repetirse
# de forma legitima.
BOILERPLATE_MIN_PAGE_RATIO = 0.3
BOILERPLATE_MIN_PAGES = 3


def _strip_boilerplate(paginas_lineas: list[list[str]]) -> list[list[str]]:
    n_paginas = len(paginas_lineas)
    if n_paginas < BOILERPLATE_MIN_PAGES:
        return paginas_lineas

    conteo = Counter(
        linea.strip()
        for lineas in paginas_lineas
        for linea in lineas
        if linea.strip()
    )
    umbral = max(2, int(n_paginas * BOILERPLATE_MIN_PAGE_RATIO))
    boilerplate = {linea for linea, n in conteo.items() if n >= umbral}

    return [
        [linea for linea in lineas if linea.strip() not in boilerplate]
        for lineas in paginas_lineas
    ]


def _lines_to_paragraphs(lineas: list[str]) -> list[str]:
    """Reagrupa lineas de una pagina en parrafos: lineas consecutivas no
    vacias se unen con espacio (son el mismo parrafo, cortado por el ancho
    de pagina de pdfplumber); una linea vacia marca un salto de parrafo.
    Sin esto, una oracion que envuelve a la siguiente linea del PDF quedaria
    con un salto de linea "pegado" entre dos palabras en vez de un espacio.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for linea in lineas:
        if linea.strip():
            current.append(linea.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def extract_pdf(path: Path) -> RawDocument:
    paginas_lineas: list[list[str]] = []
    with pdfplumber.open(path) as pdf:
        n_paginas = len(pdf.pages)
        for pagina in pdf.pages:
            texto_pagina = pagina.extract_text() or ""
            paginas_lineas.append(texto_pagina.splitlines())

    paginas_lineas = _strip_boilerplate(paginas_lineas)
    paginas_texto = ["\n\n".join(_lines_to_paragraphs(lineas)) for lineas in paginas_lineas]
    paginas_texto = [p for p in paginas_texto if p]

    texto_crudo = "\n\n".join(paginas_texto)
    return RawDocument(
        source_path=path,
        formato="pdf",
        texto_crudo=texto_crudo,
        extra_metadata={"n_paginas": n_paginas},
    )
