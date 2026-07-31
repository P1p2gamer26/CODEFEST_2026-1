"""Extraccion de texto desde PDF, preservando el orden de lectura por pagina.

Tablas e imagenes decorativas se omiten si no aportan texto legible (sec. 2.1
de la especificacion). No se hace deteccion de layout multi-columna avanzada:
para PDFs de una o dos columnas simples (el caso tipico de informes de
observatorios/think tanks) el orden top-to-bottom, left-to-right es
suficiente; layouts mas complejos son una limitacion conocida (ver
informe_tecnico.pdf).

Se usa `pypdfium2` y no `pdfplumber` por VOLUMEN: el corpus real trae 760 PDFs
y unas 30.000 paginas, con archivos de hasta 100 MB. Medido sobre este corpus,
pypdfium2 lee ~3.000 paginas en 4,5 s; pdfplumber tarda dos ordenes de
magnitud mas, lo que convierte la fase offline de minutos en horas. La calidad
del texto plano extraido es equivalente para estos documentos.

Respaldo OCR: ~5% de los PDFs del corpus (casi todos los `ALERTAS_informes*`
de la Defensoria) son escaneos sin capa de texto. Cuando un documento rinde
menos de OCR_MIN_CHARS_POR_PAGINA caracteres por pagina se rasteriza y se pasa
por tesseract. Si el binario de tesseract no esta instalado, el documento
simplemente aporta poco o ningun texto en vez de romper la corrida.
"""

import logging
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from .base import RawDocument
from .ocr import IDIOMAS as OCR_IDIOMAS
from .ocr import get_pytesseract

logger = logging.getLogger(__name__)

# Umbral: una linea que se repite (identica) en al menos esta fraccion de
# paginas se considera boilerplate (cabecera/pie de pagina/numeracion) y se
# descarta (sec. 2.2 de la especificacion). Solo se aplica con >=3 paginas
# para no penalizar documentos cortos donde una frase corta podria repetirse
# de forma legitima.
BOILERPLATE_MIN_PAGE_RATIO = 0.3
BOILERPLATE_MIN_PAGES = 3

# Por debajo de esta densidad de texto se asume que el PDF es un escaneo.
OCR_MIN_CHARS_POR_PAGINA = 200
# Tope de paginas a rasterizar por documento: el OCR es ~1-2 s por pagina y
# los informes escaneados largos no justifican bloquear la corrida completa.
# ponytail: tope fijo; subirlo si algun escaneo largo resulta ser relevante.
OCR_MAX_PAGINAS = 60
OCR_ESCALA = 2  # ~150 dpi sobre una pagina carta; suficiente para texto de informe


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
    de pagina del PDF); una linea vacia marca un salto de parrafo. Sin esto,
    una oracion que envuelve a la siguiente linea quedaria con un salto de
    linea "pegado" entre dos palabras en vez de un espacio.
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


def _ocr_paginas(doc, path: Path) -> list[list[str]]:
    """Rasteriza y pasa por tesseract las primeras OCR_MAX_PAGINAS paginas."""
    pytesseract = get_pytesseract()
    if pytesseract is None:
        logger.warning("sin OCR disponible; el escaneo %s no aporta texto", path.name)
        return []

    n = min(len(doc), OCR_MAX_PAGINAS)
    if len(doc) > n:
        logger.warning(
            "escaneo %s tiene %d paginas; solo se aplica OCR a las primeras %d",
            path.name, len(doc), n,
        )

    paginas_lineas: list[list[str]] = []
    for i in range(n):
        try:
            imagen = doc[i].render(scale=OCR_ESCALA).to_pil()
            texto = pytesseract.image_to_string(imagen, lang=OCR_IDIOMAS)
        except Exception as exc:  # binario tesseract ausente u otro fallo
            logger.warning("OCR fallo en %s pagina %d: %s", path.name, i, exc)
            return paginas_lineas
        paginas_lineas.append(texto.splitlines())
    return paginas_lineas


def extract_pdf(path: Path) -> RawDocument:
    # Dos archivos del corpus (SIPRI_22136.pdf y SIPRI_hsrc20lmip20report...)
    # son en realidad paginas HTML guardadas con extension .pdf: la descarga
    # de ADL fallo y guardo la pagina de error. Se detectan por contenido y se
    # extraen como HTML en vez de perder el documento entero.
    if path.read_bytes()[:1024].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        from .html_extractor import extract_html

        logger.warning("%s no es un PDF sino HTML: se extrae como HTML", path.name)
        return extract_html(path)

    doc = pdfium.PdfDocument(path)
    n_paginas = len(doc)
    paginas_lineas = [
        (doc[i].get_textpage().get_text_range() or "").splitlines()
        for i in range(n_paginas)
    ]

    n_chars = sum(len(linea) for lineas in paginas_lineas for linea in lineas)
    ocr_aplicado = False
    if n_paginas and n_chars < OCR_MIN_CHARS_POR_PAGINA * n_paginas:
        logger.info(
            "%s parece un escaneo (%d chars en %d paginas): se intenta OCR",
            path.name, n_chars, n_paginas,
        )
        lineas_ocr = _ocr_paginas(doc, path)
        if lineas_ocr:
            paginas_lineas = lineas_ocr
            ocr_aplicado = True

    paginas_lineas = _strip_boilerplate(paginas_lineas)
    paginas_texto = ["\n\n".join(_lines_to_paragraphs(lineas)) for lineas in paginas_lineas]
    paginas_texto = [p for p in paginas_texto if p]

    return RawDocument(
        source_path=path,
        formato="pdf",
        texto_crudo="\n\n".join(paginas_texto),
        extra_metadata={"n_paginas": n_paginas, "ocr": ocr_aplicado},
    )
