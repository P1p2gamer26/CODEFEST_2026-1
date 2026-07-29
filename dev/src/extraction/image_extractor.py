"""OCR sobre imagenes (portadas, infografias) cuando contienen texto
relevante (sec. 2.1 de la especificacion). Si tesseract-ocr no esta instalado
en el sistema, se registra el problema y se devuelve un RawDocument vacio en
lugar de romper el pipeline completo -- una imagen sin OCR disponible
simplemente no aporta chunks, no bloquea el resto del corpus.
"""

import logging
from pathlib import Path

from PIL import Image

from .base import RawDocument

logger = logging.getLogger(__name__)


def extract_image(path: Path) -> RawDocument:
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract no disponible; se omite OCR para %s", path)
        return RawDocument(source_path=path, formato="img", texto_crudo="")

    try:
        with Image.open(path) as img:
            texto_crudo = pytesseract.image_to_string(img, lang="spa+eng+por")
    except Exception as exc:  # binario tesseract ausente u otro fallo de OCR
        logger.warning("OCR fallo para %s: %s", path, exc)
        texto_crudo = ""

    return RawDocument(source_path=path, formato="img", texto_crudo=texto_crudo.strip())
