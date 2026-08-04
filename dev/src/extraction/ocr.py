"""Acceso comun a tesseract para los extractores de PDF escaneado e imagen.

`pytesseract` busca el binario en el PATH. En Windows el instalador de
UB-Mannheim no lo agrega, asi que se prueban las rutas habituales antes de
rendirse. Si no hay binario, `get_pytesseract()` devuelve None y el extractor
sigue sin OCR en vez de romper la corrida.
"""

import logging
import os
import shutil
from functools import lru_cache

logger = logging.getLogger(__name__)

IDIOMAS = "spa+eng+por"

_RUTAS_CANDIDATAS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


def _ruta_binario() -> str | None:
    ruta = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if ruta:
        return ruta
    for candidata in _RUTAS_CANDIDATAS:
        if os.path.isfile(candidata):
            return candidata
    return None


@lru_cache(maxsize=1)
def get_pytesseract():
    """Modulo pytesseract listo para usar, o None si no hay OCR disponible."""
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract no instalado: el corpus se procesa sin OCR")
        return None

    ruta = _ruta_binario()
    if ruta is None:
        logger.warning("binario de tesseract no encontrado: el corpus se procesa sin OCR")
        return None

    pytesseract.pytesseract.tesseract_cmd = ruta
    logger.info("OCR disponible: %s", ruta)
    return pytesseract
