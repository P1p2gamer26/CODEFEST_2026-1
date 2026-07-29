"""Despacha cada archivo fuente al extractor correspondiente segun su
extension (config.EXTENSION_TO_FORMATO)."""

from pathlib import Path

from ..config import EXTENSION_TO_FORMATO
from .base import RawDocument
from .html_extractor import extract_html
from .image_extractor import extract_image
from .json_extractor import extract_json
from .pbf_extractor import extract_pbf
from .pdf_extractor import extract_pdf
from .tabular_extractor import extract_csv, extract_xlsx
from .text_extractor import extract_text_md

_DISPATCH = {
    "pdf": extract_pdf,
    "html": extract_html,
    "md": extract_text_md,
    "json": extract_json,
    "csv": extract_csv,
    "xlsx": extract_xlsx,
    "img": extract_image,
    "pbf": extract_pbf,
}


def extract_file(path: Path) -> RawDocument:
    formato = EXTENSION_TO_FORMATO.get(path.suffix.lower())
    if formato is None:
        raise ValueError(f"Extension no soportada: {path.suffix} ({path})")

    extractor = _DISPATCH[formato]
    return extractor(path)
