"""Passthrough para Markdown/TXT: el contenido ya es texto plano con marcado
ligero; los encabezados (#, ##) y separadores ya sirven como senales de
segmentacion tal cual vienen (sec. 2.1 de la especificacion).
"""

from pathlib import Path

from .base import RawDocument


def extract_text_md(path: Path, formato: str = "md") -> RawDocument:
    texto_crudo = path.read_text(encoding="utf-8", errors="ignore")
    return RawDocument(source_path=path, formato=formato, texto_crudo=texto_crudo)
