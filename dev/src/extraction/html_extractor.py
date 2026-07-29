"""Extraccion de texto desde HTML: quita markup/boilerplate (nav, footer,
scripts, anuncios) via trafilatura y conserva encabezados como senales '#'
para el chunking estructural (sec. 2.1 de la especificacion).
"""

from pathlib import Path

import trafilatura

from .base import RawDocument


def extract_html(path: Path) -> RawDocument:
    html = path.read_text(encoding="utf-8", errors="ignore")

    texto_crudo = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    ) or ""

    metadata = trafilatura.extract_metadata(html)
    extra_metadata = {}
    if metadata is not None:
        extra_metadata = {
            "titulo": metadata.title,
            "autor": metadata.author,
            "fecha": metadata.date,
            "sitename": metadata.sitename,
        }

    return RawDocument(
        source_path=path,
        formato="html",
        texto_crudo=texto_crudo,
        extra_metadata=extra_metadata,
    )
