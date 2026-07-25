"""Extraccion de texto desde JSON (articulos/paginas web estructurados).

Estrategia (sec. 2.1 de la especificacion): interpretar el objeto y
seleccionar explicitamente los campos de texto conocidos (title,
body_text/body_paragraphs/text/content), concatenandolos en su orden de
aparicion; si el cuerpo es una lista de parrafos se unen preservando ese
orden. Los campos descriptivos (url, date, authors, tags) se conservan como
metadata en vez de mezclarse con el cuerpo. Si el objeto no tiene ninguno de
los campos conocidos, se usa un fallback generico que concatena todas las
hojas de tipo string del arbol JSON, en orden de aparicion.
"""

import json
from pathlib import Path
from typing import Any

from .base import RawDocument

TITLE_FIELDS = ["title", "titulo", "headline"]
BODY_LIST_FIELDS = ["body_paragraphs", "paragraphs", "paragraphs_text"]
BODY_TEXT_FIELDS = ["body_text", "text", "content", "body"]
METADATA_FIELDS = ["url", "date", "published_date", "authors", "author", "tags", "source"]


def _collect_string_leaves(node: Any, out: list[str]) -> None:
    if isinstance(node, str):
        stripped = node.strip()
        if stripped:
            out.append(stripped)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_string_leaves(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_string_leaves(item, out)


def _extract_known_fields(obj: dict) -> str | None:
    parts: list[str] = []

    for field in TITLE_FIELDS:
        if isinstance(obj.get(field), str) and obj[field].strip():
            parts.append(f"# {obj[field].strip()}")
            break

    for field in BODY_LIST_FIELDS:
        value = obj.get(field)
        if isinstance(value, list) and value:
            parts.extend(str(p).strip() for p in value if str(p).strip())
            return "\n\n".join(parts) if parts else None

    for field in BODY_TEXT_FIELDS:
        value = obj.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
            return "\n\n".join(parts)

    return None


def extract_json(path: Path) -> RawDocument:
    raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    # Si el archivo es una lista de articulos, se procesa cada uno y se
    # concatena (caso comun de "dumps" de scraping con varios articulos).
    objetos = raw if isinstance(raw, list) else [raw]

    textos = []
    extra_metadata: dict[str, Any] = {}
    for obj in objetos:
        if not isinstance(obj, dict):
            continue
        texto = _extract_known_fields(obj)
        if texto is None:
            leaves: list[str] = []
            _collect_string_leaves(obj, leaves)
            texto = "\n\n".join(leaves)
        if texto.strip():
            textos.append(texto.strip())

        for field in METADATA_FIELDS:
            if field in obj and field not in extra_metadata:
                extra_metadata[field] = obj[field]

    return RawDocument(
        source_path=path,
        formato="json",
        texto_crudo="\n\n".join(textos),
        extra_metadata=extra_metadata,
    )
