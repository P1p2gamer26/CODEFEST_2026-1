"""Extraccion de texto desde JSON (articulos/paginas web estructurados).

Estrategia (sec. 2.1 de la especificacion): interpretar el objeto y
seleccionar explicitamente los campos de texto conocidos, concatenandolos en
su orden de aparicion; si el cuerpo es una lista de parrafos se unen
preservando ese orden. Los campos puramente descriptivos (url, date, authors,
tags) se conservan como metadata en vez de mezclarse con el cuerpo.

Formas reales del corpus de ADL que motivan cada rama:

- Articulos de think tank (la mayoria): `title` + `body_paragraphs` +
  `body_text`. `body_text` es el mismo contenido ya unido, asi que se toma
  UNO de los dos, no ambos -- duplicarlo inflaria el chunking y sesgaria los
  embeddings.
- Papers de revista (~80 archivos): solo `abstract`, sin cuerpo.
- Alertas Tempranas de la Defensoria (~363 archivos, F3): el `title` es
  literalmente "Mapa" y lo sustantivo vive en `body_paragraphs` y en el dict
  `alerta_meta` (`tema_clave`, `tipo`, `fecha_emision`, municipios...). Por eso
  los campos estructurados se vuelcan SIEMPRE como pares "clave: valor", ademas
  del cuerpo.

Si el objeto no tiene ninguno de los campos conocidos, se usa un fallback
generico que concatena todas las hojas de tipo string del arbol JSON.
"""

import json
from pathlib import Path
from typing import Any

from .base import RawDocument

TITLE_FIELDS = ["title", "titulo", "headline"]
BODY_LIST_FIELDS = ["body_paragraphs", "paragraphs", "paragraphs_text"]
BODY_TEXT_FIELDS = ["body_text", "text", "content", "body"]
# Se usan como cuerpo solo si no hubo ninguno de los anteriores.
RESUMEN_FIELDS = ["abstract", "resumen", "excerpt", "summary"]
# Dicts/listas con contenido sustantivo que hay que aplanar a texto.
STRUCT_FIELDS = ["alerta_meta", "fields", "sections", "metadata", "keywords", "topics"]
METADATA_FIELDS = ["url", "date", "published_date", "authors", "author", "tags", "source", "doi"]

# Claves de estructuras que son enlaces o ruido de scraping, no contenido.
_CLAVES_RUIDO = {"url", "href", "link", "src", "image", "images", "pdf_links", "doc_links"}


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


def _render_estructura(node: Any, prefijo: str = "") -> list[str]:
    """Aplana un dict/lista a lineas "clave: valor", preservando el orden y
    descartando las claves que solo contienen enlaces."""
    lineas: list[str] = []
    if isinstance(node, dict):
        for clave, valor in node.items():
            if str(clave).lower() in _CLAVES_RUIDO:
                continue
            etiqueta = f"{prefijo}{clave}"
            if isinstance(valor, (dict, list)):
                lineas.extend(_render_estructura(valor, prefijo=f"{etiqueta}."))
            elif valor is not None and str(valor).strip():
                lineas.append(f"{etiqueta}: {str(valor).strip()}")
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                lineas.extend(_render_estructura(item, prefijo=prefijo))
            elif item is not None and str(item).strip():
                lineas.append(f"{prefijo.rstrip('.')}: {str(item).strip()}".lstrip(": "))
    return lineas


def _primer_texto(obj: dict, campos: list[str]) -> str | None:
    for field in campos:
        value = obj.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_known_fields(obj: dict) -> str | None:
    parts: list[str] = []

    titulo = _primer_texto(obj, TITLE_FIELDS)
    if titulo:
        parts.append(f"# {titulo}")

    cuerpo: list[str] = []
    for field in BODY_LIST_FIELDS:
        value = obj.get(field)
        if isinstance(value, list) and value:
            cuerpo = [str(p).strip() for p in value if str(p).strip()]
            break
    if not cuerpo:
        texto = _primer_texto(obj, BODY_TEXT_FIELDS) or _primer_texto(obj, RESUMEN_FIELDS)
        if texto:
            cuerpo = [texto]
    parts.extend(cuerpo)

    for field in STRUCT_FIELDS:
        if field in obj:
            parts.extend(_render_estructura(obj[field]))

    # Solo el titulo no es contenido suficiente: se deja que decida el fallback.
    if not cuerpo and len(parts) <= 1:
        return None
    return "\n\n".join(parts)


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
