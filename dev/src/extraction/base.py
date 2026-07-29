"""Tipo comun devuelto por todos los extractores de formato."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawDocument:
    """Resultado crudo (pre-limpieza) de extraer texto de un archivo fuente.

    `texto_crudo` conserva saltos de parrafo (\n\n) y, cuando el formato lo
    permite, encabezados marcados con '# '/'## ' para que el chunking
    estructural (src/chunking) los pueda usar como senales de segmentacion.
    """

    source_path: Path
    formato: str  # pdf | html | md | json | csv | xlsx | img | pbf
    texto_crudo: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)
