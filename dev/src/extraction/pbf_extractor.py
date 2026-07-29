"""Extraccion de texto desde archivos PBF (mapas).

NO IMPLEMENTADO en esta iteracion: es un formato nicho (Protocol Buffer
Binary Format, tipico de extractos de OpenStreetMap) que requiere una
libreria dedicada (p. ej. `osmium` o `pyrosm`) para recorrer capas y
elementos (municipios, zonas) y volcar sus atributos a texto como pares
"atributo: valor", deduplicando entre niveles de zoom (sec. 2.1 de la
especificacion). Se deja esta interfaz lista para no bloquear una futura
implementacion cuando el equipo tenga archivos PBF reales de ADL con los que
validar el parseo.
"""

from pathlib import Path

from .base import RawDocument


def extract_pbf(path: Path) -> RawDocument:
    raise NotImplementedError(
        f"Extraccion de PBF no implementada (archivo: {path}). "
        "Requiere 'osmium' o 'pyrosm'; ver docstring del modulo."
    )
