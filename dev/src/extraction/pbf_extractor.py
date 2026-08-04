"""Extraccion de texto desde archivos .pbf del corpus (sec. 2.1, "PBF: mapas").

OJO -- estos .pbf NO son OpenStreetMap Protocol Buffer Binary Format: son
Mapbox Vector Tiles (MVT). El corpus trae 73 tiles de Amazon Underworld bajo
`Amazon_Underworld/tiles/{zoom}/{x}/`, con una sola capa (`au_compilado_R02`)
cuyos features son municipios amazonicos con atributos de economias ilicitas.
Por eso se decodifican con `mapbox-vector-tile` y no con pyosmium/pyrosm.

Siguiendo la especificacion: se recorren las capas, dentro de cada una los
elementos del mapa, y sus atributos se vuelcan como pares "atributo: valor".
Como el mismo elemento se repite en varios niveles de zoom, se deduplica por
la firma de sus atributos -- dentro del archivo, ya que cada tile es un
documento independiente con su propio doc_id.

Cada feature queda como un bloque de texto separado por "\\n\\n", igual que las
filas de un CSV, para que el chunker pueda tratarlo como unidad atomica.
"""

import logging
from pathlib import Path

import mapbox_vector_tile

from .base import RawDocument

logger = logging.getLogger(__name__)

# Atributos que son identificadores internos del tile, sin valor semantico
# para la recuperacion.
_ATRIBUTOS_RUIDO = {"fid", "id"}


def _feature_a_texto(properties: dict) -> str:
    pares = [
        f"{clave}: {str(valor).strip()}"
        for clave, valor in properties.items()
        if str(clave).lower() not in _ATRIBUTOS_RUIDO
        and valor is not None
        and str(valor).strip()
    ]
    return " | ".join(pares)


def extract_pbf(path: Path) -> RawDocument:
    tile = mapbox_vector_tile.decode(path.read_bytes())

    bloques: list[str] = []
    vistos: set[str] = set()
    n_features = 0
    for nombre_capa, capa in tile.items():
        for feature in capa.get("features", []):
            n_features += 1
            texto = _feature_a_texto(feature.get("properties", {}) or {})
            if not texto or texto in vistos:
                continue
            vistos.add(texto)
            bloques.append(f"{nombre_capa} | {texto}")

    return RawDocument(
        source_path=path,
        formato="pbf",
        texto_crudo="\n\n".join(bloques),
        extra_metadata={
            "capas": list(tile.keys()),
            "n_features": n_features,
            "n_features_unicos": len(bloques),
        },
    )


if __name__ == "__main__":
    # Comprobacion minima contra un tile real del corpus.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import CORPUS_DIR

    tiles = sorted((CORPUS_DIR / "F3_Dinamicas_Territoriales").rglob("*.pbf"))
    assert tiles, "no hay tiles .pbf en el corpus"
    doc = extract_pbf(tiles[0])
    assert doc.formato == "pbf"
    assert doc.extra_metadata["capas"], "el tile no declaro ninguna capa"
    assert doc.extra_metadata["n_features"] > 0, "el tile no rindio features"
    assert doc.extra_metadata["n_features_unicos"] <= doc.extra_metadata["n_features"]
    assert "au_" in doc.texto_crudo, "no aparecen atributos de Amazon Underworld"
    print(
        f"OK {tiles[0].name}: capas={doc.extra_metadata['capas']} "
        f"features={doc.extra_metadata['n_features']} "
        f"unicos={doc.extra_metadata['n_features_unicos']} "
        f"chars={len(doc.texto_crudo)}"
    )
    print(doc.texto_crudo[:300])
