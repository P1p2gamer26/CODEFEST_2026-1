"""doc_id: identificador unico e inmutable por documento (sec. 2.3).

Por defecto se deriva del hash del CONTENIDO del archivo (no de su ruta ni de
su nombre), para que sea estable frente a renombres o reubicaciones.

IMPORTANTE -- aclarado por ADL en la sesion de Q&A final (ver
`docs/Explicacion_reto_final.md`): el emparejamiento a nivel de documento
contra el ground truth se hace con el `doc_id` que ADL suministra junto con
los datos, NO con uno que asigne el equipo. La frase de la sec. 10.2.1 del
PDF que decia que la clave era `fuente` fue un error de versionamiento del
documento y esta siendo corregida.

Por eso `resolve_doc_id()` prioriza el manifest de ADL cuando existe, y solo
cae al hash de contenido cuando no hay entrada para ese archivo (caso del
corpus de ejemplo, que no trae manifest). Asi el pipeline funciona igual con
o sin manifest, y el dia que llegue el corpus real basta con pasarlo.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DOC_ID_LENGTH = 16

# Nombres de campo candidatos dentro del manifest de ADL. Todavia no se
# conoce la convencion exacta que usaran; si el manifest real trae otra,
# ampliar estas dos tuplas es el unico cambio necesario.
_FILENAME_KEYS = ("fuente", "archivo", "filename", "file", "nombre", "path", "ruta")
_DOC_ID_KEYS = ("doc_id", "docid", "id", "documento_id")


def compute_doc_id(path: Path) -> str:
    """doc_id de respaldo: hash del contenido. Se usa cuando ADL no provee
    un doc_id para ese archivo."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:DOC_ID_LENGTH]


def _pick(row: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value:
            return str(value)
    return None


def _rows_to_manifest(rows) -> dict[str, str]:
    """Normaliza filas {campo: valor} a {nombre_de_archivo: doc_id}. Se
    indexa por nombre de archivo (no por ruta) para que el manifest siga
    sirviendo aunque el corpus se reubique en disco."""
    manifest: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        filename = _pick(row, _FILENAME_KEYS)
        doc_id = _pick(row, _DOC_ID_KEYS)
        if filename and doc_id:
            manifest[Path(filename).name] = doc_id
    return manifest


def load_doc_id_manifest(path: Path) -> dict[str, str]:
    """Carga el mapeo {nombre_de_archivo: doc_id} que entregue ADL.

    Acepta las formas mas probables sin saber todavia cual usara ADL: JSON
    (objeto plano `{"archivo.pdf": "DOC-042"}` o lista de objetos), JSONL, y
    CSV con cabecera. Si llega en un formato distinto, adaptar SOLO esta
    funcion -- el resto del pipeline no cambia.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            manifest = _rows_to_manifest(csv.DictReader(f))
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            manifest = _rows_to_manifest(json.loads(line) for line in f if line.strip())
    else:  # .json
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Objeto plano {archivo: doc_id}.
            manifest = {Path(k).name: str(v) for k, v in data.items() if v}
        else:
            manifest = _rows_to_manifest(data)

    if not manifest:
        raise ValueError(
            f"el manifest de doc_id en {path} no produjo ninguna entrada; "
            f"revisar que traiga un campo de archivo ({'/'.join(_FILENAME_KEYS)}) "
            f"y uno de id ({'/'.join(_DOC_ID_KEYS)})"
        )
    logger.info("manifest de doc_id cargado: %d entradas desde %s", len(manifest), path)
    return manifest


def resolve_doc_id(path: Path, manifest: dict[str, str] | None = None) -> str:
    """doc_id oficial del documento: el de ADL si esta en el manifest, si no
    el hash de contenido. `manifest` se indexa por nombre de archivo."""
    if manifest:
        doc_id = manifest.get(path.name)
        if doc_id:
            return doc_id
        logger.warning(
            "sin doc_id de ADL para %s, se usa el hash de contenido "
            "(no emparejara con el ground truth si el documento si estaba en el manifest)",
            path.name,
        )
    return compute_doc_id(path)
