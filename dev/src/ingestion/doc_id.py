"""doc_id: identificador unico e inmutable por documento (sec. 2.3).

ADL suministra los doc_id oficiales en `Indice_Datos_Codefest.xlsx` (hoja
`Inventario de Archivos`, columna DOC_ID, formato `F1-AIINDEX-001`). El
emparejamiento a nivel documento contra el ground truth se hace con ESE
doc_id, no con uno que asigne el equipo: la frase de la sec. 10.2.1 del PDF
que decia que la clave era `fuente` fue un error de versionamiento, aclarado
por ADL en la Q&A final (ver `docs/Explicacion_reto_final.md`).

IMPORTANTE -- el manifest NO se puede indexar solo por nombre de archivo: hay
59 nombres que aparecen dos veces con DOC_ID distintos (los PDF de CSET bajo
`pdfs/Reports` y `pdfs/Translation`, y los tiles de Amazon Underworld que se
repiten por nivel de zoom). Por eso la clave primaria es la RUTA relativa al
corpus, y el nombre de archivo solo se usa como respaldo cuando es inequivoco.

Cuando no hay entrada en el manifest se cae a un hash del contenido, que sirve
para el corpus de ejemplo pero NO empareja con el ground truth.
"""

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DOC_ID_LENGTH = 16

# Nombres de campo candidatos dentro del manifest de ADL.
_PATH_KEYS = ("ruta", "path", "carpeta_archivo", "relpath")
_FILENAME_KEYS = ("nombre", "fuente", "archivo", "filename", "file")
_DOC_ID_KEYS = ("doc_id", "docid", "id", "documento_id")


def compute_doc_id(path: Path) -> str:
    """doc_id de respaldo: hash del contenido. Se usa cuando ADL no provee
    un doc_id para ese archivo."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:DOC_ID_LENGTH]


def normalize_rel_path(rel_path) -> str:
    """Clave canonica de ruta: separadores a '/' y minusculas, para que el
    manifest (que usa '/') case con las rutas de Windows ('\\')."""
    return str(rel_path).replace("\\", "/").strip("/").lower()


@dataclass
class DocIdManifest:
    """Mapeo de ADL {documento: doc_id}, indexado por ruta relativa y, como
    respaldo, por nombre de archivo inequivoco."""

    por_ruta: dict[str, str] = field(default_factory=dict)
    por_nombre: dict[str, str] = field(default_factory=dict)
    # Nombres que aparecen en mas de una carpeta con doc_id distintos: se
    # excluyen del respaldo por nombre para no asignar el doc_id equivocado.
    nombres_ambiguos: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.por_ruta) or len(self.por_nombre)

    @property
    def doc_ids(self) -> set[str]:
        """Todos los doc_id que ADL reconoce. Lo que no este aqui no es un
        documento del reto y no puede aparecer en el ground truth."""
        return set(self.por_ruta.values()) | set(self.por_nombre.values())

    def lookup(self, rel_path=None, nombre: str | None = None) -> str | None:
        if rel_path is not None:
            doc_id = self.por_ruta.get(normalize_rel_path(rel_path))
            if doc_id:
                return doc_id
        if nombre is not None:
            return self.por_nombre.get(nombre.lower())
        return None


def _pick(row: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        value = row.get(key)
        if value:
            return str(value)
    return None


def _rows_to_manifest(rows) -> DocIdManifest:
    filas = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = _pick(row, _DOC_ID_KEYS)
        if not doc_id:
            continue
        ruta = _pick(row, _PATH_KEYS)
        nombre = _pick(row, _FILENAME_KEYS) or (Path(ruta).name if ruta else None)
        if not nombre:
            continue
        filas.append((ruta, Path(nombre).name, doc_id))

    # Un nombre es ambiguo si aparece con mas de un doc_id.
    por_nombre_todos: dict[str, set[str]] = {}
    for _, nombre, doc_id in filas:
        por_nombre_todos.setdefault(nombre.lower(), set()).add(doc_id)
    ambiguos = {n for n, ids in por_nombre_todos.items() if len(ids) > 1}

    manifest = DocIdManifest(nombres_ambiguos=ambiguos)
    for ruta, nombre, doc_id in filas:
        if ruta:
            manifest.por_ruta[normalize_rel_path(ruta)] = doc_id
        if nombre.lower() not in ambiguos:
            manifest.por_nombre[nombre.lower()] = doc_id
    return manifest


def load_doc_id_manifest(path: Path) -> DocIdManifest:
    """Carga el manifest de doc_id de ADL.

    Acepta CSV con cabecera (lo que produce `scripts/manifest_desde_xlsx.py`),
    JSONL, y JSON (objeto plano `{"ruta/archivo.pdf": "F1-CSET-001"}` o lista
    de objetos). Si ADL cambia el formato, adaptar SOLO esta funcion.
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
            manifest = _rows_to_manifest(
                {"ruta": k, "doc_id": v} for k, v in data.items() if v
            )
        else:
            manifest = _rows_to_manifest(data)

    if not manifest:
        raise ValueError(
            f"el manifest de doc_id en {path} no produjo ninguna entrada; "
            f"revisar que traiga un campo de ruta ({'/'.join(_PATH_KEYS)}) o de "
            f"archivo ({'/'.join(_FILENAME_KEYS)}) y uno de id ({'/'.join(_DOC_ID_KEYS)})"
        )
    logger.info(
        "manifest de doc_id cargado desde %s: %d rutas, %d nombres inequivocos, "
        "%d nombres ambiguos (solo resolubles por ruta)",
        path,
        len(manifest.por_ruta),
        len(manifest.por_nombre),
        len(manifest.nombres_ambiguos),
    )
    return manifest


def resolve_doc_id(
    path: Path,
    manifest: DocIdManifest | None = None,
    corpus_dir: Path | None = None,
) -> str:
    """doc_id oficial del documento: el de ADL si esta en el manifest (por
    ruta relativa a `corpus_dir`, o por nombre si es inequivoco), si no el
    hash de contenido."""
    if manifest:
        rel_path = None
        if corpus_dir is not None:
            try:
                rel_path = path.resolve().relative_to(Path(corpus_dir).resolve())
            except ValueError:
                rel_path = None
        doc_id = manifest.lookup(rel_path=rel_path, nombre=path.name)
        if doc_id:
            return doc_id
        logger.warning(
            "sin doc_id de ADL para %s, se usa el hash de contenido "
            "(no emparejara con el ground truth si el documento si estaba en el manifest)",
            path.name,
        )
    return compute_doc_id(path)
