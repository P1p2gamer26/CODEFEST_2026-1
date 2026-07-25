"""doc_id: identificador unico e inmutable por documento (sec. 2.3).

Se deriva del hash del CONTENIDO del archivo (no de su ruta ni de su nombre),
para que sea estable frente a renombres o reubicaciones y agnostico al
origen -- esta pieza no cambia cuando se reemplace corpus_ejemplo/ por el
corpus real de ADL.
"""

import hashlib
from pathlib import Path

DOC_ID_LENGTH = 16


def compute_doc_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:DOC_ID_LENGTH]
