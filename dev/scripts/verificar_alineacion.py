#!/usr/bin/env python3
"""Comprueba que un indice nuevo esta alineado con el primario de la entrega.

La cascada lee el vector del secundario con `reconstruct(fila)` asumiendo que
la fila N es el MISMO chunk en los dos indices. Si no lo es, no hay excepcion
ni aviso: solo resultados peores, atribuibles a "el encoder es malo". Este
script convierte ese fallo silencioso en un fallo ruidoso.

Uso:
    python dev/scripts/verificar_alineacion.py dev/intermedios/encoder_gte-multilingual-base
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ENCODER_PRIMARY_NAME, encoder_dir  # noqa: E402

fallos = 0


def check(cond, ok, mal):
    global fallos
    print(f"  {'OK  ' if cond else 'FALLA'}  {ok if cond else mal}")
    if not cond:
        fallos += 1


def ids(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l)["chunk_id"] for l in f if l.strip()]


def main():
    nuevo = Path(sys.argv[1])
    primario = encoder_dir(ENCODER_PRIMARY_NAME)
    print(f"nuevo:    {nuevo}\nprimario: {primario}\n")

    idx_n = faiss.read_index(str(nuevo / "index.faiss"))
    idx_p = faiss.read_index(str(primario / "index.faiss"))
    ids_n, ids_p = ids(nuevo / "metadata.jsonl"), ids(primario / "metadata.jsonl")

    check(idx_n.ntotal == idx_p.ntotal,
          f"mismo numero de vectores ({idx_n.ntotal:,})",
          f"vectores distintos: {idx_n.ntotal:,} vs {idx_p.ntotal:,}")
    check(len(ids_n) == idx_n.ntotal,
          f"metadata alineada con el indice ({len(ids_n):,} lineas)",
          f"metadata {len(ids_n):,} vs indice {idx_n.ntotal:,}")
    check(ids_n == ids_p,
          "chunk_id identicos y EN EL MISMO ORDEN que el primario",
          "los chunk_id difieren o estan en otro orden -> reconstruct() devolveria el vector equivocado")

    # Los vectores tienen que estar normalizados o IndexFlatIP no es coseno.
    muestra = np.vstack([idx_n.reconstruct(i) for i in range(0, idx_n.ntotal, max(1, idx_n.ntotal // 500))])
    normas = np.linalg.norm(muestra, axis=1)
    check(bool(np.allclose(normas, 1.0, atol=1e-3)),
          f"vectores normalizados (norma media {normas.mean():.4f})",
          f"vectores SIN normalizar (norma media {normas.mean():.4f}) -> IndexFlatIP no seria coseno")
    check(bool(np.isfinite(muestra).all()),
          "sin NaN ni infinitos",
          "hay NaN o infinitos en los vectores")

    print(f"\n{'TODO OK' if not fallos else f'{fallos} FALLO(S) -- no usar este indice'}")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
