#!/usr/bin/env python3
"""Cobertura de la extraccion: que documentos del manifest de ADL NO quedaron
en los chunks intermedios.

Un doc_id ausente vale cero en el F1@3 pase lo que pase, asi que conviene
saber cuales faltan y por que (PDF escaneado sin OCR, archivo vacio, formato
no soportado) antes de construir el indice.

Uso:
    python scripts/cobertura_corpus.py
    python scripts/cobertura_corpus.py --chunks intermedios/otro.jsonl
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHUNKS_INTERMEDIOS_PATH, CORPUS_DIR, DOC_ID_MANIFEST_PATH  # noqa: E402


def observatorio(doc_id: str) -> str:
    """'F3-SIPRI-012' -> 'F3-SIPRI'."""
    partes = doc_id.rsplit("-", 1)
    return partes[0] if len(partes) == 2 else doc_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_INTERMEDIOS_PATH)
    parser.add_argument("--manifest", type=Path, default=DOC_ID_MANIFEST_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as f:
        esperados = {fila["doc_id"]: fila["ruta"] for fila in csv.DictReader(f)}

    presentes: Counter[str] = Counter()
    with args.chunks.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                presentes[json.loads(linea)["doc_id"]] += 1

    faltantes = sorted(set(esperados) - set(presentes))
    # doc_id que no salen del manifest: resolve_doc_id() cayo al hash de
    # contenido, o sea que ese archivo no emparejo con ninguna fila del xlsx.
    huerfanos = sorted(set(presentes) - set(esperados))

    print(f"manifest      : {len(esperados)} documentos")
    print(f"con chunks    : {len(presentes) - len(huerfanos)} / {len(esperados)}")
    print(f"chunks totales: {sum(presentes.values())}")

    if faltantes:
        print(f"\nSIN CHUNKS ({len(faltantes)}), por observatorio:")
        for obs, n in Counter(observatorio(d) for d in faltantes).most_common():
            print(f"  {obs:20s} {n}")
        print("\nDetalle (doc_id -> ruta, tamano en KB):")
        for doc_id in faltantes:
            ruta = args.corpus_dir / esperados[doc_id]
            kb = round(ruta.stat().st_size / 1024) if ruta.exists() else "NO EXISTE"
            print(f"  {doc_id:20s} {esperados[doc_id]}  [{kb}]")

    if huerfanos:
        print(f"\nOJO: {len(huerfanos)} doc_id fuera del manifest (hash de contenido):")
        for doc_id in huerfanos[:20]:
            print(f"  {doc_id}")

    sys.exit(1 if faltantes or huerfanos else 0)


if __name__ == "__main__":
    main()
