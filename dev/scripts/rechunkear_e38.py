#!/usr/bin/env python3
"""E38 -- re-empaqueta el corpus ya extraido a otro presupuesto de tokens.

No re-extrae ni pasa por OCR. El corpus tiene checkpoint de los CHUNKS
(`chunks_intermedios_limpio.jsonl`), no del texto crudo de los documentos, asi
que re-chunkear a un presupuesto MAYOR obliga a reconstruir la secuencia
original de oraciones quitando el solape que el chunker introdujo, y volver a
empaquetar con `_pack_sentences`.

Spec: dev/docs/superpowers/specs/2026-08-09-rejilla-chunking-design.md
Plan: dev/docs/superpowers/plans/2026-08-09-e38-rejilla-chunking.md

Uso:
    # puerta de entrada: la reconstruccion no puede perder texto
    python dev/scripts/rechunkear_e38.py --verificar \
        --chunks dev/intermedios/chunks_intermedios_limpio.jsonl

    # generar una celda de la rejilla
    python dev/scripts/rechunkear_e38.py --generar \
        --chunks dev/intermedios/chunks_intermedios_limpio.jsonl \
        --presupuesto 512 --solape 0 \
        --salida dev/intermedios/chunks_e38_512_0.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.sentence_split import split_sentences

# Los tabulares se fragmentan por filas (`_chunk_tabular_rows`), no por
# presupuesto de tokens: la rejilla no los toca y se copian tal cual.
FORMATOS_TABULARES = ("csv", "xlsx")


def reconstruir_oraciones(chunks_del_doc):
    """Devuelve [(titulo_seccion, [oraciones])] en orden de posicion.

    Quita de cada chunk el prefijo de oraciones que repite la cola del chunk
    anterior de la MISMA seccion. Solo mira la frontera: una repeticion en
    cualquier otro punto es contenido real del documento y se conserva.
    """
    secciones = []
    for ch in sorted(chunks_del_doc, key=lambda c: c["posicion"]):
        oraciones = split_sentences(ch["texto"], ch.get("idioma"))
        if not oraciones:
            continue
        heading = ch.get("titulo_seccion")
        if secciones and secciones[-1][0] == heading:
            previas = secciones[-1][1]
            solape = 0
            for n in range(min(len(previas), len(oraciones)), 0, -1):
                if previas[-n:] == oraciones[:n]:
                    solape = n
                    break
            previas.extend(oraciones[solape:])
        else:
            secciones.append((heading, list(oraciones)))
    return secciones
