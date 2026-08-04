#!/usr/bin/env python3
"""Cuenta victorias por consulta en NDCG@10 entre dos resultados.jsonl.

eval_mini --comparar-con cuenta victorias en F1@3 y reporta el delta del
NDCG solo como promedio. Para las decisiones sobre el ranking de FRAGMENTOS
el promedio no alcanza (misma razon de siempre: ver docs/lecciones_metodologia.md),
hace falta el conteo.

Uso:
    python dev/scripts/victorias_ndcg.py A.jsonl B.jsonl [--todas]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mini import GROUND_TRUTH_MINI_PATH, cargar_jsonl, ndcg, veredicto_signos  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--todas", action="store_true", help="Incluye las etiquetadas por agentes.")
    ap.add_argument("--sin-pooling", action="store_true", help="Solo las 10 independientes.")
    args = ap.parse_args()

    gt = []
    for fila in cargar_jsonl(GROUND_TRUTH_MINI_PATH):
        if not fila["docs_relevantes"]:
            continue
        if not args.todas and fila.get("anotador"):
            continue
        if args.sin_pooling and fila.get("pool"):
            continue
        gt.append(fila)

    A = {r["query_id"]: r for r in cargar_jsonl(args.a)}
    B = {r["query_id"]: r for r in cargar_jsonl(args.b)}

    gana_a = gana_b = empate = 0
    suma_a = suma_b = 0.0
    print(f"{'consulta':10s} {args.a.stem[:16]:>16s} {args.b.stem[:16]:>16s}")
    for fila in gt:
        q = fila["query_id"]
        rel = set(fila["docs_relevantes"])
        va = ndcg(A[q].get("fragments", []), rel)
        vb = ndcg(B[q].get("fragments", []), rel)
        suma_a += va
        suma_b += vb
        if abs(va - vb) < 1e-9:
            empate += 1
        elif va > vb:
            gana_a += 1
            print(f"{q:10s} {va:16.3f} {vb:16.3f}  <-")
        else:
            gana_b += 1
            print(f"{q:10s} {va:16.3f} {vb:16.3f}  ->")

    n = len(gt)
    print(f"\nNDCG@10 medio sobre {n}: {args.a.stem} {suma_a / n:.3f} | {args.b.stem} {suma_b / n:.3f}")
    print(f"{args.a.stem} gana en {gana_a}, {args.b.stem} gana en {gana_b}, empatan {empate}")
    print(veredicto_signos(args.a.stem, gana_a, args.b.stem, gana_b))


if __name__ == "__main__":
    main()
