#!/usr/bin/env python3
"""Inspeccion cualitativa manual de resultados.jsonl frente a las consultas
de prueba: para cada consulta, imprime los documentos y fragmentos
recuperados en un formato legible por humanos.

Esto NO es una metrica de evaluacion oficial (no hay ground truth publico
para el corpus de ejemplo). Es un sustituto informal para verificar que el
pipeline retorna resultados con sentido tematico mientras no se cuenta con
el conjunto oficial de 50 consultas ni su ground truth.

Uso:
    python scripts/inspect_results.py \
        --consultas consultas_prueba/consultas_prueba.jsonl \
        --resultados Entrega/resultados.jsonl \
        --index-dir intermedios/demo_index_fake_encoder/encoder_.../
"""

import argparse
import json
import textwrap
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consultas", type=Path, required=True)
    parser.add_argument("--resultados", type=Path, required=True)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Carpeta con metadata.jsonl, para mostrar 'fuente' legible en vez de doc_id",
    )
    parser.add_argument("--max-chars", type=int, default=160)
    args = parser.parse_args()

    consultas_by_id = {c["query_id"]: c["text"] for c in load_jsonl(args.consultas)}
    resultados = load_jsonl(args.resultados)

    fuente_by_doc_id: dict[str, str] = {}
    if args.index_dir:
        metadata = load_jsonl(args.index_dir / "metadata.jsonl")
        for m in metadata:
            fuente_by_doc_id[m["doc_id"]] = m["fuente"]

    for r in resultados:
        query_text = consultas_by_id.get(r["query_id"], "(consulta desconocida)")
        print("=" * 100)
        print(f"[{r['query_id']}] {query_text}")
        print("-" * 100)
        print("Documentos:")
        for d in r["documents"]:
            fuente = fuente_by_doc_id.get(d["doc_id"], d["doc_id"])
            print(f"  {d['rank']}. {fuente}")
        print("Fragmentos:")
        for frag in r["fragments"]:
            fuente = fuente_by_doc_id.get(frag["doc_id"], frag["doc_id"])
            snippet = textwrap.shorten(frag["text"], width=args.max_chars, placeholder="...")
            print(f"  {frag['rank']:2d}. [{fuente}] {snippet}")
        print()


if __name__ == "__main__":
    main()
