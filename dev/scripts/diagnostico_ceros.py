#!/usr/bin/env python3
"""Por que una consulta da F1@3 = 0: el documento relevante no esta en el
indice, no entra al pool, o entra pero pierde la agregacion.

Distinguir los tres casos decide que se puede arreglar y que no:

  AUSENTE     el doc_id no tiene ni un chunk indexado -> irrecuperable, punto.
  FUERA       ningun chunk suyo cae en el top-K denso -> el encoder no lo ve;
              solo lo arreglaria otro encoder o otra representacion.
  EN EL POOL  tiene chunks bien rankeados pero la agregacion a documento lo
              deja fuera del top-3 -> ESTO si se arregla, es el unico caso
              donde tocar k_pool o la estrategia de agregacion puede servir.

Uso:
    python dev/scripts/diagnostico_ceros.py
    python dev/scripts/diagnostico_ceros.py --k 3000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, ENCODER_PRIMARY_NAME, N_DOCUMENTS_PER_QUERY  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.search import search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import f1  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--k-pool", type=int, default=60, help="el de la entrega")
    parser.add_argument(
        "--k", type=int, default=1000, help="hasta que profundidad se busca el relevante"
    )
    args = parser.parse_args()

    gt = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                fila = json.loads(linea)
                gt[fila["query_id"]] = set(fila["docs_relevantes"])

    consultas = {}
    with CONSULTAS_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                c = json.loads(linea)
                if c["query_id"] in gt:
                    consultas[c["query_id"]] = c["text"]

    encoder = get_encoder(name=args.encoder_name)
    index, metadata = load_index(args.encoder_name, index_dir=args.index_dir)
    indexados = {m["doc_id"] for m in metadata}
    print(f"encoder: {encoder.name} -> {index.ntotal} vectores, {len(indexados)} documentos\n")

    conteo = {"AUSENTE": 0, "FUERA": 0, "EN EL POOL": 0}
    for qid in sorted(consultas):
        relevantes = gt[qid]
        hits = search(consultas[qid], encoder, index, metadata, k=args.k)
        docs_top3 = [d.doc_id for d in aggregate_documents(
            hits[: args.k_pool], top_n=N_DOCUMENTS_PER_QUERY, strategy="sum"
        )]
        if f1(docs_top3, relevantes)[2] > 0:
            continue

        # Mejor rank alcanzado por CUALQUIER chunk de CUALQUIER documento relevante.
        mejor_rank = None
        mejor_doc = None
        for h in hits:
            if h.doc_id in relevantes:
                mejor_rank, mejor_doc = h.rank, h.doc_id
                break

        faltan_del_indice = sorted(relevantes - indexados)
        if len(faltan_del_indice) == len(relevantes):
            caso, detalle = "AUSENTE", f"ninguno indexado: {faltan_del_indice}"
        elif mejor_rank is None:
            caso, detalle = "FUERA", f"ningun chunk suyo en el top-{args.k}"
        elif mejor_rank <= args.k_pool:
            caso = "EN EL POOL"
            detalle = f"{mejor_doc} tiene un chunk en rank {mejor_rank} y aun asi no entra al top-3"
        else:
            caso, detalle = "FUERA", f"el primer chunk de {mejor_doc} esta en rank {mejor_rank}"
        conteo[caso] = conteo.get(caso, 0) + 1
        print(f"{qid}  {caso:<11s} {detalle}")
        print(f"        consulta: {consultas[qid][:100]}")
        print(f"        esperados: {sorted(relevantes)}  devueltos: {docs_top3}")

    print("\nresumen: " + ", ".join(f"{k} {v}" for k, v in conteo.items()))
    print(
        "Solo los 'EN EL POOL' son arreglables reordenando; los 'FUERA' necesitan\n"
        "otra representacion y los 'AUSENTE' no tienen arreglo."
    )


if __name__ == "__main__":
    main()
