#!/usr/bin/env python3
"""Barrido de --k-pool x --agg-strategy contra el mini ground truth.

Los 3 documentos de cada consulta salen de agregar los k_pool chunks mejor
puntuados (sec. 8.6). Con k_pool=30 sobre 128k chunks, y un corpus lleno de
series casi identicas (MAPPOEA I-XXXVI, CSIS ano a ano), esos 30 candidatos
caen casi todos en 2-3 documentos y el correcto ni entra al pool. Este script
mide el efecto de ampliarlo y de cambiar la estrategia de agregacion, sin
tocar nada de la entrega.

Carga el indice UNA vez y evalua todas las combinaciones en memoria: correr
Entrega/generador.py una vez por combinacion recargaria el encoder (~19 s) y
el indice cada vez.

Uso:
    python dev/scripts/barrido_retrieval.py
    python dev/scripts/barrido_retrieval.py --index-dir dev/intermedios/base_prueba/encoder_multilingual-e5-base
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

K_POOLS = [10, 15, 20, 30, 40, 50, 60, 80, 100, 300]
ESTRATEGIAS = ["max", "sum", "mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument(
        "--sin-pooling",
        action="store_true",
        help="Solo consultas anotadas de forma independiente. OBLIGATORIO para calibrar "
        "k_pool: las anotadas por pooling salieron de un k_pool concreto "
        "(anotar_candidatos.K_POOL), asi que favorecen a ese valor.",
    )
    args = parser.parse_args()

    gt = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                fila = json.loads(linea)
                if args.sin_pooling and fila.get("pool"):
                    continue
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
    print(f"encoder: {encoder.name} -> {index.ntotal} vectores, {len(gt)} consultas anotadas\n")

    # Se recupera UNA vez con el k_pool mayor y se recortan los prefijos: los
    # primeros k hits de una busqueda de 1000 son exactamente los mismos que
    # los de una busqueda de k (el indice es exacto, IndexFlatIP).
    k_max = max(K_POOLS)
    hits_por_consulta = {
        qid: search(texto, encoder, index, metadata, k=k_max) for qid, texto in consultas.items()
    }

    print(f"{'k_pool':>7s} " + " ".join(f"{e:>7s}" for e in ESTRATEGIAS))
    mejor = (0.0, None)
    for k_pool in K_POOLS:
        fila = []
        for estrategia in ESTRATEGIAS:
            suma = 0.0
            for qid, relevantes in gt.items():
                docs = [
                    d.doc_id
                    for d in aggregate_documents(
                        hits_por_consulta[qid][:k_pool],
                        top_n=N_DOCUMENTS_PER_QUERY,
                        strategy=estrategia,
                    )
                ]
                suma += f1(docs, relevantes)[2]
            promedio = suma / len(gt)
            fila.append(promedio)
            if promedio > mejor[0]:
                mejor = (promedio, (k_pool, estrategia))
        print(f"{k_pool:7d} " + " ".join(f"{v:7.3f}" for v in fila))

    print(f"\nmejor: k_pool={mejor[1][0]} agg={mejor[1][1]} -> F1@3 = {mejor[0]:.3f}")


if __name__ == "__main__":
    main()
