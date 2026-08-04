#!/usr/bin/env python3
"""Re-puntua con un encoder secundario leyendo su INDICE COMPLETO, tal como
lo haria `Entrega/generador.py` en produccion.

Existe para cerrar un hueco de validacion: el experimento barato calculo los
vectores del secundario solo para los candidatos del pool. Esto los lee con
`reconstruct(fila)` del indice completo, que es el camino real. Si los dos
caminos NO dan el mismo resultado, hay un problema de alineacion que la
verificacion estructural no detecta -- y se manifestaria como "el encoder
rinde peor de lo medido", no como un error.

Uso:
    python dev/scripts/rerank_indice_completo.py \
        --rerank-index dev/intermedios/encoder_gte-multilingual-base \
        --rerank-encoder gte-multilingual-base --peso 0.25 \
        --out dev/intermedios/gte_rerank/res_gte_full.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from src.config import DEV_DIR, ENCODER_PRIMARY_NAME  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rerank-index", type=Path, required=True)
    ap.add_argument("--rerank-encoder", required=True)
    ap.add_argument("--peso", type=float, default=0.25)
    ap.add_argument("--rerank-depth", type=int, default=200)
    ap.add_argument("--k-pool", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    enc_p = get_encoder(name=ENCODER_PRIMARY_NAME)
    index_p, metadata = load_index(ENCODER_PRIMARY_NAME)
    print(f"primario: {index_p.ntotal:,} vectores", flush=True)

    enc_r = get_encoder(name=args.rerank_encoder)
    index_r = faiss.read_index(str(args.rerank_index / "index.faiss"))
    print(f"secundario: {index_r.ntotal:,} vectores", flush=True)
    if index_r.ntotal != index_p.ntotal:
        raise SystemExit("ABORTA: los indices no tienen el mismo numero de vectores")

    consultas = [json.loads(l) for l in CONSULTAS.open(encoding="utf-8") if l.strip()]
    resultados = []
    for c in consultas:
        hits = search(c["text"], enc_p, index_p, metadata, k=args.rerank_depth)
        qv = enc_r.encode_query(c["text"])
        for h in hits:
            # El vector del secundario se LEE, no se recodifica: los dos
            # indices comparten el orden de filas (invariante del chunking
            # unico). Es lo que hace que la cascada cueste una vectorizacion
            # por consulta y no 200.
            h.score = h.score + args.peso * float(np.dot(qv, index_r.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        hits = hits[: args.k_pool]
        for i, h in enumerate(hits, 1):
            h.rank = i
        resultados.append(generador.build_result_object(c["query_id"], hits))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"escrito {args.out}", flush=True)


if __name__ == "__main__":
    main()
