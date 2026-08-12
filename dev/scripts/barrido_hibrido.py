#!/usr/bin/env python3
"""Denso vs. BM25 vs. hibrido RRF, contra el mini ground truth.

El recuperador de la entrega es puramente denso. La hipotesis lexica dice que
pierde los terminos discriminantes raros (el caso NBQR/CBRN). Este script la
mide sin tocar nada de la entrega.

**Lo que decide es el conteo de victorias por consulta, no el promedio.** Con
41 consultas cada una pesa 0.024: dos que cambien de lado por azar mueven la
media mas que cualquier efecto real. Se aplica la misma prueba de signos que
usan eval_mini y barrido_retrieval.

Carga el indice y construye BM25 UNA vez (BM25 tokeniza 128k chunks, ~1 min).

Uso:
    python dev/scripts/barrido_hibrido.py
    python dev/scripts/barrido_hibrido.py --sin-pooling
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, ENCODER_PRIMARY_NAME, N_DOCUMENTS_PER_QUERY  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.bm25 import BM25  # noqa: E402
from src.retrieval.fusion import rebuild_hits_from_fusion, reciprocal_rank_fusion  # noqa: E402
from src.retrieval.search import search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import f1, veredicto_signos  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"


def docs_de(hits, k_pool: int) -> list[str]:
    return [
        d.doc_id
        for d in aggregate_documents(hits[:k_pool], top_n=N_DOCUMENTS_PER_QUERY, strategy="sum")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--k-pool", type=int, default=60)  # el de la entrega, ver dev/docs/PLAN_MAESTRO.md
    parser.add_argument(
        "--sin-pooling",
        action="store_true",
        help="Solo las consultas anotadas de forma independiente: las anotadas por "
        "pooling salieron de los candidatos que propuso el recuperador denso, asi "
        "que favorecen al denso frente a cualquier alternativa.",
    )
    parser.add_argument(
        "--solo-humanas",
        action="store_true",
        help="Excluye las 9 consultas etiquetadas por el las rondas de anotacion asistida "
        "(F1 0.23 contra el humano). Obligatorio para comparar con lo medido antes "
        "del 2 ago 2026.",
    )
    args = parser.parse_args()

    gt = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                fila = json.loads(linea)
                if args.sin_pooling and fila.get("pool"):
                    continue
                if args.solo_humanas and fila.get("anotador"):
                    continue
                if not fila["docs_relevantes"]:
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
    print(f"encoder: {encoder.name} -> {index.ntotal} vectores, {len(gt)} consultas anotadas")

    t0 = time.perf_counter()
    bm25 = BM25(metadata)
    print(f"BM25 construido en {time.perf_counter() - t0:.1f} s, {len(bm25.postings)} terminos\n")

    kp = args.k_pool
    # Igual que generador.py: la agregacion a documento suma SCORES, y el score
    # de un hit denso (coseno, ~0.4) y el de uno lexico (BM25, ~20) no son
    # comparables. rebuild_hits_from_fusion reemplaza ambos por el score RRF,
    # que si lo es. Sin este paso la fusion la decide BM25 por magnitud.
    metadata_by_chunk_id = {m["chunk_id"]: m for m in metadata}
    variantes = {"denso": {}, "bm25": {}, "hibrido": {}, "union": {}}
    for qid, texto in consultas.items():
        densos = search(texto, encoder, index, metadata, k=kp)
        lexicos = bm25.search(texto, k=kp)
        fusion = rebuild_hits_from_fusion(
            reciprocal_rank_fusion([densos, lexicos], key=lambda h: h.chunk_id),
            metadata_by_chunk_id,
            limit=kp,
        )
        # UNION: distinta de `hibrido`. La fusion RRF trunca a kp, asi que lo
        # lexico DESPLAZA candidatos densos; la union se queda con los kp
        # densos y le AGREGA los lexicos. Es el rescate tal como funcionaria
        # en produccion -- no puede empeorar por perdida de recall, solo por
        # dilucion del pool.
        union = rebuild_hits_from_fusion(
            reciprocal_rank_fusion([densos, lexicos], key=lambda h: h.chunk_id),
            metadata_by_chunk_id,
            limit=kp * 2,
        )
        variantes["denso"][qid] = docs_de(densos, kp)
        variantes["bm25"][qid] = docs_de(lexicos, kp)
        variantes["hibrido"][qid] = docs_de(fusion, kp)
        variantes["union"][qid] = docs_de(union, kp * 2)

    puntajes = {
        nombre: {qid: f1(docs, gt[qid])[2] for qid, docs in por_consulta.items()}
        for nombre, por_consulta in variantes.items()
    }

    print(f"{'consulta':10s} " + " ".join(f"{n:>9s}" for n in variantes))
    for qid in sorted(consultas):
        print(f"{qid:10s} " + " ".join(f"{puntajes[n][qid]:9.2f}" for n in variantes))
    print(
        f"\n{'F1@3 medio':10s} "
        + " ".join(f"{sum(puntajes[n].values()) / len(gt):9.3f}" for n in variantes)
    )

    for retador in ("bm25", "hibrido", "union"):
        gana_a = gana_b = empate = 0
        for qid in consultas:
            va, vb = puntajes["denso"][qid], puntajes[retador][qid]
            if abs(va - vb) < 1e-9:
                empate += 1
            elif va > vb:
                gana_a += 1
            else:
                gana_b += 1
        print(f"\ndenso gana en {gana_a}, {retador} gana en {gana_b}, empatan {empate}")
        print(veredicto_signos("denso", gana_a, retador, gana_b))


if __name__ == "__main__":
    main()
