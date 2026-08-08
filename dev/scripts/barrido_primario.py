#!/usr/bin/env python3
"""Barrido del ENCODER PRIMARIO bajo el regimen entregado (E02 de la cola).

`barrido_estructuras.py` ya comparo primarios, pero con k_pool=60, agregacion
`sum` y sin glosario -- las tres cosas cambiaron. Este script repite la
comparacion en la configuracion que hoy produce `Entrega/resultados.jsonl`:
k_pool=100, agregacion top5, glosario activo, cascada triple con peso 0,25.

HIPOTESIS, escrita antes de medir (E02): `multilingual-e5-small` recupera
mejor que MiniLM como primario porque arregla la truncacion a 128 tokens que
afecta al 96% de los chunks, sin costar mas CPU (mismo orden de parametros,
misma dimension 384).

JUSTIFICACION MECANICA: MiniLM no ve la mayor parte del texto que indexa --
`docs/plan_encoders.md` seccion 0 mide que el 96% de los chunks supera su
ventana de 128 tokens. e5-small cuadruplica la ventana con el mismo tamano.
Es la unica forma barata de separar "la ventana no importa" de "e5-base era
peor por otra razon" (e5-base perdio como primario, pero es 3x mas grande y
la ventana no era la unica variable que cambiaba).

Se miden CUATRO celdas, no dos: los dos primarios con la cascada completa
(que es lo que se entregaria) y los dos solos (que aisla el recall del
primario, que es de lo que habla la hipotesis). Un primario puede recuperar
mejor y aun asi perder en cascada, o al reves.

REGLA DE DECISION FIJADA ANTES DE MEDIR: se adopta un primario nuevo solo si
el IC al 90% del delta pareado excluye una perdida de 0,02 EN LAS DOS
muestras (las 50 y las 10 independientes). Ante empate se prefiere MiniLM,
que es el entregado. El cambio toca fragmentos, asi que decide NDCG@10.

Uso:
    python dev/scripts/barrido_primario.py
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import (  # noqa: E402
    bootstrap_delta,
    cargar_jsonl,
    f1,
    ndcg,
    veredicto_bootstrap,
)

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
E5S = "multilingual-e5-small"

# El indice de e5-small no vive en Entrega/ (regla 6 del handoff): se
# construye en dev/intermedios/e5small/ y se carga desde ahi.
E5S_DIR = DEV_DIR / "intermedios" / "e5small" / f"encoder_{E5S}"

RERANK_DEPTH = 200
K_POOL = 100
ESTRATEGIA = "top5"
PESO = 0.25

# (etiqueta, primario, [(secundario, peso), ...])
CELDAS = [
    ("MiniLM->gte+e5 (entregado)", MINILM, [(GTE, PESO), (E5, PESO)]),
    ("e5small->gte+e5", E5S, [(GTE, PESO), (E5, PESO)]),
    ("MiniLM solo", MINILM, []),
    ("e5small solo", E5S, []),
]
BASE = CELDAS[0][0]


def evaluar_celda(primario, secundarios, consultas, cache_idx):
    enc_p = get_encoder(name=primario)
    idx_p, metadata = cache_idx[primario]
    resultados = {}
    for c in consultas:
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)
        for sec, peso in secundarios:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += peso * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        recorte = hits[:K_POOL]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        resultados[c["query_id"]] = generador.build_result_object(
            c["query_id"], recorte, agg_strategy=ESTRATEGIA
        )
    return resultados


def metricas(res, gt):
    f1s, nds = [], []
    for g in gt:
        qid = g["query_id"]
        if qid not in res:
            continue
        rel = set(g["docs_relevantes"])
        f1s.append(f1([d["doc_id"] for d in res[qid]["documents"][:3]], rel)[2])
        nds.append(ndcg(res[qid]["fragments"], rel))
    return f1s, nds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "primario")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} consultas evaluables, {len(gt_indep)} independientes\n")

    # Los cuatro indices comparten metadata byte-identica (invariante del
    # chunking unico, punto 8): se carga una sola vez y se reusa.
    cache_idx = {}
    idx, metadata = load_index(MINILM)
    cache_idx[MINILM] = (idx, metadata)
    for nombre, d in ((E5, encoder_dir(E5)), (GTE, encoder_dir(GTE)), (E5S, E5S_DIR)):
        ruta = d / "index.faiss"
        if not ruta.exists():
            sys.exit(f"falta el indice de {nombre}: {ruta}")
        cache_idx[nombre] = (faiss.read_index(str(ruta)), metadata)
    for nombre, (ix, _) in cache_idx.items():
        print(f"  {nombre}: {ix.ntotal:,} vectores")
        if ix.ntotal != len(metadata):
            sys.exit(f"{nombre} desalineado: {ix.ntotal} vectores vs {len(metadata)} chunks")

    print(f"\nevaluando (pool={K_POOL}, agg={ESTRATEGIA}, glosario=on, peso={PESO})...", flush=True)
    tablas = {}
    for etiqueta, prim, secs in CELDAS:
        res = evaluar_celda(prim, secs, consultas, cache_idx)
        tablas[etiqueta] = res
        salida = args.out_dir / (etiqueta.split(" ")[0].replace("->", "_") + ".jsonl")
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        print(f"  {etiqueta}: escrito {salida.name}", flush=True)

    print(f"\n{'celda':30s} {'F1(50)':>8s} {'NDCG(50)':>9s} {'F1(indep)':>10s} {'NDCG(indep)':>12s}")
    guardadas = {}
    for etiqueta, res in tablas.items():
        f_a, n_a = metricas(res, gt_todo)
        f_i, n_i = metricas(res, gt_indep)
        guardadas[etiqueta] = (f_a, n_a, f_i, n_i)
        print(
            f"{etiqueta:30s} {sum(f_a)/len(f_a):8.3f} {sum(n_a)/len(n_a):9.3f} "
            f"{sum(f_i)/len(f_i):10.3f} {sum(n_i)/len(n_i):12.3f}"
        )

    print(f"\ndeltas pareados contra '{BASE}', IC al 90%:")
    base = guardadas[BASE]
    for etiqueta in tablas:
        if etiqueta == BASE:
            continue
        print(f"\n  === {etiqueta} ===")
        for nombre, i in (("F1@3   50   ", 0), ("NDCG@10 50  ", 1),
                          ("F1@3   indep", 2), ("NDCG@10 indep", 3)):
            pares = [x - y for x, y in zip(guardadas[etiqueta][i], base[i])]
            d, lo, hi = bootstrap_delta(pares)
            g = sum(1 for x, y in zip(guardadas[etiqueta][i], base[i]) if x > y + 1e-9)
            p_ = sum(1 for x, y in zip(guardadas[etiqueta][i], base[i]) if x < y - 1e-9)
            e = len(base[i]) - g - p_
            print(f"    {nombre}  delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"gana {g}, pierde {p_}, empata {e}  -> {veredicto_bootstrap(d, lo, hi)}")


if __name__ == "__main__":
    main()
