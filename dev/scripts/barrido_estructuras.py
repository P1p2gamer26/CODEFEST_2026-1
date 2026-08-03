#!/usr/bin/env python3
"""Barrido de ESTRUCTURAS de encoders, con los tres indices ya construidos.

Cuesta cero computo nuevo: ningun pasaje se re-codifica. Los vectores del
secundario se leen con `reconstruct(fila)` y los tres indices comparten el
orden de filas (invariante del chunking unico), asi que cambiar quien es
primario y quien re-puntua es solo aritmetica.

Lo que se prueba, y por que ninguna se habia podido medir antes:

- **gte como primario.** Hasta hoy no existia su indice completo. Es la
  hipotesis con mejor fundamento mecanico: su penalizacion por idioma medida
  es ~0 (contra +0,07 de MiniLM) y el fallo documentado del pool es
  cross-lingual (NBQR/CBRN). Que e5 fallara como primario no dice nada de
  gte: e5 tambien fallaba como re-puntuador y gte no.
- **Cascada triple.** MiniLM trae candidatos y los re-puntuan gte y e5 a la
  vez. Nunca se probo porque hacia falta tener los tres indices.

REGLA FIJADA ANTES DE MEDIR (docs/lecciones_metodologia.md): se adopta una
estructura nueva solo si mejora el NDCG@10 en LAS DOS muestras y no pierde
consultas en la independiente. El promedio no decide. Probar cinco
estructuras y quedarse con el maximo seria sobreajuste; por eso el criterio
exige consistencia entre muestras, no un maximo.

Uso:
    python dev/scripts/barrido_estructuras.py
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
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import cargar_jsonl, f1, ndcg, veredicto_signos  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"

# (nombre, primario, [(secundario, peso), ...])
ESTRUCTURAS = [
    ("entregada: MiniLM->gte", MINILM, [(GTE, 0.25)]),
    ("gte solo", GTE, []),
    ("gte->MiniLM", GTE, [(MINILM, 0.25)]),
    ("gte->e5", GTE, [(E5, 0.25)]),
    ("MiniLM->gte+e5", MINILM, [(GTE, 0.25), (E5, 0.25)]),
]

RERANK_DEPTH = 200
K_POOL = 60


def evaluar_estructura(nombre, primario, secundarios, consultas, cache_idx):
    enc_p = get_encoder(name=primario)
    idx_p, metadata = cache_idx[primario]
    resultados = []
    for c in consultas:
        hits = search(c["text"], enc_p, idx_p, metadata, k=RERANK_DEPTH)
        for sec, peso in secundarios:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(c["text"])
            for h in hits:
                h.score += peso * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:K_POOL]
        for i, h in enumerate(hits, 1):
            h.rank = i
        resultados.append(generador.build_result_object(c["query_id"], hits))
    return {r["query_id"]: r for r in resultados}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "estructuras")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"] and not f.get("anotador")]
    gt_indep = [f for f in gt_todo if not f.get("pool")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} anotadas a mano, {len(gt_indep)} independientes\n")

    # Los tres indices comparten metadata byte-identica: se carga una vez.
    cache_idx = {}
    for nombre in (MINILM, E5, GTE):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            print(f"  falta el indice de {nombre}, se omiten sus estructuras")
            continue
        if not cache_idx:
            idx, meta = load_index(nombre)
            cache_idx[nombre] = (idx, meta)
            metadata = meta
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores")
    print()

    tablas = {}
    for nombre, prim, secs in ESTRUCTURAS:
        if prim not in cache_idx or any(s not in cache_idx for s, _ in secs):
            continue
        res = evaluar_estructura(nombre, prim, secs, consultas, cache_idx)
        salida = args.out_dir / (nombre.replace(" ", "_").replace(">", "").replace("->", "_").replace(":", "") + ".jsonl")
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        tablas[nombre] = res
        print(f"  {nombre}: escrito {salida.name}", flush=True)

    def metricas(res, gt):
        f1s = [f1([d["doc_id"] for d in res[g["query_id"]]["documents"][:3]], set(g["docs_relevantes"]))[2] for g in gt]
        nd = [ndcg(res[g["query_id"]]["fragments"], set(g["docs_relevantes"])) for g in gt]
        return sum(f1s) / len(f1s), sum(nd) / len(nd), f1s, nd

    print(f"\n{'estructura':26s} {'F1(41)':>7s} {'NDCG(41)':>9s} {'F1(10)':>7s} {'NDCG(10)':>9s}")
    base = ESTRUCTURAS[0][0]
    guardadas = {}
    for nombre in tablas:
        a, b, _, nd41 = metricas(tablas[nombre], gt_todo)
        c, d, _, nd10 = metricas(tablas[nombre], gt_indep)
        guardadas[nombre] = (nd41, nd10)
        print(f"{nombre:26s} {a:7.3f} {b:9.3f} {c:7.3f} {d:9.3f}")

    print(f"\nvictorias en NDCG@10 contra '{base}':")
    for nombre in tablas:
        if nombre == base:
            continue
        for etiqueta, i in (("41 anotadas", 0), ("10 indep.", 1)):
            g = p_ = e = 0
            for x, y in zip(guardadas[nombre][i], guardadas[base][i]):
                if abs(x - y) < 1e-9:
                    e += 1
                elif x > y:
                    g += 1
                else:
                    p_ += 1
            print(f"  {nombre:26s} {etiqueta:12s} gana {g}, pierde {p_}, empata {e}")


if __name__ == "__main__":
    main()
