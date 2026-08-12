#!/usr/bin/env python3
"""E20: que el conteo de chunks deje de decidir el ranking de documentos.

HIPOTESIS, escrita antes de medir: reservar el tercer cupo para el documento
con el mejor CHUNK individual sube el F1@3, porque los relevantes que hoy
pierden tienen un mejor chunk equivalente al de los que ganan y pierden solo
por aportar menos chunks al pool.

JUSTIFICACION MECANICA (E19). Los que ocupan cupo aportan 7 chunks al pool
(mediana) y saturan el tope de top5 el 92.7% de las veces; los relevantes que
pierden aportan 3 y saturan el 18.5%. Su mejor chunk vale 1.647 contra 1.672:
1.5% de diferencia. Bajo suma de los 5 mejores, un documento con un chunk
excelente vale 1.68 y uno con cinco mediocres vale 8.0 -- ningun chunk es lo
bastante bueno para compensar el conteo.

RIESGOS, fijados ANTES de medir:
  - El mejor chunk individual es lo que hace `max`, que perdio 0.226 contra
    0.306. Pero eso fue max para los TRES cupos con k_pool=60. Si `1+2` no es
    peor que `2+1`, el efecto es ruido y se rechaza.
  - VETO: si la ganancia se concentra en las 9 consultas de etiqueta de
    agente, se rechaza por sesgo (mismo veto que E11 y E12).
  - Regla de E09: la fila base tiene que reproducir 0.440 / 0.490 / 0.476.
  - Se decide con NDCG@10 (el cambio arrastra fragmentos por la alineacion).
  - No toca `Entrega/` (regla 6 del handoff).

Uso:
    python dev/scripts/barrido_cupo_chunk_e20.py
"""

import argparse
import json
import math
import sys
from collections import defaultdict
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
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60

CELDAS = ("entregada", "cupo-2+1", "cupo-1+2", "mean-top5", "top5-sqrt")
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

_agg_original = generador.aggregate_documents
DocumentHit = None  # se resuelve al importar generador


def _scores_por_doc(hits):
    d = defaultdict(list)
    for h in hits:
        d[h.doc_id].append(h.score)
    return d


def _ranked(agg_scores, top_n):
    orden = sorted(agg_scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [DocumentHit(rank=i, doc_id=k, score=v)
            for i, (k, v) in enumerate(orden, 1)]


def agg_cupo(hits, top_n=3, strategy=AGG, por_agregado=2):
    """`por_agregado` cupos por el ranking agregado; el resto por mejor chunk.

    El documento elegido por mejor chunk no puede estar ya entre los primeros:
    de lo contrario el cupo se desperdiciaria y el cambio seria inerte.
    """
    base = _agg_original(hits, top_n=len(hits), strategy=strategy)
    elegidos = base[:por_agregado]
    ya = {d.doc_id for d in elegidos}
    mejor = {doc: max(s) for doc, s in _scores_por_doc(hits).items()}
    resto = sorted((d for d in mejor if d not in ya), key=lambda d: -mejor[d])
    for doc in resto[: top_n - por_agregado]:
        elegidos.append(DocumentHit(rank=len(elegidos) + 1, doc_id=doc,
                                    score=mejor[doc]))
        ya.add(doc)
    # Relleno por si el pool tuviera menos documentos distintos que cupos.
    for d in base[por_agregado:]:
        if len(elegidos) >= top_n:
            break
        if d.doc_id not in ya:
            elegidos.append(DocumentHit(rank=len(elegidos) + 1, doc_id=d.doc_id,
                                        score=d.score))
            ya.add(d.doc_id)
    return elegidos[:top_n]


def agg_mean_top5(hits, top_n=3, strategy=AGG):
    """Media de los 5 mejores: quita por completo el premio al conteo."""
    m = int(AGG[3:])
    return _ranked({doc: sum(sorted(s, reverse=True)[:m]) / min(len(s), m)
                    for doc, s in _scores_por_doc(hits).items()}, top_n)


def agg_top5_sqrt(hits, top_n=3, strategy=AGG):
    """Suma de los 5 mejores sobre la raiz del conteo efectivo: a medio camino
    entre `sum` (lineal en el conteo) y `mean` (invariante al conteo)."""
    m = int(AGG[3:])
    return _ranked({doc: sum(sorted(s, reverse=True)[:m]) / math.sqrt(min(len(s), m))
                    for doc, s in _scores_por_doc(hits).items()}, top_n)


AGREGADORES = {
    "entregada": _agg_original,
    "cupo-2+1": lambda h, top_n=3, strategy=AGG: agg_cupo(h, top_n, strategy, 2),
    "cupo-1+2": lambda h, top_n=3, strategy=AGG: agg_cupo(h, top_n, strategy, 1),
    "mean-top5": agg_mean_top5,
    "top5-sqrt": agg_top5_sqrt,
}


def construir_hits(consultas, cache_idx):
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for n, c in enumerate(consultas, 1):
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=PROF)[:PROF]
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        pool = sorted(hits, key=lambda h: -h.score)[:K_POOL]
        for i, h in enumerate(pool, 1):
            h.rank = i
        por_consulta[c["query_id"]] = pool
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta


def celda_resultados(hits_por_consulta, celda):
    generador.aggregate_documents = AGREGADORES[celda]
    try:
        return {qid: generador.build_result_object(qid, hits, agg_strategy=AGG)
                for qid, hits in hits_por_consulta.items()}
    finally:
        generador.aggregate_documents = _agg_original


def metricas(res, gt):
    out = [[], [], []]
    for g in gt:
        r = res.get(g["query_id"])
        if not r:
            continue
        rel = set(g["docs_relevantes"])
        out[0].append(f1([d["doc_id"] for d in r["documents"][:3]], rel)[2])
        out[1].append(ndcg(r["fragments"], rel))
        out[2].append(ndcg_penalizado(r["fragments"], rel))
    return out


def main() -> None:
    global DocumentHit
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "cupo_chunk_e20")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    DocumentHit = generador.DocumentHit

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_ag = [g for g in gt_todo if g.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas, "
          f"{len(gt_ag)} asistidas\n")

    cache_idx, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (
                faiss.read_index(str(encoder_dir(nombre) / "index.faiss")), metadata)
    print(f"  {cache_idx[MINILM][0].ntotal:,} vectores por indice\n", flush=True)

    hits_por_consulta = construir_hits(consultas, cache_idx)

    guardadas, por_agente = {}, {}
    for celda in CELDAS:
        res = celda_resultados(hits_por_consulta, celda)
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                            + metricas(res, gt_hum))
        por_agente[celda] = metricas(res, gt_ag)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':14s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):14s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)\n")

    base = guardadas[BASE]
    for k in CELDAS:
        if k == BASE:
            continue
        print(f"\n  === {k} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            gana = sum(1 for d in deltas if d > 1e-9)
            pierde = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{gana}g/{pierde}p  {'pasa' if lo > -0.02 else 'no pasa'}")
        # Veto pre-registrado: la ganancia no puede vivir en las 9 asistidas.
        d_ag = sum(por_agente[k][0]) / len(por_agente[k][0]) - \
            sum(por_agente[BASE][0]) / len(por_agente[BASE][0])
        d_hum = sum(guardadas[k][6]) / len(guardadas[k][6]) - \
            sum(base[6]) / len(base[6])
        print(f"  VETO: delta F1 en las 9 asistidas {d_ag:+.3f} contra "
              f"{d_hum:+.3f} en las 41 humanas"
              + ("  <- se activa" if d_ag > 0 and d_hum <= 0 else ""))


if __name__ == "__main__":
    main()
