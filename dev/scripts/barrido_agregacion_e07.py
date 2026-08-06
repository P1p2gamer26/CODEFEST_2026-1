#!/usr/bin/env python3
"""E07: re-abrir la agregacion a documento bajo el regimen ACTUAL.

HALLAZGO QUE LO MOTIVA (scripts/diagnostico_ceros.py, 6 ago 2026, sobre la
entrega en 0.440): de las 20 consultas que fallan, **16 tienen el documento
correcto DENTRO del pool y ninguna lo tiene ausente del indice**. q022 y q040
tienen un chunk relevante en **rank 1** y aun asi el documento no entra al
top-3; q044 en rank 3, q039 en rank 4, q034 en rank 5. El cuello de botella
dejo de ser la recuperacion y paso a ser la agregacion.

HIPOTESIS, escrita antes de medir: `top5` premia al documento que aporta
MUCHOS chunks al pool por encima del que aporta UNO excelente. Un documento
con 5 fragmentos mediocres suma cinco terminos; uno con un unico fragmento en
rank 1 suma uno solo. Es el mismo modo de fallo de `sum` sin tope (que ya se
midio: con pool 200 se derrumba a 0.298) nada mas que con el tope en 5. Si es
cierto, **un M mas chico deberia ganar**, y el efecto tiene que verse
concentrado en las consultas cuyo documento relevante esta arriba del pool
pero aporta pocos chunks.

POR QUE SE RE-ABRE ALGO YA DESCARTADO: `topM` se midio en su momento y no se
adopto, pero fue con **k_pool=60 y peso 0.25**. Con pool 60 casi ningun
documento aporta mas de 5 chunks, asi que `top5` y `sum` eran **la misma
operacion** y el barrido no podia distinguir nada. Hoy el pool es 100 y el
peso 0.60: el regimen es otro. Es exactamente el argumento que hizo que E01
valiera la pena, y la leccion 5 (las notas propias envejecen).

RIESGO DECLARADO: barrer M contra 50 consultas es la maquina de sobreajuste
de la leccion 2. Mitigaciones fijadas ANTES de medir:
  - Se adopta solo si el IC al 90% del delta pareado excluye una perdida de
    0,02 en las DOS muestras (50 y 10 independientes), y ante empate se
    conserva `top5`.
  - La hipotesis predice una DIRECCION (M mas chico gana). Si el ganador
    fuera `sum` o un M mayor, eso refuta la explicacion mecanica y el
    resultado se trata como ruido, no como hallazgo.

Coste: cero codificacion nueva y cero logica nueva -- `aggregate.py` ya trae
max / mean / sum / topM.

Uso:
    python dev/scripts/barrido_agregacion_e07.py
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
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

# La configuracion ENTREGADA, fija en este barrido: solo la agregacion se mueve.
RERANK_DEPTH = 200
PESO = 0.60
K_POOL = 100

# Ordenadas de mas concentrada a mas dispersa, que es la direccion que la
# hipotesis predice. `mean` va al final porque no es un punto de esa recta:
# divide por el numero de chunks en vez de truncarlo, asi que castiga al
# documento con muchos chunks en vez de solo dejar de premiarlo.
ESTRATEGIAS = ("max", "top2", "top3", "top5", "top8", "sum", "mean")
BASE = "top5"  # la entregada


def construir_pools(consultas, cache_idx):
    """Pool y similitudes crudas, una sola vez: la agregacion no toca FAISS."""
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for c in consultas:
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)[:RERANK_DEPTH]
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        recorte = hits[:K_POOL]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        por_consulta[c["query_id"]] = recorte
    return por_consulta


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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "agg_e07")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    gt_hum = [f for f in gt_todo if not f.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes\n")

    cache_idx = {}
    metadata = None
    for nombre in (MINILM, *SECUNDARIOS):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores", flush=True)

    print(f"\nrecuperando (peso={PESO}, pool={K_POOL}, glosario on)...", flush=True)
    pools = construir_pools(consultas, cache_idx)

    guardadas = {}
    for est in ESTRATEGIAS:
        res = {
            qid: generador.build_result_object(qid, hits, agg_strategy=est)
            for qid, hits in pools.items()
        }
        with (args.out_dir / f"{est}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        f_a, n_a = metricas(res, gt_todo)
        f_i, n_i = metricas(res, gt_indep)
        f_h, n_h = metricas(res, gt_hum)
        guardadas[est] = (f_a, n_a, f_i, n_i, f_h, n_h)

    print(f"\n{'estrategia':12s} {'F1(50)':>8s} {'NDCG(50)':>9s} {'F1(ind)':>8s} {'NDCG(ind)':>10s} {'F1(hum)':>8s} {'NDCG(hum)':>10s}")
    for est in ESTRATEGIAS:
        g = guardadas[est]
        etiqueta = est + (" *" if est == BASE else "")
        print(
            f"{etiqueta:12s} " + " ".join(
                f"{sum(v)/len(v):>{w}.3f}" for v, w in zip(g, (8, 9, 8, 10, 8, 10))
            )
        )
    print(f"\n  * = la entregada")

    print(f"\ndeltas pareados contra '{BASE}', IC al 90%:")
    base = guardadas[BASE]
    for est in ESTRATEGIAS:
        if est == BASE:
            continue
        print(f"\n  === {est} ===")
        for nombre, i in (
            ("F1@3    50   ", 0),
            ("NDCG@10 50   ", 1),
            ("F1@3    indep", 2),
            ("NDCG@10 indep", 3),
            ("F1@3    human", 4),
            ("NDCG@10 human", 5),
        ):
            deltas = [x - y for x, y in zip(guardadas[est][i], base[i])]
            media, bajo, alto = bootstrap_delta(deltas)
            gana = sum(1 for d in deltas if d > 1e-9)
            pierde = sum(1 for d in deltas if d < -1e-9)
            ok = "pasa" if bajo > -0.02 else "no pasa"
            print(f"  {nombre}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {gana}g/{pierde}p  {ok}")


if __name__ == "__main__":
    main()
