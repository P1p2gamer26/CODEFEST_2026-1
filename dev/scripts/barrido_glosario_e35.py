#!/usr/bin/env python3
"""E35, FASE 2: mide UNA POR UNA las entradas candidatas de la fase 1.

Es el arnes de `barrido_glosario_e03.py` puesto al dia con la configuracion
ENTREGADA de hoy: peso de re-puntuacion 0.60 (no 0.25), post-filtrado por
fenomeno con umbral 0.8 y el texto de la consulta expandida pasado a
`build_result_object` (E23). Se llama a `generador.build_result_object` en vez
de re-implementar el orden de fragmentos: copiarlo ya costo una medicion
completa una vez (ver barrido_entidades_e30.py).

Como la expansion cambia QUE candidatos entran al pool, hace falta FAISS. Solo
se re-corren las consultas que la entrada toca: las demas devuelven un texto
identico y por tanto el mismo vector y el mismo resultado.

    .venv/Scripts/python.exe dev/scripts/barrido_glosario_e35.py
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
from src.retrieval import glosario as glos  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import (  # noqa: E402
    bootstrap_delta,
    cargar_jsonl,
    f1,
    ndcg,
    ndcg_penalizado,
)

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
PESO = generador.DEFAULT_RERANK_WEIGHT
RERANK_DEPTH = generador.DEFAULT_RERANK_DEPTH
K_POOL = generador.DEFAULT_K_POOL
ESTRATEGIA = "top5"

# (termino ES, expansion EN, chunks ES, chunks EN, nota). Los conteos salen de
# `auditoria_glosario_e35.py` sobre los 128.526 chunks del indice entregado.
# Las dos primeras PASAN el criterio de entrada; las dos ultimas se declaran
# SOSPECHOSAS antes de medir y estan aqui para dejar registro, no porque se
# espere adoptarlas.
CANDIDATOS = [
    ("capacidades espaciales", "space capabilities", 13, 2928,
     "pasa: q031 no trae sigla ni palabra inglesa"),
    ("operaciones espaciales", "space operations", 15, 1546,
     "pasa: q027 no trae puente (q030 ya expande por la entrada 'dinamicas')"),
    ("corea del norte", "North Korea DPRK", 6, 1188,
     "SOSPECHOSA: 'Corea'/'Korea' es cuasi-cognado, la consulta puede ya tener puente"),
    ("sistemas satelitales", "satellite systems", 2, 512,
     "SOSPECHOSA: 'satelital'/'satellite' es cognado, mismo fallo que 'laser'"),
]


def responder(consultas, cache_idx, expansion_extra=None):
    """Camino online entregado. `expansion_extra` es (termino, ingles)."""
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    resultados = {}
    for c in consultas:
        texto = glos.expandir_consulta(c["text"])
        if expansion_extra:
            termino, ingles = expansion_extra
            if termino in glos._normalizar(c["text"]):
                texto = f"{texto} {ingles}"
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)[:RERANK_DEPTH]
        for sec in (GTE, E5):
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                if h.fila >= 0:
                    h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
            hits.sort(key=lambda h: -h.score)
        hits = hits[:K_POOL]
        for i, h in enumerate(hits, 1):
            h.rank = i
        resultados[c["query_id"]] = generador.build_result_object(
            c["query_id"], hits, agg_strategy=ESTRATEGIA, texto_consulta=texto
        )
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "glosario_e35")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consultas = cargar_jsonl(CONSULTAS)
    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes, "
          f"{len(gt_hum)} humanas")

    cache_idx = {}
    metadata = None
    for nombre in (MINILM, GTE, E5):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores", flush=True)

    print(f"\nbase (glosario actual, peso={PESO}, pool={K_POOL}, agg={ESTRATEGIA})...",
          flush=True)
    base = responder(consultas, cache_idx)
    with (args.out_dir / "base.jsonl").open("w", encoding="utf-8") as f:
        for q in sorted(base):
            f.write(json.dumps(base[q], ensure_ascii=False) + "\n")

    def metricas(res, gt):
        out = [[], [], []]
        for g in gt:
            r = res[g["query_id"]]
            rel = set(g["docs_relevantes"])
            out[0].append(f1([d["doc_id"] for d in r["documents"][:3]], rel)[2])
            out[1].append(ndcg(r["fragments"], rel))
            out[2].append(ndcg_penalizado(r["fragments"], rel))
        return out

    COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
            "F1(hum)", "ND(hum)", "NDp(hum)")

    def celda(res):
        return metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)

    cel_base = celda(base)
    print(f"\n{'celda':30s}" + "".join(f"{c:>9s}" for c in COLS))
    print(f"{'(base, glosario actual) *':30s}"
          + "".join(f"{sum(v)/len(v):>9.3f}" for v in cel_base))
    print("  * tiene que dar 0.455/0.516/0.499 en las 50 y 0.433/0.474/0.467 "
          "en las independientes\n")

    celdas = {}
    for termino, ingles, c_es, c_en, nota in CANDIDATOS:
        afectadas = [c for c in consultas if termino in glos._normalizar(c["text"])]
        nuevos = responder(afectadas, cache_idx, (termino, ingles))
        res = dict(base)
        res.update(nuevos)
        with (args.out_dir / (termino.replace(" ", "_") + ".jsonl")).open(
            "w", encoding="utf-8"
        ) as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        celdas[termino] = (celda(res), [c["query_id"] for c in afectadas],
                           ingles, c_es, c_en, nota)
        print(f"{termino[:29]:30s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in celdas[termino][0]))

    orden_q = [g["query_id"] for g in gt_todo]
    print("\n" + "=" * 78)
    for termino, (cel, afect, ingles, c_es, c_en, nota) in celdas.items():
        print(f"\n=== {termino} -> {ingles}  (ES {c_es} / EN {c_en}; toca {afect}) ===")
        print(f"    {nota}")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(cel[j], cel_base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            g_ = sum(1 for d in deltas if d > 1e-9)
            p_ = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{g_}g/{p_}p  {'pasa' if lo > -0.02 else 'NO pasa'}")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, cel[0], cel_base[0])}
        print(f"  F1 gana  {[q for q in orden_q if d_f1[q] > 1e-9]}")
        print(f"  F1 pierde {[q for q in orden_q if d_f1[q] < -1e-9]}")


if __name__ == "__main__":
    main()
