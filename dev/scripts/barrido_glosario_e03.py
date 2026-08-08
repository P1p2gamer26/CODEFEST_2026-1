#!/usr/bin/env python3
"""E03: mide entradas NUEVAS del glosario ES->EN, UNA POR UNA.

`barrido_glosario.py` midio la tabla entera contra no tenerla, y ademas quedo
en el regimen viejo (k_pool=60, `sum`). Este script mide el efecto marginal de
cada entrada candidata sobre la tabla YA adoptada, en el regimen entregado
(k_pool=100, agg=top5, cascada gte+e5 peso 0.25).

POR QUE UNA POR UNA, fijado antes de medir: la tabla actual perdio una entrada
("maniobras de proximidad") que costaba F1 0.33 -> 0.00 en q021 y que en
bloque habria quedado escondida detras de las ganancias de las otras. Medir en
bloque no permite atribuir el efecto a la entrada.

CRITERIO DE ENTRADA, en dos partes y las dos obligatorias:
  1. La forma inglesa es al menos 10x mas frecuente en el corpus que la
     espanola (contado sobre los 128.526 chunks, sin mirar el ground truth).
  2. La CONSULTA no tiene ningun puente al vocabulario del corpus -- ni la
     sigla inglesa ni el termino en ingles ya escritos en ella. No alcanza con
     que el termino espanol sea raro: es exactamente lo que costo q021.

HALLAZGO DEL CONTEO que condiciona todo el experimento: la asimetria ES/EN
existe SOLO en los fenomenos 1 y 2 (IA y espacio, corpus en ingles). En el
fenomeno 3 (territorial, corpus en espanol) el conteo va al reves --
"restitucion de tierras" 617/22, "reclutamiento" 682/2, "control territorial"
285/41, "mineria ilegal" 148/1. Expandir esas consultas al ingles las alejaria
del vocabulario de sus documentos. Por eso ningun candidato de q033-q050 entra
salvo el que pasa el conteo, y ese se mide aparte y con sospecha.

REGLA DE DECISION: se adopta una entrada solo si el IC al 90% del delta
pareado excluye una perdida de 0,02 en las 50, y no pierde ninguna consulta.
Ante empate no entra: una entrada que no hace nada es superficie de riesgo
para consultas futuras.

Uso:
    python dev/scripts/barrido_glosario_e03.py
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
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, veredicto_bootstrap  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = [(GTE, 0.25), (E5, 0.25)]
RERANK_DEPTH = 200
K_POOL = 100
ESTRATEGIA = "top5"

# (termino ES, expansion EN, chunks ES, chunks EN, consultas que toca)
# Los conteos son los medidos sobre los 128.526 chunks del indice entregado.
CANDIDATOS = [
    ("sistemas no tripulados", "unmanned systems UAV", 0, 142, "q002"),
    ("amenazas ciberneticas", "cyber threats", 3, 127, "q013"),
    ("infraestructuras criticas", "critical infrastructure", 17, 433, "q013"),
    ("derecho internacional en el espacio", "international space law", 2, 424, "q017"),
    ("dominio espacial", "space domain", 23, 965, "q032"),
    # Sospechoso declarado ANTES de medir: "laser" es un cognado, o sea que la
    # consulta YA tiene puente al corpus ingles aunque el bigrama "capacidades
    # laser" no aparezca. Falla la parte 2 del criterio; se mide para dejarlo
    # registrado, no porque se espere adoptarlo.
    ("capacidades laser", "laser weapons", 0, 178, "q024"),
    # Sospechoso declarado ANTES de medir: unico candidato del fenomeno 3 que
    # pasa el conteo, pero sus documentos relevantes son colombianos y en
    # espanol. "critical minerals" es vocabulario de cadenas de suministro de
    # semiconductores (fenomeno 1): el riesgo es que arrastre la consulta al
    # fenomeno equivocado.
    ("minerales estrategicos", "critical minerals", 0, 74, "q046"),
]


def responder(consultas, cache_idx, expansion_extra=None):
    """Responde las consultas dadas. `expansion_extra` es (termino, ingles)."""
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
        for sec, peso in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += peso * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:K_POOL]
        for i, h in enumerate(hits, 1):
            h.rank = i
        resultados[c["query_id"]] = generador.build_result_object(
            c["query_id"], hits, agg_strategy=ESTRATEGIA
        )
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "glosario_e03")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consultas = cargar_jsonl(CONSULTAS)
    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    por_id = {c["query_id"]: c for c in consultas}

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
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores")

    print(f"\nbase (glosario actual, pool={K_POOL}, agg={ESTRATEGIA})...", flush=True)
    base = responder(consultas, cache_idx)
    with (args.out_dir / "base.jsonl").open("w", encoding="utf-8") as f:
        for q in sorted(base):
            f.write(json.dumps(base[q], ensure_ascii=False) + "\n")

    def evaluar(res, gt, fn):
        return [fn(res[g["query_id"]], g) for g in gt]

    F1 = lambda r, g: f1([d["doc_id"] for d in r["documents"][:3]], set(g["docs_relevantes"]))[2]
    ND = lambda r, g: ndcg(r["fragments"], set(g["docs_relevantes"]))

    print(f"\n{'entrada':38s} {'toca':6s} {'F1(50)':>8s} {'NDCG(50)':>9s} {'F1(ind)':>8s} {'NDCG(ind)':>10s}")
    b50f, b50n = evaluar(base, gt_todo, F1), evaluar(base, gt_todo, ND)
    bidf, bidn = evaluar(base, gt_indep, F1), evaluar(base, gt_indep, ND)
    print(f"{'(base, glosario actual)':38s} {'-':6s} {sum(b50f)/len(b50f):8.3f} "
          f"{sum(b50n)/len(b50n):9.3f} {sum(bidf)/len(bidf):8.3f} {sum(bidn)/len(bidn):10.3f}")

    veredictos = {}
    for termino, ingles, c_es, c_en, qids in CANDIDATOS:
        # Solo se re-corren las consultas que la entrada toca: las demas
        # devuelven un texto identico y por tanto el mismo vector.
        afectadas = [c for c in consultas if termino in glos._normalizar(c["text"])]
        nuevos = responder(afectadas, cache_idx, (termino, ingles))
        res = dict(base)
        res.update(nuevos)
        with (args.out_dir / (termino.replace(" ", "_") + ".jsonl")).open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        a50f, a50n = evaluar(res, gt_todo, F1), evaluar(res, gt_todo, ND)
        aidf, aidn = evaluar(res, gt_indep, F1), evaluar(res, gt_indep, ND)
        print(f"{termino[:37]:38s} {len(afectadas):<6d} {sum(a50f)/len(a50f):8.3f} "
              f"{sum(a50n)/len(a50n):9.3f} {sum(aidf)/len(aidf):8.3f} {sum(aidn)/len(aidn):10.3f}")
        veredictos[termino] = (ingles, c_es, c_en, [c["query_id"] for c in afectadas],
                               (a50f, b50f), (a50n, b50n), (aidf, bidf), (aidn, bidn))

    print("\n" + "=" * 78)
    for termino, (ingles, c_es, c_en, afect, *series) in veredictos.items():
        print(f"\n=== {termino} -> {ingles}  (ES {c_es} / EN {c_en} chunks; toca {afect}) ===")
        for nombre, (va, vb) in zip(("F1@3 50", "NDCG@10 50", "F1@3 indep", "NDCG@10 indep"), series):
            media, bajo, alto = bootstrap_delta([x - y for x, y in zip(va, vb)])
            gana = sum(1 for x, y in zip(va, vb) if x > y + 1e-9)
            pierde = sum(1 for x, y in zip(va, vb) if x < y - 1e-9)
            print(f"  {nombre:14s} delta {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  "
                  f"gana {gana}, pierde {pierde}, empata {len(va)-gana-pierde}")


if __name__ == "__main__":
    main()
