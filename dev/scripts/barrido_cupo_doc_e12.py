#!/usr/bin/env python3
"""E12: el reparto de los 10 fragmentos ENTRE los 3 documentos del top-3.

HIPOTESIS, escrita antes de medir: repartir los 10 fragmentos entre los 3
documentos del top-3 con un cupo maximo por documento mejora el NDCG@10, porque
hoy nada impide que un solo documento se lleve 8 de los 10 cupos y, si ese
documento es el equivocado, la respuesta entrega ocho ceros.

JUSTIFICACION MECANICA. `ordenar_para_fragmentos` ordena por (top-3, idioma,
aparato) y dentro de cada bloque por score, **sin ninguna nocion de a cual de
los tres documentos pertenece cada fragmento**. La alineacion ya concentra el
98% de los fragmentos en el top-3, asi que el reparto INTERNO de esos 10 cupos
entre 3 documentos es la variable que quedo sin tocar cuando se adopto. Y la
diferencia de score entre el documento 1 y el 3 es chica comparada con la de
estar o no estar en el top-3: con F1@3 0.440 el caso tipico es acertar 1 o 2 de
3, o sea que cubrir los tres es cubrir donde esta la respuesta.

NO es el barrido de `--cupo-alineado` ya refutado (monotono, IC bajo cero en las
seis celdas): ese reservaba cupos FUERA del top-3 y los gastaba en documentos
que la respuesta declara NO relevantes. Este redistribuye DENTRO del top-3 y no
toca un solo documento del ranking.

RIESGOS Y MITIGACIONES, fijados ANTES de medir:
  - `F1@3` tiene que salir **+0.000 en las tres muestras en todas las celdas**:
    solo se reordenan fragmentos, `aggregate_documents` no se toca. Si el F1 se
    mueve, el arnes esta mal y el barrido **no se lee**.
  - Metrica de decision: `NDCG@10`, mirando tambien `NDp`. Nuestro NDCG hereda
    la relevancia del DOCUMENTO, o sea que es la que mejor puede ver este
    cambio y tambien la que puede premiarlo por construccion: **si NDCG sube y
    NDp no lo sigue, sospechar**.
  - Un cupo estricto puede forzar la entrada de un fragmento malo del documento
    3 desplazando uno bueno del 1. Es el intercambio que se mide, y por eso la
    grilla incluye cupos flojos (6) y no solo el reparto perfecto.
  - Se CUENTAN los fragmentos ilegibles aparte (como en E11): el cupo puede
    sacar uno legible y meter uno en coreano. Si suben, se rechaza aunque el
    NDCG mejore (leccion 7).
  - Ante empate se conserva la entregada.
  - Regla de E09: la fila base tiene que reproducir 0.440 / 0.490 / 0.476.

COSTE: una sola pasada de FAISS; las celdas son reordenamientos del mismo pool.

Uso:
    python dev/scripts/barrido_cupo_doc_e12.py
"""

import argparse
import json
import sys
from collections import Counter
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

# Cada celda es el cupo maximo de fragmentos por documento antes de dar paso a
# la siguiente vuelta. `None` = la entregada (sin cupo). 1 = round-robin puro.
CELDAS = {"entregada": None, "cupo6": 6, "cupo5": 5, "cupo4": 4, "round-robin": 1}
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

_ordenar_original = generador.ordenar_para_fragmentos


def construir_pools(consultas, cache_idx):
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
        hits.sort(key=lambda h: -h.score)
        recorte = hits[:K_POOL]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        por_consulta[c["query_id"]] = recorte
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta


def parche_cupo(cupo: int):
    """`ordenar_para_fragmentos` con cupo por documento, sobre el orden original.

    Se reusa el orden entregado y solo se re-agrupa: al i-esimo fragmento de un
    documento le toca la vuelta `i // cupo`, y se ordena por (fuera del top-3,
    vuelta, posicion original). Asi el cupo NO reemplaza los criterios de
    idioma y aparato -- quedan dentro de la posicion original, o sea que siguen
    decidiendo dentro de cada vuelta.
    """

    def ordenar(hits, doc_ids_prioritarios=None, priorizar_idioma=True, **kw):
        base = _ordenar_original(
            hits,
            doc_ids_prioritarios=doc_ids_prioritarios,
            priorizar_idioma=priorizar_idioma,
            **kw,
        )
        top = set(doc_ids_prioritarios or ())
        vistos: Counter = Counter()
        claves = []
        for pos, h in enumerate(base):
            vistos[h.doc_id] += 1
            vuelta = (vistos[h.doc_id] - 1) // cupo
            claves.append(((1 if (top and h.doc_id not in top) else 0), vuelta, pos, h))
        # El cupo solo reparte DENTRO del top-3: fuera de el los documentos
        # aportan uno o dos hits y agrupar por vuelta no cambia nada.
        claves.sort(key=lambda t: t[:3])
        return [t[3] for t in claves]

    return ordenar


def celda_resultados(pools, celda):
    cupo = CELDAS[celda]
    generador.ordenar_para_fragmentos = (
        _ordenar_original if cupo is None else parche_cupo(cupo)
    )
    try:
        return {
            qid: generador.build_result_object(qid, hits, agg_strategy=AGG)
            for qid, hits in pools.items()
        }
    finally:
        generador.ordenar_para_fragmentos = _ordenar_original


def contar_ilegibles(res, pools) -> int:
    """La cifra que el NDCG no ve y que puede vetar el cambio (riesgo nº 4)."""
    idioma = {h.chunk_id: h.idioma for hits in pools.values() for h in hits}
    return sum(
        1
        for r in res.values()
        for fr in r["fragments"]
        if idioma.get(fr["chunk_id"], "es") not in generador.IDIOMAS_LEGIBLES
    )


def reparto(res) -> str:
    """Cuantos documentos distintos aportan los 10 fragmentos, y el maximo."""
    docs = [Counter(fr["doc_id"] for fr in r["fragments"]) for r in res.values()]
    distintos = sum(len(c) for c in docs) / len(docs)
    peor = max(max(c.values()) for c in docs)
    mediana_max = sorted(max(c.values()) for c in docs)[len(docs) // 2]
    return f"{distintos:.2f} docs/consulta, max {peor}, mediana del max {mediana_max}"


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "cupo_e12")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas\n")

    cache_idx, metadata = {}, None
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

    print(f"\nrecuperando {PROF} y re-puntuando (peso={PESO})...", flush=True)
    pools = construir_pools(consultas, cache_idx)

    guardadas, ilegibles, repartos = {}, {}, {}
    for celda in CELDAS:
        res = celda_resultados(pools, celda)
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        ilegibles[celda] = contar_ilegibles(res, pools)
        repartos[celda] = reparto(res)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':14s}" + "".join(f"{c:>9s}" for c in COLS) + f"{'ilegibles':>11s}")
    for k in guardadas:
        print(f"{k + (' *' if k == BASE else ''):14s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k])
              + f"{ilegibles[k]:>11d}")
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)")
    print("  F1@3 tiene que ser identico en TODAS las celdas: los documentos no se tocan.\n")

    print("reparto de los 10 fragmentos entre documentos:")
    for k, v in repartos.items():
        print(f"  {k:14s} {v}")

    print("\ndeltas pareados contra la entregada, IC al 90% (criterio: bajo > -0.02).")
    print("LA METRICA DE DECISION ES NDCG@10 (y NDp la acompana), fijada antes de medir:")
    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        print(f"\n  === {k} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, bajo, alto = bootstrap_delta(deltas)
            g = sum(1 for d in deltas if d > 1e-9)
            p = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  "
                  f"{g}g/{p}p  {'pasa' if bajo > -0.02 else 'no pasa'}")


if __name__ == "__main__":
    main()
