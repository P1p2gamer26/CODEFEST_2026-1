#!/usr/bin/env python3
"""E32 (filtrar el pool por fenomeno dominante) y E33 (la M de topM).

Los dos son los unicos experimentos que el proyecto APLAZO POR ESCRITO hasta
tener el ground truth en 50 consultas. Ya estan las 50. Ninguno es una idea
nueva: son deudas, y este barrido existe para cerrarlas.

Sin FAISS: lee `dev/intermedios/pools_entregados.json` (volcar_pools.py) y
reusa el arnes de E22/E23/E30. Los criterios de orden de fragmentos NO se
copian: se llama a `ordenar_para_fragmentos` de src/retrieval/truncate.py
(el error contrario costo una medicion completa, ver barrido_entidades_e30.py).

    .venv/Scripts/python.exe dev/scripts/barrido_fenomeno_topm_e32_e33.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.aggregate import (  # noqa: E402
    aggregate_documents,
    filtrar_por_fenomeno_dominante,
)
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

BASE = "entregada"
# E32: umbrales de abstencion, los mismos de la medicion original con 41.
UMBRALES = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# E33: pocos valores discretos, pre-registrados. NO se toma el argmax.
EMES = (3, 5, 8, 12)

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")


def resultado(qid, hits, texto_consulta, agg="top5", umbral=None):
    """Camino online aplanado con los defaults entregados (cupo_alineado=10)."""
    if umbral is not None:
        hits = filtrar_por_fenomeno_dominante(hits, umbral=umbral)
    doc_hits = aggregate_documents(hits, top_n=3, strategy=agg)
    top_ids = [d.doc_id for d in doc_hits]
    toks = frozenset(tokens_de(texto_consulta))
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks)
    )
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [{"rank": f["rank"], "chunk_id": f["chunk_id"],
                       "doc_id": f["doc_id"], "text": f["text"]} for f in frags],
    }


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "fenomeno_topm_e32_e33")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}
    expandidas = {c["query_id"]: expandir_consulta(c["text"])
                  for c in cargar_jsonl(CONSULTAS)}

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes, {len(gt_hum)} humanas\n")

    # ---- PUERTA DE ENTRADA de E32: cuanto se gasta hoy en el fenomeno malo ----
    gtm = {g["query_id"]: set(g["docs_relevantes"]) for g in gt_todo}
    fen_de_doc = {}
    for p in pools.values():
        for c in p:
            fen_de_doc[c["doc_id"]] = c["fenomeno"]
    un_solo = sum(1 for g in gt_todo
                  if len({fen_de_doc.get(d) for d in gtm[g["query_id"]] if d in fen_de_doc}) <= 1)
    print("=== PUERTA DE ENTRADA (E32) ===")
    print(f"  consultas con todos sus relevantes en UN fenomeno: {un_solo}/{len(gt_todo)}")

    celdas, archivos = {}, {}

    def registrar(nombre, agg="top5", umbral=None):
        res = {q: resultado(q, h, expandidas[q], agg=agg, umbral=umbral)
               for q, h in hits_por_q.items()}
        archivos[nombre] = res
        with (args.out_dir / f"{nombre}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        celdas[nombre] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                          + metricas(res, gt_hum))
        # sec. 9.2: exactamente 3 documentos y 10 fragmentos, siempre
        malos = [q for q, r in res.items()
                 if len(r["documents"]) != 3 or len(r["fragments"]) != 10]
        if malos:
            print(f"  !! {nombre}: {len(malos)} consultas sin 3 docs / 10 frags -> {malos}")
        return res

    registrar(BASE)
    for u in UMBRALES:
        registrar(f"e32-u{u:.1f}", umbral=u)
    for m in EMES:
        if m != 5:
            registrar(f"e33-top{m}", agg=f"top{m}")

    # cupos gastados en el fenomeno equivocado, sobre la fila base
    gastados = tot = 0
    for g in gt_todo:
        q = g["query_id"]
        fens_rel = {fen_de_doc.get(d) for d in gtm[q] if d in fen_de_doc}
        for d in archivos[BASE][q]["documents"][:3]:
            tot += 1
            if fens_rel and fen_de_doc.get(d["doc_id"]) not in fens_rel:
                gastados += 1
    print(f"  cupos en el fenomeno equivocado (fila base): {gastados}/{tot} "
          f"({100*gastados/max(tot,1):.0f}%)")
    # cuantas consultas cambian de pool con cada umbral
    for u in UMBRALES:
        n = sum(1 for q, h in hits_por_q.items()
                if len(filtrar_por_fenomeno_dominante(h, umbral=u)) != len(h))
        print(f"  umbral {u:.1f}: el filtro actua en {n}/50 consultas")
    print()

    print(f"{'celda':22s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in celdas:
        print(f"{k + (' *' if k == BASE else ''):22s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in celdas[k]))
    print("\n  * = la entregada; tiene que dar 0.440/0.506/0.491 y 0.400/0.436/0.429 (E09)\n")

    base = celdas[BASE]
    orden_q = [g["query_id"] for g in gt_todo]
    for k in celdas:
        if k == BASE:
            continue
        docs = sum(1 for q in archivos[k]
                   for a, b in zip(archivos[k][q]["documents"], archivos[BASE][q]["documents"])
                   if a["doc_id"] != b["doc_id"])
        print(f"  === {k} ===  {docs}/150 documentos cambian")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(celdas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            g_ = sum(1 for d in deltas if d > 1e-9)
            p_ = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g_}g/{p_}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in celdas[k][0] if v == 0)
        print(f"  VETO consultas con F1@3=0: {ceros} (entregada: {sum(1 for v in base[0] if v == 0)})")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, celdas[k][0], base[0])}
        gana = [q for q in orden_q if d_f1[q] > 1e-9]
        pierde = [q for q in orden_q if d_f1[q] < -1e-9]
        print(f"  F1 gana {len(gana)}: {gana}")
        print(f"  F1 pierde {len(pierde)}: {pierde}")
        print(f"  q019 {d_f1.get('q019', 0):+.3f}   q027 {d_f1.get('q027', 0):+.3f}\n")


if __name__ == "__main__":
    main()
