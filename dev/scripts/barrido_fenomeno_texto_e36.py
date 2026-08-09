#!/usr/bin/env python3
"""E36: el fenomeno decidido por el TEXTO de la consulta contra el voto del pool.

FASE 1 (puerta de entrada): en cuantas de las 50 discrepan, y quien acierta.
FASE 2: dos variantes discretas, pre-registradas, sobre la base ENTREGADA
        (que ya lleva E32, `filtrar_por_fenomeno_dominante` con umbral 0.8):

    voto  -> el clasificador decide el fenomeno; el filtro se aplica siempre.
    veto  -> si el clasificador discrepa del pool, NO se filtra.

Sin FAISS: reusa `pools_entregados.json` y el arnes de E32/E33. El orden de
fragmentos lo hace `ordenar_para_fragmentos`, no una copia.

    .venv/Scripts/python.exe dev/scripts/barrido_fenomeno_texto_e36.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from clasificador_fenomeno_e36 import cargar_tabla, clasificar  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
UMBRAL = 0.8  # el de E32, adoptado

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")


def voto_del_pool(hits):
    """El voto ponderado por score de E32: (fenomeno, dominancia)."""
    peso = defaultdict(float)
    for h in hits:
        if h.fenomeno is not None:
            peso[h.fenomeno] += max(h.score, 0.0)
    if not peso:
        return None, 0.0
    total = sum(peso.values())
    ganador, mejor = max(peso.items(), key=lambda kv: kv[1])
    return ganador, (mejor / total if total > 0 else 0.0)


def resultado(qid, hits, texto_expandido, fenomeno=None):
    """Camino online entregado. `fenomeno` None = no filtrar."""
    if fenomeno is not None:
        hits = [h for h in hits if h.fenomeno == fenomeno] or hits
    doc_hits = aggregate_documents(hits, top_n=3, strategy="top5")
    top_ids = [d.doc_id for d in doc_hits]
    toks = frozenset(tokens_de(texto_expandido))
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
                    default=DEV_DIR / "intermedios" / "fenomeno_texto_e36")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    expandidas = {q: expandir_consulta(t) for q, t in consultas.items()}

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes, {len(gt_hum)} humanas\n")

    tabla = cargar_tabla()
    texto_fen = {q: clasificar(t, tabla)[0] for q, t in consultas.items()}
    pool_fen = {q: voto_del_pool(h) for q, h in hits_por_q.items()}

    # fenomeno de cada doc_id, para saber quien tiene razon
    fen_de_doc = {c["doc_id"]: c["fenomeno"] for p in pools.values() for c in p}
    rel_fen = {}
    for g in gt_todo:
        rel_fen[g["query_id"]] = {fen_de_doc[d] for d in g["docs_relevantes"]
                                  if d in fen_de_doc}

    print("=== FASE 1: PUERTA DE ENTRADA ===")
    print("  el clasificador de texto se construyo SOLO con frecuencias del corpus")
    print("  (dev/scripts/clasificador_fenomeno_e36.py); el ground truth no se miro.\n")
    disc = [q for q in sorted(hits_por_q) if texto_fen[q] != pool_fen[q][0]]
    print(f"  discrepan en {len(disc)}/50: {disc}")
    print(f"\n  {'qid':6s}{'texto':>6s}{'pool':>6s}{'dom':>7s}  relevantes en   quien acierta")
    tv = pv = amb = 0
    for q in disc:
        rf = rel_fen.get(q)
        if rf is None:
            quien = "sin GT"
        elif len(rf) > 1:
            quien = "ambos/ninguno (relevantes en 2 fenomenos)"
            amb += 1
        elif texto_fen[q] in rf:
            quien = "TEXTO"
            tv += 1
        elif pool_fen[q][0] in rf:
            quien = "pool"
            pv += 1
        else:
            quien = "ninguno"
        print(f"  {q:6s}{texto_fen[q]:>6d}{pool_fen[q][0]:>6d}{pool_fen[q][1]:>7.2f}"
              f"  {sorted(rf) if rf else '-'}   {quien}")
    print(f"\n  cuando discrepan: texto acierta {tv}, pool acierta {pv}, "
          f"ambiguas {amb} (q019 y similares no cuentan)")
    if len(disc) < 5:
        print("  !! menos de 5 discrepancias: NO CONCLUYENTE POR CONSTRUCCION")
    if tv < pv:
        print("  !! el texto acierta MENOS que el pool: experimento muerto, no se sigue")
        return

    # ---- FASE 2 ----
    celdas, archivos = {}, {}

    def registrar(nombre, fen_por_q):
        res = {q: resultado(q, h, expandidas[q], fenomeno=fen_por_q(q, h))
               for q, h in hits_por_q.items()}
        archivos[nombre] = res
        with (args.out_dir / f"{nombre}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        celdas[nombre] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                          + metricas(res, gt_hum))
        malos = [q for q, r in res.items()
                 if len(r["documents"]) != 3 or len(r["fragments"]) != 10]
        print(f"  {nombre}: {'OK 3 docs / 10 frags en las 50' if not malos else '!! ROMPE sec 9.2 -> ' + str(malos)}")

    def base(q, h):
        """E32 tal cual: filtra por el voto del pool si supera el umbral."""
        fen, dom = pool_fen[q]
        return fen if dom >= UMBRAL else None

    def voto(q, h):
        """El clasificador de texto decide; se filtra siempre."""
        return texto_fen[q]

    def veto(q, h):
        """E32, pero si el texto discrepa del pool no se filtra."""
        return None if q in set(disc) else base(q, h)

    print("\n=== FASE 2 ===")
    registrar("entregada", base)
    registrar("e36-voto", voto)
    registrar("e36-veto", veto)

    print(f"\n{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in celdas:
        print(f"{k + (' *' if k == 'entregada' else ''):16s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in celdas[k]))
    print("\n  * = la entregada; tiene que dar 0.455/0.516/0.499 y 0.433/0.474/0.467\n")

    b = celdas["entregada"]
    orden_q = [g["query_id"] for g in gt_todo]
    for k in celdas:
        if k == "entregada":
            continue
        docs = sum(1 for q in archivos[k]
                   for a, c in zip(archivos[k][q]["documents"],
                                   archivos["entregada"][q]["documents"])
                   if a["doc_id"] != c["doc_id"])
        print(f"  === {k} ===  {docs}/150 documentos cambian")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(celdas[k][j], b[j])]
            media, lo, hi = bootstrap_delta(deltas)
            g_ = sum(1 for d in deltas if d > 1e-9)
            p_ = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g_}g/{p_}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in celdas[k][0] if v == 0)
        print(f"  VETO consultas con F1@3=0: {ceros} (entregada: {sum(1 for v in b[0] if v == 0)})")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, celdas[k][0], b[0])}
        print(f"  F1 gana {[q for q in orden_q if d_f1[q] > 1e-9]}")
        print(f"  F1 pierde {[q for q in orden_q if d_f1[q] < -1e-9]}")
        print(f"  q019 {d_f1.get('q019', 0):+.3f}   q027 {d_f1.get('q027', 0):+.3f}\n")


if __name__ == "__main__":
    main()
