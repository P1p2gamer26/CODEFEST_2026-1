#!/usr/bin/env python3
"""E37: re-calibrar `k_pool` y la M de topM DESPUES de que E32 cambiara el regimen.

E32 (adoptado hace una hora) descarta del pool los chunks del fenomeno no
dominante, asi que las consultas donde el filtro actua agregan sobre un pool
EFECTIVO menor que 100. Pero `k_pool=100` se fijo cuando el filtro no existia y
E33 midio topM sobre la base 0.440/0.506, o sea sobre el sistema ANTERIOR a E32.
Es el mismo argumento con el que se adopto E01 (peso 0.25 -> 0.60): el parametro
se fijo bajo un regimen que ya no existe. Sin ese cambio de regimen, re-barrer
estos dos parametros seria la maquina de sobreajuste de la leccion 2.

Sin FAISS: reusa `pools_entregados.json --con-similitudes` (200 candidatos
crudos con el coseno de cada encoder) y el cache `meta_crudos.json`, igual que
barrido_cascada_e27_e28_e29.py. `filtrar_por_fenomeno_dominante` y
`ordenar_para_fragmentos` se LLAMAN, no se reimplementan.

    .venv/Scripts/python.exe dev/scripts/barrido_pool_topm_e37.py
"""

import argparse
import json
import sys
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

from barrido_cascada_e27_e28_e29 import GTE, E5, MINILM, cargar, hits_de  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

UMBRAL = 0.8            # E32, adoptado
PESOS = {GTE: 0.60, E5: 0.60}
K_POOLS = (100, 150, 200)
EMES = (5, 8)
BASE = "k100:top5 *"    # lo entregado
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")


def pool_k(crudos_q, sims_q, k):
    """Los k mejores por el score entregado, con un solo recorte (como generador.py).

    NO se reusa `pool_variante` de E27-E29: esa funcion recorta a su propio
    K_POOL=100 por dentro, asi que k=150/200 habrian salido silenciosamente
    iguales a 100. Se replica su misma secuencia de ordenaciones para que la
    fila base coincida hasta en los empates.
    """
    vivos = [(i, s) for i, s in enumerate(sims_q[MINILM])]
    for enc in (GTE, E5):
        vivos = [(i, s + PESOS[enc] * sims_q[enc][i]) for i, s in vivos]
        vivos.sort(key=lambda t: -t[1])
    return vivos[:k]


def resultado(qid, hits, texto_expandido, agg):
    hits = filtrar_por_fenomeno_dominante(hits, umbral=UMBRAL)
    doc_hits = aggregate_documents(hits, top_n=3, strategy=agg)
    top_ids = [d.doc_id for d in doc_hits]
    toks = frozenset(tokens_de(texto_expandido))
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks)
    )
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [{"rank": f["rank"], "chunk_id": f["chunk_id"], "doc_id": f["doc_id"],
                       "text": f["text"]} for f in frags],
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "pool_topm_e37")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, crudos, sims, meta = cargar()
    print("config del volcado:", config)
    assert config["peso"] == 0.60 and config["profundidad"] == 200 and config["k_pool"] == 100

    consultas = {c["query_id"]: expandir_consulta(c["text"]) for c in cargar_jsonl(CONSULTAS)}
    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_ind = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_ind)} independientes, {len(gt_hum)} humanas\n")

    # ---- FASE 1: cuanto encoge el pool. Si encoge poco, el experimento muere aca.
    print("=== FASE 1: tamano del pool antes y despues del filtro E32 (umbral 0.8) ===")
    encogen = {}
    for k in K_POOLS:
        filas = []
        for q in sorted(crudos):
            h = hits_de(pool_k(crudos[q], sims[q], k), crudos[q], meta)
            filas.append((q, len(h), len(filtrar_por_fenomeno_dominante(h, umbral=UMBRAL))))
        encogen[k] = filas
        actua = [f for f in filas if f[2] != f[1]]
        efectivos = [f[2] for f in actua]
        print(f"  k_pool={k}: el filtro actua en {len(actua)}/50 consultas; "
              f"pool efectivo ahi: min {min(efectivos)} mediana "
              f"{sorted(efectivos)[len(efectivos)//2]} max {max(efectivos)}; "
              f"media sobre las 50 = {sum(f[2] for f in filas)/len(filas):.1f}")
    print("\n  detalle a k_pool=100 (solo donde el filtro actua):")
    for q, antes, desp in encogen[100]:
        if desp != antes:
            print(f"    {q}: {antes} -> {desp}  ({100*desp/antes:.0f}%)")
    print()

    # ---- FASE 2: nueve lecturas por celda, 6 celdas
    celdas, archivos = {}, {}
    for k in K_POOLS:
        for m in EMES:
            nombre = f"k{k}:top{m}" + (" *" if (k, m) == (100, 5) else "")
            res = {q: resultado(q, hits_de(pool_k(crudos[q], sims[q], k), crudos[q], meta),
                                consultas[q], f"top{m}")
                   for q in crudos}
            archivos[nombre] = res
            celdas[nombre] = (metricas(res, gt_todo) + metricas(res, gt_ind)
                              + metricas(res, gt_hum))
            malos = [q for q, r in res.items()
                     if len(r["documents"]) != 3 or len(r["fragments"]) != 10]
            if malos:
                print(f"  !! {nombre}: SEC 9.2 ROTA en {malos}")
            seguro = nombre.replace(":", "_").replace(" *", "")
            with (args.out_dir / f"{seguro}.jsonl").open("w", encoding="utf-8") as f:
                for q in sorted(res):
                    f.write(json.dumps(res[q], ensure_ascii=False) + "\n")

    print(f"{'celda':14s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in celdas:
        print(f"{k:14s}" + "".join(f"{sum(v)/len(v):>9.3f}" for v in celdas[k]))
    print("\n  * = la entregada; tiene que dar 0.455/0.516/0.499 y 0.433/0.474/0.467\n")

    base = celdas[BASE]
    ceros_base = sum(1 for v in base[0] if v == 0)
    orden_q = [g["query_id"] for g in gt_todo]
    for k in celdas:
        if k == BASE:
            continue
        lineas = sum(1 for q in archivos[k] if archivos[k][q] != archivos[BASE][q])
        docs = sum(1 for q in archivos[k]
                   for a, b in zip(archivos[k][q]["documents"], archivos[BASE][q]["documents"])
                   if a["doc_id"] != b["doc_id"])
        print(f"  === {k} ===  {lineas}/50 lineas, {docs}/150 documentos cambian")
        for j, nombre in enumerate(COLS):
            d = [x - y for x, y in zip(celdas[k][j], base[j])]
            med, lo, hi = bootstrap_delta(d)
            g = sum(1 for x in d if x > 1e-9)
            p = sum(1 for x in d if x < -1e-9)
            print(f"  {nombre:9s}: {med:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g}g/{p}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in celdas[k][0] if v == 0)
        print(f"  VETO consultas con F1@3=0: {ceros} (entregada: {ceros_base})"
              f"{'  <-- VETO' if ceros > ceros_base else ''}")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, celdas[k][0], base[0])}
        print(f"  F1 gana: {[q for q in orden_q if d_f1[q] > 1e-9]}")
        print(f"  F1 pierde: {[q for q in orden_q if d_f1[q] < -1e-9]}\n")


if __name__ == "__main__":
    main()
