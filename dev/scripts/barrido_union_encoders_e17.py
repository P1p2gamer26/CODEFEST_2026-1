#!/usr/bin/env python3
"""E17: unir el pool de MiniLM con el de gte y re-puntuar la union.

HIPOTESIS, pre-registrada en dev/experimentos/cola.jsonl: unir los dos pools
recupera documentos que MiniLM solo nunca ve.

JUSTIFICACION MECANICA (de la cola, sin cambios): gte como PRIMARIO se
descarto por sesgo de pooling, no por calidad, y alinea entre idiomas mejor
que MiniLM (penalizacion por idioma -0.027/+0.036 contra +0.052/+0.091). El
sesgo de pooling juega EN CONTRA aqui: los documentos que solo gte trae nunca
se le mostraron a un anotador, asi que cualquier ganancia medida es COTA
INFERIOR. Coste de codificacion cero: los tres indices existen.

RIESGO, de la cola: (1) declarar el sesgo y NO concluir "gte no aporta" ante
empate; (2) dilucion -- por eso va la celda de profundidad igualada; (3) si
gana SOLO en las 9 de etiqueta asistida, se rechaza; (4) E18 acota el techo a
8 pares de 207 (3.9%) con reparto simetrico (8 solo-gte contra 8 solo-MiniLM),
asi que la expectativa esta corregida a la baja ANTES de correr.

Todas las celdas se re-puntuan igual: score = cos_minilm + 0.60*(cos_gte +
cos_e5). Los candidatos que solo aporta gte reciben su cos_minilm por
reconstruct(fila), o sea que la formula es identica para todos y lo que cambia
es unicamente QUE candidatos entran.

Regla de E09: la fila base tiene que reproducir 0.440 / 0.490 / 0.476.
NO toca Entrega/.

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_union_encoders_e17.py
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
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

K_POOL, AGG, PESO = 100, "top5", 0.60

# celda -> (profundidad MiniLM, profundidad gte). 0 = esa rama no aporta.
CELDAS = {
    "entregada": (200, 0),
    "union200": (200, 200),
    "union100": (100, 100),
}
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")


CACHE_GTE = DEV_DIR / "intermedios" / "union_e17" / "filas_gte.json"


def fase_gte(consultas, prof):
    """Fase 1, aislada: con el indice y el modelo de gte cargados y NADA mas,
    volcar a disco los `prof` mejores filas por consulta.

    Va aparte porque tener a la vez los tres indices (~940 MB) y el modelo de
    gte no entra en esta maquina: el harness de E15 lo hacia, pero ahi gte
    solo se usaba por reconstruct(fila) y su modelo se cargaba despues de las
    busquedas. Aca gte tiene que BUSCAR.
    """
    idx_g, metadata = load_index(GTE)
    enc_g = get_encoder(name=GTE)
    out = {}
    for n, c in enumerate(consultas, 1):
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_g, idx_g, metadata, k=prof)[:prof]
        out[c["query_id"]] = [h.fila for h in hits]
        print(f"    gte {n}/{len(consultas)}", end="\r", flush=True)
    print()
    CACHE_GTE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_GTE.write_text(json.dumps(out), encoding="utf-8")
    print(f"  filas de gte -> {CACHE_GTE}")


def construir_pools(consultas, cache_idx, metadata):
    """Un pool re-puntuado por celda y consulta. Una sola pasada de FAISS por
    rama y profundidad maxima; las celdas menores salen por rebanado."""
    from src.retrieval.search import Hit

    enc_m = get_encoder(name=MINILM)
    idx_m = cache_idx[MINILM]
    prof_max_m = max(p for p, _ in CELDAS.values())
    filas_gte = json.loads(CACHE_GTE.read_text(encoding="utf-8"))

    pools = {c: {} for c in CELDAS}
    aportes = {c: 0 for c in CELDAS}   # candidatos que solo trajo gte
    for n, c in enumerate(consultas, 1):
        qid, texto = c["query_id"], expandir_consulta(c["text"])
        hits_m = search(texto, enc_m, idx_m, metadata, k=prof_max_m)[:prof_max_m]
        # score base homogeneo: coseno con MiniLM para TODO candidato.
        qv_m = enc_m.encode_query(texto)
        hits_g = []
        for fila in filas_gte[qid]:
            meta = metadata[fila]
            hits_g.append(Hit(
                rank=0, score=float(np.dot(qv_m, idx_m.reconstruct(fila))),
                chunk_id=meta["chunk_id"], doc_id=meta["doc_id"],
                fuente=meta["fuente"], texto=meta["texto"],
                formato=meta["formato"], fenomeno=meta.get("fenomeno"),
                idioma=meta.get("idioma"), fila=fila,
            ))
        qv_sec = {s: get_encoder(name=s).encode_query(texto) for s in SECUNDARIOS}

        for celda, (pm, pg) in CELDAS.items():
            porf = {}
            for h in hits_m[:pm]:
                porf.setdefault(h.fila, h)
            solo_gte = 0
            for h in hits_g[:pg]:
                if h.fila not in porf:
                    porf[h.fila] = h
                    solo_gte += 1
            aportes[celda] += solo_gte
            # copias independientes: la cascada muta score en sitio
            cand = [type(h)(**{**h.__dict__}) for h in porf.values()]
            for sec in SECUNDARIOS:
                idx_s, qv = cache_idx[sec], qv_sec[sec]
                for h in cand:
                    h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
            pool = sorted(cand, key=lambda h: -h.score)[:K_POOL]
            for i, h in enumerate(pool, 1):
                h.rank = i
            pools[celda][qid] = pool
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return pools, aportes


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
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "union_e17")
    ap.add_argument("--fase", choices=("gte", "barrido"), default="barrido")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.fase == "gte":
        fase_gte(cargar_jsonl(CONSULTAS), max(g for _, g in CELDAS.values()))
        return

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_agente = {g["query_id"] for g in gt_todo if g.get("anotador")}
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas\n")

    cache_idx, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            cache_idx[nombre], metadata = load_index(nombre)
        else:
            cache_idx[nombre] = faiss.read_index(str(d / "index.faiss"))
        print(f"  {nombre}: {cache_idx[nombre].ntotal:,} vectores", flush=True)

    print("\nconstruyendo pools...", flush=True)
    pools, aportes = construir_pools(consultas, cache_idx, metadata)
    for celda, n in aportes.items():
        print(f"  {celda:10s} candidatos que SOLO aporta gte: {n} "
              f"({n / max(1, len(consultas)):.1f} por consulta)")

    guardadas = {}
    for celda in CELDAS:
        res = {qid: generador.build_result_object(qid, pool, agg_strategy=AGG)
               for qid, pool in pools[celda].items()}
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                            + metricas(res, gt_hum))
        guardadas[celda + "__res"] = res

    print(f"\n{'celda':12s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):12s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)")

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
        # veto pre-registrado nº 3: victorias concentradas en etiqueta asistida
        dq = {g["query_id"]: (f1([d["doc_id"] for d in
                                 guardadas[k + '__res'][g["query_id"]]["documents"][:3]],
                                set(g["docs_relevantes"]))[2]
                              - f1([d["doc_id"] for d in
                                   guardadas[BASE + '__res'][g["query_id"]]["documents"][:3]],
                                  set(g["docs_relevantes"]))[2])
              for g in gt_todo}
        g_ag = sum(1 for q, d in dq.items() if d > 1e-9 and q in gt_agente)
        g_hu = sum(1 for q, d in dq.items() if d > 1e-9 and q not in gt_agente)
        p_ag = sum(1 for q, d in dq.items() if d < -1e-9 and q in gt_agente)
        p_hu = sum(1 for q, d in dq.items() if d < -1e-9 and q not in gt_agente)
        print(f"  reparto F1: humanas {g_hu}g/{p_hu}p, etiqueta-agente {g_ag}g/{p_ag}p")


if __name__ == "__main__":
    main()
