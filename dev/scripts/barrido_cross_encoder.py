#!/usr/bin/env python3
"""E39: `BAAI/bge-reranker-v2-m3` (cross-encoder) como re-puntuador.

HIPOTESIS, escrita antes de medir: un cross-encoder multilingue puntua pares
(consulta, fragmento) y ataca el modo de fallo documentado del pool -- el
desequilibrio ES/EN de los fenomenos 1 y 2 (NBQR/CBRN, on-orbit servicing),
donde gte compensa con penalizacion por idioma casi nula pero sigue sin leer
el par. ADL confirmo que la restriccion de la sec. 8.3 aplica a arquitecturas
decoder: el cross-encoder (XLM-RoBERTa, Apache 2.0) esta permitido.

JUSTIFICACION MECANICA. El eje "otro encoder" se cerro tres veces (E04, E25,
E31) pero siempre con BI-encoders usados como producto punto sobre vectores
almacenados. Un cross-encoder relee el texto del par, o sea que es otra
arquitectura de uso, y la condicion de E31 para reabrir se cumple a medias:
las 10 consultas independientes son el pool anotado propio.

QUE NO SE PRUEBA. No se construye indice nuevo: el re-puntuador actua sobre
los 200 candidatos que el primario ya trajo (pools_entregados.json), que es
donde actuan hoy gte+e5. Si esta medicion gana, la adopcion solo agrega un
re-puntuador al camino online -- sin indice nuevo, sin impacto LFS/Release.

RIESGOS DECLARADOS, fijados antes de medir:
  - Costo online: medido en el smoke test, ~850 ms/pair en CPU del evaluador
    => 85 s/consulta a 100 pares. Si el cross-encoder gana, ese costo se
    declara como trade-off explicito; no se re-calibra a ciegas.
  - Los puntajes del cross-encoder son logits (sigmoid -> [0,1]) y se suman a
    cosenos ([-1,1]): el peso 0.60 se calibro para cosenos. Por eso hay una
    celda CONTROL de escala (min-max por consulta) y el peso se toca en UN
    punto discreto (0.25), no en grilla.
  - Criterio de adopcion: IC al 90% del delta pareado excluye perdida de 0.02
    en las DOS muestras (50 y 10 independientes), ante empate se conserva la
    entregada, y las 11 consultas con F1@3 = 0 no pueden subir. Las 9 con
    etiqueta de agente van aparte.

E39b (variante de ventana completa): el 512 de MAX_LEN trunca los chunks que
superan esa longitud (~0.2% de los candidatos, p99 de tokens por chunk = 466).
Re-correr fase 1 con --max-length 8192 y fase 2 sobre esos scores decide si el
truncamiento explica la perdida en independientes. La paridad de la celda base
en fase 2 es ademas un chequeo de determinismo: los pares cortos no se truncan
ni a 512 ni a 8192 y deben reproducir los mismos scores.

Uso:
    .venv-cuda/Scripts/python.exe dev/scripts/barrido_cross_encoder.py --fase 1 [--max-length 8192]
    .venv/Scripts/python.exe      dev/scripts/barrido_cross_encoder.py --fase 2 [--max-length 8192]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from eval_mini import (  # noqa: E402
    bootstrap_delta,
    cargar_jsonl,
    f1,
    ndcg,
    ndcg_penalizado,
)

from generador import Hit, build_result_object, expandir_consulta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dev" / "intermedios" / "pools_entregados.json"
METADATA = (
    ROOT
    / "Entrega"
    / "base_vectorial"
    / "encoder_paraphrase-multilingual-MiniLM-L12-v2"
    / "metadata.jsonl"
)
CONSULTAS = ROOT / "dev" / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = ROOT / "dev" / "eval" / "ground_truth_mini.jsonl"
RESULTADOS = ROOT / "Entrega" / "resultados.jsonl"
SALIDA = ROOT / "dev" / "intermedios" / "cross_e39"

MODEL_ID = "BAAI/bge-reranker-v2-m3"
PESO = 0.60       # E01
AGG = "top5"      # E07
K_POOL = 100
MAX_LEN = 512

M_P, M_G, M_E = (
    "paraphrase-multilingual-MiniLM-L12-v2",
    "gte-multilingual-base",
    "multilingual-e5-base",
)

CELDAS = [
    ("BASE entregada (gte+e5@0.60)", "base", None),
    ("S100 add cross@0.60 (pool)", "s100_add", 0.60),
    ("S200 add cross@0.60", "s200_add", 0.60),
    ("S200 add cross@0.25", "s200_add", 0.25),
    ("S200 add cross@0.60 minmax", "s200_add_mm", 0.60),
    ("S100 replace cross@0.60 (pool)", "s100_rep", 0.60),
    ("S200 replace cross@0.60", "s200_rep", 0.60),
]
BASE = CELDAS[0][0]
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")


def cargar_candidatos(dump: dict, metadata: dict) -> dict[str, list[dict]]:
    """Los 200 candidatos por consulta, con texto y puntajes alineados.

    `crudos` y `similitudes` del dump comparten posicion. El pool (100) es el
    top-100 por base = s_p + 0.6 s_g + 0.6 s_e, verificado con error 0.0.
    """
    out = {}
    for qid, crudos in dump["crudos"].items():
        sim = dump["similitudes"][qid]
        cands = []
        for i, c in enumerate(crudos):
            meta = metadata[c["chunk_id"]]
            cands.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "fuente": meta["fuente"],
                "texto": meta["texto"],
                "formato": meta["formato"],
                "fenomeno": meta.get("fenomeno"),
                "idioma": meta.get("idioma"),
                "fila": c["fila"],
                "s_p": sim[M_P][i],
                "s_g": sim[M_G][i],
                "s_e": sim[M_E][i],
            })
        out[qid] = cands
    return out


def base_score(c: dict) -> float:
    return c["s_p"] + PESO * c["s_g"] + PESO * c["s_e"]


def leer_metadata_necesaria(dump: dict) -> dict:
    ids = {c["chunk_id"] for crudos in dump["crudos"].values() for c in crudos}
    out = {}
    with open(METADATA, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["chunk_id"] in ids:
                out[o["chunk_id"]] = o
    print(f"metadata cargada: {len(out)} de {len(ids)} chunks necesarios")
    return out


def puntaje_celda(c: dict, tipo: str, peso: float, x: float, lo: float, rng: float) -> float:
    if tipo == "base":
        return base_score(c)
    if tipo == "s100_add":
        return base_score(c) + peso * x
    if tipo == "s200_add":
        return base_score(c) + peso * x
    if tipo == "s200_add_mm":
        x_mm = (x - lo) / rng if rng > 1e-12 else 0.5
        return base_score(c) + peso * x_mm
    if tipo == "s100_rep":
        return c["s_p"] + peso * x
    if tipo == "s200_rep":
        return c["s_p"] + peso * x
    raise ValueError(tipo)


def armas_hits(cands: list[dict]) -> list[Hit]:
    hits = []
    for i, c in enumerate(cands, start=1):
        hits.append(Hit(
            rank=i,
            score=c["score"],
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            fuente=c["fuente"],
            texto=c["texto"],
            formato=c["formato"],
            fenomeno=c["fenomeno"],
            idioma=c["idioma"],
            fila=c["fila"],
        ))
    return hits


def evaluar_consulta(qid: str, cands: list[dict], tipo: str, peso: float, x_row: list[float],
                     consultas_por_id: dict) -> dict:
    pool_ids = {c["chunk_id"] for c in cands if c.get("en_pool")}
    lo = min(x_row)
    rng = max(x_row) - lo
    escoreados = []
    for c, x in zip(cands, x_row):
        if tipo.startswith("s100") and c["chunk_id"] not in pool_ids:
            continue
        c["score"] = puntaje_celda(c, tipo, peso, x, lo, rng)
        escoreados.append(c)
    escoreados.sort(key=lambda c: -c["score"])
    top = escoreados[:K_POOL]
    for i, c in enumerate(top, start=1):
        c["rank"] = i
    texto_expandido = consultas_por_id[qid]["expandida"]
    return build_result_object(qid, armas_hits(top), agg_strategy=AGG,
                               texto_consulta=texto_expandido)


def fase1(consultas, max_length) -> None:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    dump = json.load(open(DUMP, encoding="utf-8"))
    metadata = leer_metadata_necesaria(dump)
    cands_por_q = cargar_candidatos(dump, metadata)

    consultas_por_id = {c["query_id"]: c for c in consultas}
    print(f"cargando {MODEL_ID} en cuda (max_length={max_length})...", flush=True)
    t0 = time.time()
    model = CrossEncoder(MODEL_ID, max_length=max_length, device="cuda")
    print(f"carga: {time.time() - t0:.1f} s", flush=True)

    SALIDA.mkdir(parents=True, exist_ok=True)
    orden = []
    filas_x = []
    for c in consultas:
        qid = c["query_id"]
        expandida = expandir_consulta(c["text"])
        consultas_por_id[qid]["expandida"] = expandida
        cands = cands_por_q[qid]
        pares = [(expandida, cc["texto"]) for cc in cands]
        t0 = time.time()
        x = model.predict(pares, apply_sigmoid=True, batch_size=24)
        print(f"  {qid}: {len(pares)} pares en {time.time() - t0:.1f} s", flush=True)
        orden.append(qid)
        filas_x.append(np.asarray(x, dtype="float32"))

    X = np.stack(filas_x)
    if max_length == MAX_LEN:
        np.save(SALIDA / "X.npy", X)
        (SALIDA / "orden.json").write_text(json.dumps(orden), encoding="utf-8")
    else:
        np.save(SALIDA / f"X{max_length}.npy", X)
        (SALIDA / f"orden{max_length}.json").write_text(json.dumps(orden), encoding="utf-8")
    print(f"fase 1 lista: {X.shape} (max_length={max_length}) -> {SALIDA}", flush=True)


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


def fase2(consultas, max_length) -> None:
    if max_length == MAX_LEN:
        X = np.load(SALIDA / "X.npy")
        orden = json.loads((SALIDA / "orden.json").read_text(encoding="utf-8"))
    else:
        X = np.load(SALIDA / f"X{max_length}.npy")
        orden = json.loads((SALIDA / f"orden{max_length}.json").read_text(encoding="utf-8"))
    print(f"fase 2 sobre scores de max_length={max_length} ({X.shape})")
    dump = json.load(open(DUMP, encoding="utf-8"))
    metadata = leer_metadata_necesaria(dump)
    cands_por_q = cargar_candidatos(dump, metadata)
    # marcar los que estaban en el pool entregado (top-100 por base)
    for qid, cands in cands_por_q.items():
        pool_ids = {e["chunk_id"] for e in dump["pools"][qid]}
        for c in cands:
            c["en_pool"] = c["chunk_id"] in pool_ids

    consultas_por_id = {}
    for c in consultas:
        c["expandida"] = expandir_consulta(c["text"])
        consultas_por_id[c["query_id"]] = c

    gt_todo = cargar_jsonl(GT)
    gt = [g for g in gt_todo if g.get("docs_relevantes")]
    indep = [g for g in gt if not g.get("pool")]
    humanas = [g for g in gt if g.get("anotador") != "panel-agentes"]
    print(f"ground truth: {len(gt)} con relevantes, {len(indep)} independientes, {len(humanas)} humanas")

    res_base = {}
    for qi, qid in enumerate(orden):
        res_base[qid] = evaluar_consulta(qid, cands_por_q[qid], "base", PESO, X[qi], consultas_por_id)
    # paridad: la celda base debe reproducir resultados.jsonl
    entregados = {o["query_id"]: o for o in cargar_jsonl(RESULTADOS)}
    dif = sum(
        1 for qid, r in res_base.items()
        if r != entregados.get(qid)
    )
    print(f"PARIDAD base vs resultados.jsonl: {50 - dif} de 50 lineas identicas")

    filas_tabla, crudo = [], {}
    for etiqueta, tipo, peso in CELDAS:
        res = {}
        for qi, qid in enumerate(orden):
            res[qid] = evaluar_consulta(qid, cands_por_q[qid], tipo, peso or PESO, X[qi], consultas_por_id)
        vals = []
        for sub in (gt, indep, humanas):
            vals.extend(np.mean(m) if m else float("nan") for m in metricas(res, sub))
        crudo[etiqueta] = {"gt": metricas(res, gt), "ind": metricas(res, indep), "hum": metricas(res, humanas)}
        filas_tabla.append((etiqueta, vals))
        print(f"  {etiqueta:30} " + " ".join(f"{v:.3f}" for v in vals), flush=True)

    print("\n| celda | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for etiqueta, vals in filas_tabla:
        print(f"| {etiqueta} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    print(f"\nDeltas pareados contra '{BASE}' (IC al 90% por bootstrap):")
    for etiqueta, _, _ in CELDAS[1:]:
        for muestra in ("gt", "ind", "hum"):
            for j, nombre in enumerate(("F1", "ND", "NDp")):
                a, b = crudo[BASE][muestra][j], crudo[etiqueta][muestra][j]
                if not a:
                    continue
                d, lo, hi = bootstrap_delta([x - y for x, y in zip(b, a)])
                marca = "  <-- pasa" if lo > -0.02 and d > 0 else ""
                print(f"  {etiqueta:30} {muestra:4} {nombre:4} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{marca}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fase", type=int, choices=(1, 2), required=True)
    ap.add_argument("--max-length", type=int, default=MAX_LEN)
    args = ap.parse_args()

    consultas = cargar_jsonl(CONSULTAS)
    (fase1 if args.fase == 1 else fase2)(consultas, args.max_length)


if __name__ == "__main__":
    main()
