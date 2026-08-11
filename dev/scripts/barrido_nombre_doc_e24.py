#!/usr/bin/env python3
"""E24: el NOMBRE del documento como senal externa a los scores de chunk.

HIPOTESIS, escrita antes de medir (ver cola.jsonl): anadir
cos(consulta, nombre del documento) como desempate entre los documentos que
SATURAN el tope de `top5` sube el F1@3.

JUSTIFICACION MECANICA. E19/E20: el ranking de documentos lo decide el conteo
de chunks, y el 92.7% de los que ocupan cupo satura el tope de top5 -- dentro
de ese conjunto el conteo es constante y por tanto NO discrimina (E20 midio que
dividir por la raiz del conteo da un archivo byte a byte identico). El
diagnostico de las 11 consultas con F1@3=0 (diagnostico_ceros_e22.py) muestra
que ninguna es fallo de pool y que el mejor chunk del perdedor no es peor que
el del ganador. Hace falta una senal EXTERNA a los scores de chunk. `fuente`
son titulos de verdad, 1691 distintos, y ninguno entro jamas a un vector.

FORMULACION SIN PARAMETRO (la que decide): los documentos saturados se permutan
ENTRE SI, dentro de las posiciones que ya ocupan en el ranking agregado, por
cos(consulta, nombre). Nada mas se mueve. El peso continuo va como CONTROL y
NO es adoptable por ser el argmax de una grilla (leccion 2).

VETOS pre-registrados:
  - Si la ganancia se concentra en las 9 consultas de etiqueta de agente, se
    rechaza (misma firma que doc_rrf y gte-primario).
  - Si el efecto NO aparece en las 11 consultas con F1@3=0, la justificacion
    mecanica era falsa y hay que decirlo aunque el promedio suba.
  - Regla de E09: la fila base reproduce 0.440 / 0.490 / 0.476.
  - No toca `Entrega/`.

Uso:
    python dev/scripts/barrido_nombre_doc_e24.py
"""

import argparse
import json
import statistics
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
from nombres_doc_e24 import MIN_TOKENS, mapa_fuentes  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60
M_TOPE = int(AGG[3:])          # un documento satura si aporta >= 5 chunks

CELDAS = ("entregada", "nombre-satura", "nombre-w010", "nombre-w030")
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

# Las 11 con F1@3 = 0 en la entrega (diagnostico_ceros_e22.py). Es donde la
# justificacion mecanica predice el efecto.
CEROS = "q003 q005 q007 q011 q014 q015 q034 q037 q044 q046 q047".split()

_agg_original = generador.aggregate_documents
DocumentHit = None
CTX = {"cos": {}}              # doc_id -> cos(consulta, nombre), por consulta


def _cos_docs(docs):
    """cos por documento; los OPACOS reciben el neutro (mediana de los demas)."""
    c = CTX["cos"]
    vals = [c[d] for d in docs if d in c]
    neutro = statistics.median(vals) if vals else 0.0
    return {d: c.get(d, neutro) for d in docs}


def _chunks_por_doc(hits):
    d = defaultdict(int)
    for h in hits:
        d[h.doc_id] += 1
    return d


def agg_satura(hits, top_n=3, strategy=AGG):
    """Permuta ENTRE SI los documentos saturados, dejando el resto en su sitio."""
    base = _agg_original(hits, top_n=len(hits), strategy=strategy)
    n_chunks = _chunks_por_doc(hits)
    idx_sat = [i for i, d in enumerate(base) if n_chunks[d.doc_id] >= M_TOPE]
    cos = _cos_docs([base[i].doc_id for i in idx_sat])
    # `sorted` es estable: si dos empatan en cos conservan el orden agregado.
    orden = sorted(idx_sat, key=lambda i: -cos[base[i].doc_id])
    salida = list(base)
    for destino, origen in zip(idx_sat, orden):
        salida[destino] = base[origen]
    return [DocumentHit(rank=i, doc_id=d.doc_id, score=d.score)
            for i, d in enumerate(salida[:top_n], 1)]


def agg_peso(hits, top_n, strategy, w):
    """CONTROL: peso continuo sobre TODOS los documentos. No adoptable."""
    base = _agg_original(hits, top_n=len(hits), strategy=strategy)
    cos = _cos_docs([d.doc_id for d in base])
    orden = sorted(base, key=lambda d: -(d.score + w * cos[d.doc_id]))
    return [DocumentHit(rank=i, doc_id=d.doc_id, score=d.score)
            for i, d in enumerate(orden[:top_n], 1)]


AGREGADORES = {
    "entregada": _agg_original,
    "nombre-satura": agg_satura,
    "nombre-w010": lambda h, top_n=3, strategy=AGG: agg_peso(h, top_n, strategy, 0.10),
    "nombre-w030": lambda h, top_n=3, strategy=AGG: agg_peso(h, top_n, strategy, 0.30),
}


def construir(consultas, cache_idx, metadata):
    """Pool por consulta + cos(consulta, nombre) por documento informativo."""
    mapa = mapa_fuentes(metadata)
    informativos = {d: n for d, (_, n) in mapa.items()
                    if len(n.split()) >= MIN_TOKENS}
    docs = sorted(informativos)
    enc_p = get_encoder(name=MINILM)
    print(f"  codificando {len(docs):,} nombres informativos...", flush=True)
    vecs = np.asarray(enc_p.encode_passages([informativos[d] for d in docs]),
                      dtype="float32")

    idx_p, _ = cache_idx[MINILM]
    por_consulta, cos_por_consulta = {}, {}
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
        qv = np.asarray(enc_p.encode_query(texto), dtype="float32")
        cos_por_consulta[c["query_id"]] = dict(zip(docs, map(float, vecs @ qv)))
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta, cos_por_consulta, informativos


def auditar_pares(por_consulta, cos_por_consulta, informativos):
    """Sobre cuantos documentos actua realmente el criterio sin parametro."""
    tot_sat, tot_sat_inf, permutables, en_top3 = 0, 0, 0, 0
    for qid, hits in por_consulta.items():
        CTX["cos"] = cos_por_consulta[qid]
        base = _agg_original(hits, top_n=len(hits), strategy=AGG)
        n_chunks = _chunks_por_doc(hits)
        sat = [d.doc_id for d in base if n_chunks[d.doc_id] >= M_TOPE]
        tot_sat += len(sat)
        tot_sat_inf += sum(1 for d in sat if d in informativos)
        permutables += len(sat) * (len(sat) - 1) // 2
        en_top3 += sum(1 for d in base[:3] if n_chunks[d.doc_id] >= M_TOPE)
    n = len(por_consulta)
    print(f"\n  PUERTA DE ENTRADA (sobre {n} consultas)")
    print(f"    documentos saturados por consulta:     {tot_sat / n:6.1f}")
    print(f"    ...de ellos con nombre informativo:    {tot_sat_inf / n:6.1f} "
          f"({tot_sat_inf / max(tot_sat, 1):.1%})")
    print(f"    pares permutables por consulta:        {permutables / n:6.1f}")
    print(f"    del top-3 entregado, saturados:        {en_top3 / (3 * n):.1%}")


def celda_resultados(por_consulta, cos_por_consulta, celda):
    generador.aggregate_documents = AGREGADORES[celda]
    try:
        out = {}
        for qid, hits in por_consulta.items():
            CTX["cos"] = cos_por_consulta[qid]
            out[qid] = generador.build_result_object(qid, hits, agg_strategy=AGG)
        return out
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "nombre_doc_e24")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    DocumentHit = generador.DocumentHit

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_ag = [g for g in gt_todo if g.get("anotador")]
    gt_ceros = [g for g in gt_todo if g["query_id"] in CEROS]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas, "
          f"{len(gt_ag)} de agente, {len(gt_ceros)} con F1=0\n")

    cache_idx, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (
                faiss.read_index(str(encoder_dir(nombre) / "index.faiss")), metadata)
    print(f"  {cache_idx[MINILM][0].ntotal:,} vectores por indice\n", flush=True)

    por_consulta, cos_por_consulta, informativos = construir(
        consultas, cache_idx, metadata)
    auditar_pares(por_consulta, cos_por_consulta, informativos)

    guardadas, por_agente, por_ceros, lineas = {}, {}, {}, {}
    for celda in CELDAS:
        res = celda_resultados(por_consulta, cos_por_consulta, celda)
        ruta = args.out_dir / f"{celda}.jsonl"
        with ruta.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        lineas[celda] = ruta.read_bytes().splitlines()
        guardadas[celda] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                            + metricas(res, gt_hum))
        por_agente[celda] = metricas(res, gt_ag)
        por_ceros[celda] = metricas(res, gt_ceros)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):16s}"
              + "".join(f"{sum(v) / len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)\n")

    base = guardadas[BASE]
    for k in CELDAS:
        if k == BASE:
            continue
        distintas = sum(1 for a, b in zip(lineas[k], lineas[BASE]) if a != b)
        print(f"\n  === {k} ===   {distintas}/50 lineas distintas de la entregada")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            gana = sum(1 for d in deltas if d > 1e-9)
            pierde = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{gana}g/{pierde}p  {'pasa' if lo > -0.02 else 'no pasa'}")
        d_ag = (sum(por_agente[k][0]) / len(por_agente[k][0])
                - sum(por_agente[BASE][0]) / len(por_agente[BASE][0]))
        d_hum = sum(guardadas[k][6]) / len(guardadas[k][6]) - sum(base[6]) / len(base[6])
        print(f"  VETO agente: delta F1 {d_ag:+.3f} contra {d_hum:+.3f} en las 41 "
              "humanas" + ("  <- SE ACTIVA" if d_ag > 0 and d_hum <= 0 else ""))
        for j, nombre in zip((0, 1), ("F1", "NDCG")):
            v, b = por_ceros[k][j], por_ceros[BASE][j]
            gana = sum(1 for x, y in zip(v, b) if x > y + 1e-9)
            print(f"  11 CEROS {nombre:4s}: {sum(b) / len(b):.3f} -> "
                  f"{sum(v) / len(v):.3f}   {gana} rescatadas")


if __name__ == "__main__":
    main()
