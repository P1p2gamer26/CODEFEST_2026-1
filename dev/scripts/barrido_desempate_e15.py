#!/usr/bin/env python3
"""E15: desempate estable por identificador, para que la entrega deje de
depender de la maquina que la genera.

HIPOTESIS, escrita antes de medir: fijar un desempate explicito por `chunk_id`
(y por `doc_id` en la agregacion) NO mueve ninguna de las nueve lecturas, y a
cambio vuelve la salida reproducible en cualquier CPU.

JUSTIFICACION MECANICA. `dev/experimentos/hallazgo_reproducibilidad.md` midio
que la corrida en frio local difiere en 2 lineas de 50 contra el
`resultados.jsonl` generado en la VM, con F1@3 y NDCG@10 identicos. Una de las
dos causas son **empates EXACTOS**: cuatro fragmentos de F2-CSIS-049 comparten
el score 1.6724899768829347 bit a bit. Ante empate exacto, `sort` (estable)
conserva el orden de llegada desde FAISS, que depende de la maquina. Un
desempate por `chunk_id` lo fija. La otra causa (q005, brecha de 2.6e-05 entre
dos documentos) NO es un empate y este cambio no puede arreglarla: se mide
para acotar cuanto del problema se cierra, no para pretender que se cierra
entero.

Por que el criterio de decision aqui NO es "gana metrica": es la leccion 7 --
un defecto se arregla porque es un defecto. Lo que se le exige al cambio es
**no perder**: si mueve las metricas mas alla del ruido, se rechaza, porque
entonces no seria un desempate sino una politica de ordenamiento nueva.

RIESGOS Y MITIGACIONES, fijados ANTES de medir:
  - Un desempate por identificador introduce un sesgo sistematico (los
    chunk_id bajos ganan siempre). Entre scores EXACTAMENTE iguales eso es
    arbitrario pero inocuo; lo que no puede pasar es que toque candidatos con
    scores distintos. Por eso el desempate va como CLAVE SECUNDARIA, nunca
    reemplazando al score.
  - Si alguna lectura se mueve mas de 0.02 en contra, se rechaza.
  - Regla de E09: la fila base tiene que reproducir 0.440 / 0.490 / 0.476.
  - No toca `Entrega/` (regla 6 del handoff): es decision humana.

COSTE: una sola pasada de FAISS; las celdas reordenan el mismo pool.

Uso:
    python dev/scripts/barrido_desempate_e15.py
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

PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60

CELDAS = ("entregada", "desempate")
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

_agg_original = generador.aggregate_documents


def construir_hits(consultas, cache_idx):
    """Recupera y re-puntua UNA vez. No ordena: cada celda ordena a su manera."""
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
        por_consulta[c["query_id"]] = hits
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta


def ordenar_pool(hits, desempatar: bool):
    orden = sorted(hits, key=(lambda h: (-h.score, h.chunk_id)) if desempatar
                   else (lambda h: -h.score))
    recorte = orden[:K_POOL]
    for i, h in enumerate(recorte, 1):
        h.rank = i
    return recorte


def agg_desempatada(hits, top_n=3, strategy=AGG):
    """Agrega igual que siempre y reordena con doc_id como clave secundaria.

    Se pide top_n grande y se recorta despues: si se recortara antes, el
    desempate no podria ver a los documentos que quedaron justo fuera.
    """
    docs = _agg_original(hits, top_n=len(hits), strategy=strategy)
    docs.sort(key=lambda d: (-d.score, d.doc_id))
    return docs[:top_n]


def celda_resultados(hits_por_consulta, celda):
    desempatar = celda == "desempate"
    if desempatar:
        generador.aggregate_documents = agg_desempatada
    try:
        return {
            qid: generador.build_result_object(
                qid, ordenar_pool(hits, desempatar), agg_strategy=AGG
            )
            for qid, hits in hits_por_consulta.items()
        }
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "desempate_e15")
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
    hits_por_consulta = construir_hits(consultas, cache_idx)

    guardadas, archivos = {}, {}
    for celda in CELDAS:
        res = celda_resultados(hits_por_consulta, celda)
        ruta = args.out_dir / f"{celda}.jsonl"
        with ruta.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        archivos[celda] = ruta
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':14s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in guardadas:
        print(f"{k + (' *' if k == BASE else ''):14s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)\n")

    # Cuantas lineas cambia: es la mitad util del experimento, porque el
    # objetivo NO es mover metrica sino fijar el orden.
    lineas_base = archivos[BASE].read_text(encoding="utf-8").splitlines()
    lineas_var = archivos["desempate"].read_text(encoding="utf-8").splitlines()
    distintas = sum(1 for a, b in zip(lineas_base, lineas_var) if a != b)
    print(f"lineas distintas contra la entregada: {distintas} de {len(lineas_base)}")

    print("\ndeltas pareados, IC al 90%. El criterio aqui es NO PERDER:")
    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        print(f"\n  === {k} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            gana = sum(1 for d in deltas if d > 1e-9)
            pierde = sum(1 for d in deltas if d < -1e-9)
            veredicto = "pasa" if lo > -0.02 else "no pasa"
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"{gana}g/{pierde}p  {veredicto}")


if __name__ == "__main__":
    main()
