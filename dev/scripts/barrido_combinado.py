#!/usr/bin/env python3
"""E06: las dos palancas adoptables de la ronda, JUNTAS.

E01/E01b dejo el peso 0.60 como adoptable (F1@3 50: 0.402 -> 0.425) y E03
dejo tres entradas nuevas de glosario (0.402 -> 0.423). Las dos se midieron
por separado contra la MISMA base entregada, asi que sumar +0.023 y +0.021 y
esperar 0.44 seria extrapolar: no se sabe si se componen o si rescatan las
mismas consultas.

JUSTIFICACION MECANICA, escrita antes de medir: las dos palancas actuan en
puntos distintos del camino -- el glosario cambia el VECTOR DE CONSULTA (o
sea, que candidatos entran al pool) y el peso cambia el ORDEN dentro del pool
ya recuperado. Sobre esa base deberian componerse. Pero E03 rescata q002,
q017 y q032 metiendo documentos nuevos al pool, y el peso 0.60 reordena: si
el re-puntuador con mas peso hunde justo a esos documentos nuevos, la
combinacion vale menos que cualquiera de las dos.

REGLA DE DECISION, fijada antes de ver los numeros (la misma de E01 y E03):
se adopta la combinacion solo si el IC al 90% del delta pareado contra lo
entregado excluye una perdida de 0,02 en las dos muestras. Si la combinacion
NO supera a la mejor palanca individual, se entrega la palanca sola: dos
cambios que no se componen son un cambio de mas.

Coste: cero codificacion nueva, los tres indices ya existen.

Uso:
    python dev/scripts/barrido_combinado.py
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
)

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)
RERANK_DEPTH = 200
K_POOL = 100
ESTRATEGIA = "top5"

# Las tres que E03 dejo adoptables. Las otras cuatro candidatas quedaron
# fuera por la regla de antemano (efecto nulo no entra, y "capacidades laser"
# perdia q024).
EXTRA_E03 = (
    ("derecho internacional en el espacio", "international space law"),
    ("dominio espacial", "space domain"),
    ("sistemas no tripulados", "unmanned systems UAV"),
)

# (etiqueta, peso del re-puntuador, entradas extra de glosario)
CELDAS = (
    ("entregado (0.25, glos base)", 0.25, ()),
    ("E01 solo (0.60, glos base)", 0.60, ()),
    ("E03 solo (0.25, glos+3)", 0.25, EXTRA_E03),
    ("E01+E03 (0.60, glos+3)", 0.60, EXTRA_E03),
)
BASE = CELDAS[0][0]


def expandir(texto: str, extra) -> str:
    """La expansion adoptada mas las entradas candidatas que apliquen."""
    salida = glos.expandir_consulta(texto)
    normalizada = glos._normalizar(texto)
    for termino, ingles in extra:
        if termino in normalizada:
            salida = f"{salida} {ingles}"
    return salida


def candidatos(consultas, cache_idx, extra):
    """Pool y similitudes crudas para una expansion dada.

    Se calcula una vez por VARIANTE DE GLOSARIO (no por celda): el peso se
    aplica despues sobre esta misma lista, asi que las dos celdas que
    comparten glosario comparten tambien esta recuperacion.
    """
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for c in consultas:
        texto = expandir(c["text"], extra)
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)[:RERANK_DEPTH]
        primarios = [h.score for h in hits]
        sims = {}
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            sims[sec] = [float(np.dot(qv, idx_s.reconstruct(h.fila))) for h in hits]
        por_consulta[c["query_id"]] = (hits, primarios, sims)
    return por_consulta


def evaluar_celda(por_consulta, peso):
    resultados = {}
    for qid, (hits, primarios, sims) in por_consulta.items():
        for i, h in enumerate(hits):
            h.score = primarios[i] + sum(peso * sims[sec][i] for sec in SECUNDARIOS)
        recorte = sorted(hits, key=lambda h: -h.score)[:K_POOL]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        resultados[qid] = generador.build_result_object(
            qid, recorte, agg_strategy=ESTRATEGIA
        )
    return resultados


def metricas(res, gt):
    f1s, nds = [], []
    for g in gt:
        qid = g["query_id"]
        if qid not in res:
            continue
        rel = set(g["docs_relevantes"])
        f1s.append(f1([d["doc_id"] for d in res[qid]["documents"][:3]], rel)[2])
        nds.append(ndcg(res[qid]["fragments"], rel))
    return f1s, nds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir", type=Path, default=DEV_DIR / "intermedios" / "combinado"
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    gt_hum = [f for f in gt_todo if not f.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes, {len(gt_hum)} humanas\n")

    cache_idx = {}
    metadata = None
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

    pools = {}
    for clave, extra in (("base", ()), ("mas3", EXTRA_E03)):
        print(f"\nrecuperando con glosario {clave}...", flush=True)
        pools[clave] = candidatos(consultas, cache_idx, extra)

    guardadas, tablas = {}, {}
    for etiqueta, peso, extra in CELDAS:
        res = evaluar_celda(pools["mas3" if extra else "base"], peso)
        tablas[etiqueta] = res
        nombre = etiqueta.split(" ")[0].replace("(", "").lower()
        salida = args.out_dir / f"{nombre}_{peso:.2f}_{'mas3' if extra else 'base'}.jsonl"
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        f_a, n_a = metricas(res, gt_todo)
        f_i, n_i = metricas(res, gt_indep)
        f_h, n_h = metricas(res, gt_hum)
        guardadas[etiqueta] = (f_a, n_a, f_i, n_i, f_h, n_h)

    print(f"\n{'celda':30s} {'F1(50)':>8s} {'NDCG(50)':>9s} {'F1(ind)':>8s} {'NDCG(ind)':>10s} {'F1(hum)':>8s} {'NDCG(hum)':>10s}")
    for etiqueta, *_ in CELDAS:
        g = guardadas[etiqueta]
        print(
            f"{etiqueta:30s} " + " ".join(
                f"{sum(v)/len(v):>{w}.3f}" for v, w in zip(g, (8, 9, 8, 10, 8, 10))
            )
        )

    print(f"\ndeltas pareados contra '{BASE}', IC al 90%:")
    base = guardadas[BASE]
    for etiqueta, *_ in CELDAS[1:]:
        print(f"\n  === {etiqueta} ===")
        for nombre, i in (
            ("F1@3    50   ", 0),
            ("NDCG@10 50   ", 1),
            ("F1@3    indep", 2),
            ("NDCG@10 indep", 3),
            ("F1@3    human", 4),
            ("NDCG@10 human", 5),
        ):
            deltas = [x - y for x, y in zip(guardadas[etiqueta][i], base[i])]
            media, bajo, alto = bootstrap_delta(deltas)
            gana = sum(1 for d in deltas if d > 1e-9)
            pierde = sum(1 for d in deltas if d < -1e-9)
            ok = "ADOPTABLE" if bajo > -0.02 else "no pasa"
            print(f"  {nombre}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {gana}g/{pierde}p  {ok}")

    # La comparacion que decide si la combinacion vale la pena: contra la
    # mejor palanca individual, no solo contra lo entregado.
    print("\ndelta de la combinacion contra cada palanca sola:")
    for rival in (CELDAS[1][0], CELDAS[2][0]):
        print(f"\n  === E01+E03 vs {rival} ===")
        for nombre, i in (("F1@3 50", 0), ("NDCG@10 50", 1), ("F1@3 indep", 2), ("NDCG@10 indep", 3)):
            deltas = [x - y for x, y in zip(guardadas[CELDAS[3][0]][i], guardadas[rival][i])]
            media, bajo, alto = bootstrap_delta(deltas)
            print(f"  {nombre:14s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]")


if __name__ == "__main__":
    main()
