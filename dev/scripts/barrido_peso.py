#!/usr/bin/env python3
"""Barrido del PESO del re-puntuador sobre la cascada triple ya entregada.

E01 de dev/experimentos/cola.jsonl. El peso 0,25 se fijo cuando la cascada
corria con k_pool=60, agregacion sum y sin glosario (ver barrido_pool.py y
barrido_glosario.py, ya adoptados). Las tres cosas cambiaron desde entonces:
el pool paso a 100, la agregacion a top5 y el glosario quedo activo. Nada de
eso volvio a barrer el peso, asi que no hay razon para que 0,25 siga siendo
el optimo bajo el regimen nuevo.

JUSTIFICACION MECANICA, escrita antes de ver los numeros: el peso pondera la
similitud del re-puntuador (gte, e5) contra la del primario (MiniLM) al sumar
`h.score += peso * sim_secundario`. Al ampliar el pool a 100 entran
candidatos con score primario mas bajo -- el mismo peso absoluto pesa mas en
relacion a un score primario chico que a uno grande, asi que el punto donde
el re-puntuador empieza a dominar (o a diluirse) se mueve con el pool.

REGLA DE DECISION FIJADA ANTES DE MEDIR (docs/lecciones_metodologia.md): un
peso continuo barrido contra 50 consultas es la maquina de sobreajuste de la
leccion 2 (elegir el maximo de una grilla es la misma trampa que ya costo una
decision equivocada). Se adopta un peso nuevo solo si el IC al 90% del delta
pareado excluye una perdida de 0,02 EN LAS DOS MUESTRAS (las 50 y las 10
independientes), y ante empate se prefiere el valor entregado.

Coste: cero codificacion nueva. Los tres indices ya existen y los vectores de
los candidatos se leen con `reconstruct(fila)`, igual que barrido_pool.py.

Uso:
    python dev/scripts/barrido_peso.py
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
from eval_mini import (  # noqa: E402
    bootstrap_delta,
    cargar_jsonl,
    f1,
    ndcg,
    veredicto_bootstrap,
)

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"

PRIMARIO = MINILM
SECUNDARIOS = (GTE, E5)
RERANK_DEPTH = 200

# La configuracion entregada, fija en este barrido: solo el peso se mueve.
K_POOL = 100
ESTRATEGIA = "top5"
EXPANDIR = True

PESOS = [0.10, 0.25, 0.40, 0.60]
BASE_PESO = 0.25  # el entregado


def candidatos_por_consulta(consultas, cache_idx):
    """Recupera y guarda las similitudes CRUDAS de cada secundario, una vez.

    El peso se aplica DESPUES sobre esta misma lista: ninguna celda del
    barrido vuelve a tocar FAISS ni a recodificar la consulta.
    """
    enc_p = get_encoder(name=PRIMARIO)
    idx_p, metadata = cache_idx[PRIMARIO]
    por_consulta = {}
    for c in consultas:
        texto = expandir_consulta(c["text"]) if EXPANDIR else c["text"]
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)
        hits = hits[:RERANK_DEPTH]
        sims = {}
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            sims[sec] = [float(np.dot(qv, idx_s.reconstruct(h.fila))) for h in hits]
        por_consulta[c["query_id"]] = (hits, sims)
    return por_consulta


def evaluar_celda(por_consulta, peso):
    resultados = {}
    for qid, (hits, sims) in por_consulta.items():
        # Recalcular desde el score primario puro (guardado antes del barrido)
        # para no acumular el peso de una celda anterior.
        for i, h in enumerate(hits):
            h.score = h.score_primario + sum(peso * sims[sec][i] for sec in SECUNDARIOS)
        hits_ordenados = sorted(hits, key=lambda h: -h.score)
        recorte = hits_ordenados[:K_POOL]
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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "peso")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    gt_hum = [f for f in gt_todo if not f.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} consultas evaluables, {len(gt_indep)} independientes\n")

    cache_idx = {}
    metadata = None
    for nombre in (PRIMARIO, *SECUNDARIOS):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores")

    print(f"\nrecuperando y re-puntuando (pool={K_POOL}, agg={ESTRATEGIA}, glosario={EXPANDIR})...", flush=True)
    por_consulta = candidatos_por_consulta(consultas, cache_idx)
    # Guardar el score primario puro antes de que evaluar_celda lo mute.
    for hits, _ in por_consulta.values():
        for h in hits:
            h.score_primario = h.score

    tablas = {}
    for peso in PESOS:
        res = evaluar_celda(por_consulta, peso)
        tablas[peso] = res
        salida = args.out_dir / f"peso{peso:.2f}.jsonl"
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        print(f"  escrito {salida.name}", flush=True)

    print(f"\n{'peso':10s} {'F1(50)':>8s} {'NDCG(50)':>9s} {'F1(indep)':>10s} {'NDCG(indep)':>12s}")
    guardadas = {}
    for peso, res in tablas.items():
        f_a, n_a = metricas(res, gt_todo)
        f_i, n_i = metricas(res, gt_indep)
        f_h, n_h = metricas(res, gt_hum)
        guardadas[peso] = (f_a, n_a, f_i, n_i, f_h, n_h)
        etiqueta = f"{peso:.2f}" + (" (entregado)" if peso == BASE_PESO else "")
        print(
            f"{etiqueta:20s} {sum(f_a)/len(f_a):8.3f} {sum(n_a)/len(n_a):9.3f} "
            f"{sum(f_i)/len(f_i):10.3f} {sum(n_i)/len(n_i):12.3f}"
        )

    print(f"\ndeltas pareados contra el peso entregado ({BASE_PESO}), IC al 90%:")
    base = guardadas[BASE_PESO]
    for peso in tablas:
        if peso == BASE_PESO:
            continue
        print(f"\n  === peso {peso:.2f} ===")
        for etiqueta, i in (
            ("F1@3   50   ", 0),
            ("NDCG@10 50  ", 1),
            ("F1@3   indep", 2),
            ("NDCG@10 indep", 3),
            ("F1@3   humanas", 4),
            ("NDCG@10 humanas", 5),
        ):
            deltas = [x - y for x, y in zip(guardadas[peso][i], base[i])]
            media, bajo, alto = bootstrap_delta(deltas)
            print(f"  {etiqueta}: {media:+.3f}")
            print("  " + veredicto_bootstrap(media, bajo, alto).strip())


if __name__ == "__main__":
    main()
