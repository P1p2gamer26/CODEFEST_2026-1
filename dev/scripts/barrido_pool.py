#!/usr/bin/env python3
"""Barrido de ANCHO DE POOL sobre la cascada triple ya entregada.

Por que este barrido y no el de barrido_retrieval.py
----------------------------------------------------
`barrido_retrieval.py` mide k_pool x estrategia con UN encoder y solo F1@3.
Este mide lo mismo sobre la configuracion que de verdad se entrega -- MiniLM
primario, gte y e5 re-puntuando con peso 0,25 -- y reporta ademas NDCG@10.

Lo que se prueba es lo unico que el propio codigo declara sin medir.
`Entrega/generador.py` recorta a `k_pool` DESPUES del re-rank, y su comentario
lo dice con todas las letras: eso hace que la profundidad extra sirva para
reordenar y no para ampliar el pool, "que es un cambio distinto y no medido".
La cascada trae 200 candidatos y se tiran 140 antes de agregar a documento.

JUSTIFICACION MECANICA, escrita antes de ver los numeros: `diagnostico_ceros.py`
mostro que 15 de las 17 consultas en cero tienen el documento correcto DENTRO
del pool -- para esas, ampliar no puede ayudar. Pero las dos que fallan por
pool (q001, donde el corpus dice CBRN y la consulta NBQR; q038, con los
informes de MAPP/OEA) no lo tienen, y son exactamente las que un pool mas
ancho podria rescatar.

RIESGO CONOCIDO, tambien declarado antes: `sum` no tiene tope, asi que un pool
mas grande favorece a los documentos con muchos chunks. Por eso la grilla cruza
con `top5`, que acota esa ventaja.

REGLA DE DECISION FIJADA ANTES DE MEDIR (docs/lecciones_metodologia.md): se
adopta si el IC al 90% del delta pareado excluye una perdida de 0,02, con el
F1@3 decidiendo (el cambio toca el ranking de DOCUMENTOS) y el NDCG@10 como
segunda lectura. Guardia: si gana en las 50 y se derrumba en las 10
independientes, es sesgo de pooling y no se adopta. La grilla es de SEIS
celdas a proposito -- elegir el maximo de una grilla grande es el sobreajuste
que ya costo una decision equivocada en este proyecto.

Coste: cero codificacion nueva. Los tres indices ya existen y los vectores de
los candidatos se leen con `reconstruct(fila)`.

Uso:
    python dev/scripts/barrido_pool.py
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

# La cascada entregada, tal cual: primario y sus dos re-puntuadores.
PRIMARIO = MINILM
SECUNDARIOS = [(GTE, 0.25), (E5, 0.25)]
RERANK_DEPTH = 200

K_POOLS = [60, 100, 200]
ESTRATEGIAS = ["sum", "top5"]
BASE = (60, "sum")  # la configuracion entregada


def candidatos_por_consulta(consultas, cache_idx):
    """Los 200 candidatos re-puntuados de cada consulta, una sola vez.

    El ancho del pool y la estrategia de agregacion se aplican DESPUES, sobre
    esta misma lista: ninguna celda de la grilla vuelve a tocar FAISS.
    """
    enc_p = get_encoder(name=PRIMARIO)
    idx_p, metadata = cache_idx[PRIMARIO]
    por_consulta = {}
    for c in consultas:
        hits = search(c["text"], enc_p, idx_p, metadata, k=RERANK_DEPTH)
        hits = hits[:RERANK_DEPTH]
        for sec, peso in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(c["text"])
            for h in hits:
                h.score += peso * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        por_consulta[c["query_id"]] = hits
    return por_consulta


def evaluar_celda(por_consulta, k_pool, estrategia):
    resultados = {}
    for qid, hits in por_consulta.items():
        recorte = hits[:k_pool]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        resultados[qid] = generador.build_result_object(
            qid, recorte, agg_strategy=estrategia
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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "pool")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Una consulta con `docs_relevantes` vacio no es evaluable: se miro candidato
    # por candidato y ninguno respondia. Excluirla del promedio, no del archivo.
    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_indep = [f for f in gt_todo if not f.get("pool") and not f.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} consultas evaluables, {len(gt_indep)} independientes\n")

    # Los tres indices comparten metadata byte-identica: se carga una vez.
    cache_idx = {}
    metadata = None
    for nombre in (PRIMARIO, *(s for s, _ in SECUNDARIOS)):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores")

    print("\nrecuperando y re-puntuando una sola vez...", flush=True)
    por_consulta = candidatos_por_consulta(consultas, cache_idx)

    tablas = {}
    for k_pool in K_POOLS:
        for estrategia in ESTRATEGIAS:
            res = evaluar_celda(por_consulta, k_pool, estrategia)
            tablas[(k_pool, estrategia)] = res
            salida = args.out_dir / f"pool{k_pool}_{estrategia}.jsonl"
            with salida.open("w", encoding="utf-8") as f:
                for q in sorted(res):
                    f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
            print(f"  escrito {salida.name}", flush=True)

    print(f"\n{'celda':16s} {'F1(todas)':>10s} {'NDCG':>7s} {'F1(indep)':>10s} {'NDCG':>7s}")
    guardadas = {}
    for celda, res in tablas.items():
        f_a, n_a = metricas(res, gt_todo)
        f_i, n_i = metricas(res, gt_indep)
        guardadas[celda] = (f_a, n_a, f_i, n_i)
        etiqueta = f"{celda[0]}:{celda[1]}" + (" (entregada)" if celda == BASE else "")
        print(
            f"{etiqueta:16s} {sum(f_a)/len(f_a):10.3f} {sum(n_a)/len(n_a):7.3f} "
            f"{sum(f_i)/len(f_i):10.3f} {sum(n_i)/len(n_i):7.3f}"
        )

    print(f"\ndeltas pareados contra la entregada ({BASE[0]}:{BASE[1]}), IC al 90%:")
    base = guardadas[BASE]
    for celda in tablas:
        if celda == BASE:
            continue
        print(f"\n  === {celda[0]}:{celda[1]} ===")
        for etiqueta, i in (
            ("F1@3   todas", 0),
            ("NDCG@10 todas", 1),
            ("F1@3   indep", 2),
            ("NDCG@10 indep", 3),
        ):
            deltas = [x - y for x, y in zip(guardadas[celda][i], base[i])]
            media, bajo, alto = bootstrap_delta(deltas)
            print(f"  {etiqueta}: {media:+.3f}")
            print("  " + veredicto_bootstrap(media, bajo, alto).strip())


if __name__ == "__main__":
    main()
