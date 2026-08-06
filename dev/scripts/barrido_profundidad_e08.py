#!/usr/bin/env python3
"""E08: re-abrir `rerank_depth` x `k_pool` bajo el regimen del peso 0.60.

HIPOTESIS, escrita antes de medir: al triplicar el peso del re-puntuador
(0.25 -> 0.60, E01) los candidatos PROFUNDOS pasan a poder subir al top-3, y
por tanto `rerank_depth` deja de ser inerte.

POR QUE SE RE-ABRE ALGO YA MEDIDO. `rerank-depth` 400 y 600 dieron **51
empates de 51** y la nota de las notas del proyecto explica por que con una razon
mecanica explicita: "con peso 0.25 el re-puntuador solo reordena, y los
candidatos de las posiciones 200-600 tienen scores demasiado bajos para
subir al top-3". Esa razon **depende del peso**, y el peso ya no es 0.25.
La aritmetica: para que un candidato en la posicion 500 alcance al que esta
en el top-3 hace falta que el re-puntuador aporte mas que la brecha del
primario; con peso 0.25 el re-puntuador contribuye a lo sumo 0.25 y con 0.60
contribuye hasta 0.60 -- 2.4x mas autoridad para reordenar. Es exactamente la
firma que hizo que E01 y E07 valieran la pena, y la leccion 5 (las notas
propias envejecen, incluidas las mecanicas).

`k_pool` entra en el mismo barrido porque el 100 se fijo en la ronda del 4 de
agosto con **peso 0.25 y el glosario sin las tres entradas de E03**, o sea
bajo un regimen que ya no existe. Y son el mismo parametro visto dos veces:
`rerank_depth` decide cuantos candidatos se re-puntuan y `k_pool` cuantos
sobreviven a la agregacion, asi que barrerlos por separado no puede ver su
interaccion.

RIESGO DECLARADO, con las mitigaciones fijadas ANTES de medir:
  - Es una grilla de 16 celdas contra 50 consultas, o sea la maquina de
    sobreajuste de la leccion 2 a plena potencia. Se adopta solo si el IC al
    90% del delta pareado excluye una perdida de 0,02 en las DOS muestras
    (50 y 10 independientes), y **ante empate se conserva la celda entregada**
    (200:100).
  - La hipotesis predice una DIRECCION: si algo gana, tiene que ser una
    profundidad MAYOR que 200. Si el ganador fuera una profundidad menor, eso
    refuta la explicacion mecanica y el resultado se trata como ruido.
  - E07 ya mostro que `top8` y `sum` son identicos con pool 100, o sea que
    ningun documento aporta mas de 8 chunks. Al ampliar el pool esa cota se
    afloja y `top5` podria volver a quedarse corto: por eso se reporta
    tambien `top8` en las celdas de pool ancho.

COSTE: una sola pasada de FAISS. Los pools se construyen UNA vez a la
profundidad maxima y las celdas menores salen por rebanado -- el re-puntuado
es independiente por candidato, asi que los primeros 200 de la lista de 1000
tienen exactamente los mismos scores que una corrida a profundidad 200.
Cero codificacion nueva.

Uso:
    python dev/scripts/barrido_profundidad_e08.py
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

PESO = 0.60          # E01, fijo en este barrido
AGG = "top5"         # E07, fijo
PROF_MAX = 1000      # de aca sale todo por rebanado
PROFUNDIDADES = (200, 400, 600, 1000)
POOLS = (60, 100, 150, 200)
BASE = (200, 100)    # la celda entregada

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")


def construir_pools_max(consultas, cache_idx):
    """Re-puntua PROF_MAX candidatos por consulta, una sola vez.

    Devuelve la lista completa ya re-puntuada pero SIN ordenar ni recortar:
    cada celda del barrido ordena y recorta su propia rebanada.
    """
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for n, c in enumerate(consultas, 1):
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=PROF_MAX)[:PROF_MAX]
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


def celda(pools, prof, k_pool, agg):
    """Ordena y recorta la rebanada de profundidad `prof`, luego agrega."""
    res = {}
    for qid, hits in pools.items():
        recorte = sorted(hits[:prof], key=lambda h: -h.score)[:k_pool]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        res[qid] = generador.build_result_object(qid, recorte, agg_strategy=agg)
    return res


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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "prof_e08")
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

    print(f"\nre-puntuando {PROF_MAX} candidatos por consulta (peso={PESO})...", flush=True)
    pools = construir_pools_max(consultas, cache_idx)

    guardadas = {}
    for prof in PROFUNDIDADES:
        for k_pool in POOLS:
            if k_pool > prof:
                continue  # recortar a mas de lo que se re-puntuo no significa nada
            res = celda(pools, prof, k_pool, AGG)
            with (args.out_dir / f"d{prof}_k{k_pool}.jsonl").open("w", encoding="utf-8") as f:
                for q in sorted(res):
                    f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
            guardadas[(prof, k_pool)] = (
                metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
            )
            print(f"  d={prof} k={k_pool} listo", flush=True)

    # top8 en los pools anchos: E07 mostro que el tope de top5 se afloja al ampliar
    for prof, k_pool in ((1000, 200), (600, 200), (1000, 150)):
        if (prof, k_pool) not in guardadas:
            continue
        res = celda(pools, prof, k_pool, "top8")
        guardadas[(prof, k_pool, "top8")] = (
            metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        )

    print(f"\n{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in guardadas:
        et = f"d{k[0]}:k{k[1]}" + (f":{k[2]}" if len(k) > 2 else "")
        et += " *" if k == BASE else ""
        print(f"{et:16s}" + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada (top5 salvo donde diga otra cosa)\n")

    print(f"deltas pareados contra la entregada d{BASE[0]}:k{BASE[1]}, IC al 90% (criterio: bajo > -0.02):")
    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        et = f"d{k[0]}:k{k[1]}" + (f":{k[2]}" if len(k) > 2 else "")
        print(f"\n  === {et} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, bajo, alto = bootstrap_delta(deltas)
            g = sum(1 for d in deltas if d > 1e-9)
            p = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {g}g/{p}p  {'pasa' if bajo > -0.02 else 'no pasa'}")


if __name__ == "__main__":
    main()
