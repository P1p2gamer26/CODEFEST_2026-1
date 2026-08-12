#!/usr/bin/env python3
"""E31: configuraciones de la cascada de CUATRO encoders (bge-m3 como tercer
re-puntuador), pre-registrado en dev/experimentos/cola.jsonl.

Cero FAISS y cero modelos: `pools_entregados.json --con-similitudes` ya trae
los 200 candidatos crudos del primario con el coseno de los CUATRO encoders.
El score entregado se reconstruye exacto:

    score = cos_minilm + 0.60*cos_gte + 0.60*cos_e5  -> top 100 -> agg top5

Celdas discretas (no grilla, leccion 2) y, para cada una, su CONTROL DE PESO:
la misma autoridad total repartida entre gte y e5, SIN bge. Sin ese control no
se distingue "el encoder aporta" de "mas peso al re-rank".

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_cascada_e31.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402

from barrido_cascada_e27_e28_e29 import (  # noqa: E402
    BASE, COLS, E5, GTE, K_POOL, MINILM, cargar, hits_de, metricas, resultado,
)
from eval_mini import bootstrap_delta, cargar_jsonl  # noqa: E402

BGE = "bge-m3"
CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

# nombre -> pesos por re-puntuador. La autoridad total va en el comentario:
# cada celda con bge se compara contra el control de su MISMA autoridad total.
CELDAS = {
    BASE:                    {GTE: 0.60, E5: 0.60},                # 1.20
    "gte+e5+bge @0.60":      {GTE: 0.60, E5: 0.60, BGE: 0.60},     # 1.80
    "ctrl gte+e5 @0.90":     {GTE: 0.90, E5: 0.90},                # 1.80
    "gte+bge @0.60":         {GTE: 0.60, BGE: 0.60},               # 1.20 -> ctrl = BASE
    "bge+e5 @0.60":          {E5: 0.60, BGE: 0.60},                # 1.20 -> ctrl = BASE
    "bge solo @0.60":        {BGE: 0.60},                          # 0.60
    "ctrl gte+e5 @0.30":     {GTE: 0.30, E5: 0.30},                # 0.60
}
# celda -> control con el que se la compara ademas de contra la base
CONTROL = {
    "gte+e5+bge @0.60": "ctrl gte+e5 @0.90",
    "gte+bge @0.60": BASE,
    "bge+e5 @0.60": BASE,
    "bge solo @0.60": "ctrl gte+e5 @0.30",
}


def pool(crudos, sims, pesos):
    """Suma las similitudes ponderadas sobre los 200 crudos y recorta a K_POOL.
    Un solo recorte, igual que generador.py (el orden es conmutativo)."""
    v = [(i, sims[MINILM][i] + sum(w * sims[e][i] for e, w in pesos.items()))
         for i in range(len(crudos))]
    v.sort(key=lambda t: -t[1])
    return v[:K_POOL]


def reporte(nombre, ref, med, res, ceros_ref):
    lineas = sum(1 for q in res[nombre] if res[nombre][q] != res[ref][q])
    print(f"  === {nombre}  vs  {ref} ===  {lineas}/50 lineas cambian")
    for j, col in enumerate(COLS):
        d = [x - y for x, y in zip(med[nombre][j], med[ref][j])]
        m, lo, hi = bootstrap_delta(d)
        g = sum(1 for x in d if x > 1e-9)
        p = sum(1 for x in d if x < -1e-9)
        print(f"  {col:9s}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g}g/{p}p  "
              f"{'pasa' if lo > -0.02 else 'NO pasa'}")
    ceros = sum(1 for v in med[nombre][0] if v == 0)
    print(f"  consultas con F1@3=0: {ceros} (ref: {ceros_ref})"
          f"{'  <-- VETO' if ceros > ceros_ref else ''}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "cascada_e31")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, crudos, sims, meta = cargar()
    assert config["peso"] == 0.60 and config["profundidad"] == 200 and config["k_pool"] == K_POOL
    for q in crudos:
        assert BGE in sims[q], f"{q} sin similitudes de {BGE}"

    consultas = {c["query_id"]: expandir_consulta(c["text"]) for c in cargar_jsonl(CONSULTAS)}
    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_ind = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_age = [g for g in gt_todo if g.get("anotador")]
    print(f"{len(gt_todo)} evaluables | {len(gt_ind)} indep | "
          f"{len(gt_hum)} humanas | {len(gt_age)} agente\n")

    res = {}
    for nombre, pesos in CELDAS.items():
        res[nombre] = {q: resultado(q, hits_de(pool(crudos[q], sims[q], pesos), crudos[q], meta),
                                    consultas[q])
                       for q in crudos}
    med = {k: metricas(v, gt_todo) + metricas(v, gt_ind) for k, v in res.items()}
    desg = {k: metricas(v, gt_hum) + metricas(v, gt_age) for k, v in res.items()}

    print(f"{'celda':22s}" + "".join(f"{c:>9s}" for c in COLS) + f"{'F1(41h)':>9s}{'F1(9a)':>9s}"
          f"{'ND(41h)':>9s}{'ND(9a)':>9s}")
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):22s}"
              + "".join(f"{sum(v) / len(v):>9.3f}" for v in med[k])
              + f"{sum(desg[k][0]) / len(desg[k][0]):>9.3f}"
              + f"{sum(desg[k][3]) / len(desg[k][3]):>9.3f}"
              + f"{sum(desg[k][1]) / len(desg[k][1]):>9.3f}"
              + f"{sum(desg[k][4]) / len(desg[k][4]):>9.3f}")
    print("\n  * = la entregada; tiene que dar 0.440 / 0.506 / 0.491 y "
          "0.400 / 0.436 / 0.429 (regla de E09)\n")

    ceros_base = sum(1 for v in med[BASE][0] if v == 0)
    for k in CELDAS:
        if k == BASE:
            continue
        reporte(k, BASE, med, res, ceros_base)
        ref = CONTROL.get(k)
        if ref and ref != BASE:
            reporte(k, ref, med, res, sum(1 for v in med[ref][0] if v == 0))

    for k, v in res.items():
        seguro = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in k)
        with (args.out_dir / f"{seguro}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(v):
                f.write(json.dumps(v[q], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
