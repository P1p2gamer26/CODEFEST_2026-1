#!/usr/bin/env python3
"""RE-MEDICION: bge-m3 como tercer re-puntuador, ahora SOBRE E32 (10 ago 2026).

HISTORIAL. E25 y E31 midieron bge-m3 contra la config que se entregaba ese
dia (F1@3 0.440 / NDCG@10 0.506 / NDCGp 0.491) y el eje "otro encoder" quedo
cerrado. Desde entonces se adopto E32 (post-filtrado por fenomeno dominante,
umbral 0.8), que subio a 0.455 / 0.516 / 0.499, y E22/E23 ya estaban dentro.
Este barrido re-mide las mismas celdas de E31 contra EL SISTEMA ACTUAL: el
filtro de E32 es la diferencia con la medicion anterior.

PRE-REGISTRO, fijado antes de medir:
  - HIPOTESIS nula: bge no aporta nada por encima del efecto de darle mas peso
    total al re-rank. Por eso cada celda con bge se compara contra su CONTROL
    de la MISMA autoridad total sin bge, ademas de contra la entregada.
  - Criterio de adopcion: el IC al 90% del delta pareado excluye una perdida
    de 0.02 en las DOS muestras (50 y 10 independientes) Y el conteo de
    consultas con F1@3=0 no sube de 11 (el veto de E32). Ante empate se
    conserva la entregada.
  - Si gana habria que construir el indice completo de bge (5,5 h de GPU ya
    hechas: `dev/intermedios/bge/encoder_bge-m3/`), sumar ~527 MB a la
    entrega y una dependencia mas para el evaluador. Si no confirma, el eje
    queda cerrado por cuarta vez (E04, E25, E31).

Sin FAISS y sin modelos: `pools_entregados.json --con-similitudes` ya trae los
200 candidatos crudos del primario con el coseno de los CUATRO encoders, y
`meta_crudos.json` la metadata. El camino entregado se reconstruye exacto:

    score = cos_minilm + 0.60*cos_gte + 0.60*cos_e5  -> top 100
          -> filtrar_por_fenomeno_dominante(0.8)     -> agg top5
          -> ordenar_para_fragmentos (E22/E23)       -> <=250 palabras

QUE NO SE PRUEBA: la representacion **sparse** de bge-m3 (es BM25 con otro
nombre y el hibrido lexico perdio dos veces), y el cross-encoder de E39 (otro
experimento, ya medido aparte).

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_bge_sobre_e32.py
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
    cargar, hits_de, K_POOL, MINILM, GTE, E5,
)
from barrido_fenomeno_topm_e32_e33 import (  # noqa: E402
    metricas, resultado,
)
from eval_mini import bootstrap_delta, cargar_jsonl  # noqa: E402

BGE = "bge-m3"
CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

BASE = "entregada"
UMBRAL_FENOMENO = 0.8   # E32, el entregado
AGG = "top5"            # E07

# nombre -> pesos por re-puntuador. La autoridad total va en el comentario.
CELDAS = {
    BASE:                    {GTE: 0.60, E5: 0.60},                # 1.20
    "gte+e5+bge @0.60":      {GTE: 0.60, E5: 0.60, BGE: 0.60},     # 1.80
    "ctrl gte+e5 @0.90":     {GTE: 0.90, E5: 0.90},                # 1.80
    "gte+e5+bge @0.40":      {GTE: 0.40, E5: 0.40, BGE: 0.40},     # 1.20
    "gte+bge @0.60":         {GTE: 0.60, BGE: 0.60},               # 1.20
    "bge+e5 @0.60":          {E5: 0.60, BGE: 0.60},                # 1.20
    "bge solo @0.60":        {BGE: 0.60},                          # 0.60
    "ctrl gte+e5 @0.30":     {GTE: 0.30, E5: 0.30},                # 0.60
}
# celda -> control contra el que se la compara ademas de contra la base
CONTROL = {
    "gte+e5+bge @0.60": "ctrl gte+e5 @0.90",
    "gte+e5+bge @0.40": BASE,
    "gte+bge @0.60": BASE,
    "bge+e5 @0.60": BASE,
    "bge solo @0.60": "ctrl gte+e5 @0.30",
}

# valores esperados de la entregada E32 (regla de E09). Si no dan, algo del
# camino reconstruido no es el entregado y NO se puede medir sobre E32.
ESPERADO_50 = (0.455, 0.516, 0.499)
ESPERADO_IND = (0.433, 0.474, 0.467)
ESPERADO_HUM = (0.486, 0.537, 0.520)
CERO_VETO = 11

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)", "F1(age)")


def pool(crudos, sims, pesos):
    """Suma las similitudes ponderadas sobre los 200 crudos y recorta a
    K_POOL. Un solo recorte, igual que generador.py (orden conmutativo)."""
    v = [(i, sims[MINILM][i] + sum(w * sims[e][i] for e, w in pesos.items()))
         for i in range(len(crudos))]
    v.sort(key=lambda t: -t[1])
    return v[:K_POOL]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "bge_sobre_e32")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, crudos, sims, meta = cargar()
    assert config["peso"] == 0.60 and config["profundidad"] == 200 and config["k_pool"] == K_POOL
    for q in crudos:
        assert BGE in sims[q], f"{q} sin similitudes de {BGE}"
    print("config del volcado:", config)

    consultas = {c["query_id"]: expandir_consulta(c["text"]) for c in cargar_jsonl(CONSULTAS)}
    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_ind = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_age = [g for g in gt_todo if g.get("anotador")]
    print(f"{len(gt_todo)} evaluables | {len(gt_hum)} humanas | {len(gt_age)} agente | "
          f"{len(gt_ind)} independientes\n")

    res = {}
    for nombre, pesos in CELDAS.items():
        res[nombre] = {
            q: resultado(q, hits_de(pool(crudos[q], sims[q], pesos), crudos[q], meta),
                         consultas[q], agg=AGG, umbral=UMBRAL_FENOMENO)
            for q in crudos
        }

    def col(v):
        return sum(v) / len(v) if v else float("nan")

    med = {}
    for nombre in CELDAS:
        med[nombre] = (metricas(res[nombre], gt_todo) + metricas(res[nombre], gt_ind)
                       + metricas(res[nombre], gt_hum))
        med[nombre].append(metricas(res[nombre], gt_age)[0])

    print(f"{'celda':22s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):22s}"
              + "".join(f"{col(v):>9.3f}" for v in med[k]))

    base50, base_ind, base_hum = (col(med[BASE][0]), col(med[BASE][1]), col(med[BASE][2])), \
        (col(med[BASE][3]), col(med[BASE][4]), col(med[BASE][5])), \
        (col(med[BASE][6]), col(med[BASE][7]), col(med[BASE][8]))
    print("\n  regla de E09 (la entregada tiene que dar):")
    for nombre, got, esp in (("50", base50, ESPERADO_50),
                             ("ind", base_ind, ESPERADO_IND),
                             ("hum", base_hum, ESPERADO_HUM)):
        ok = all(abs(g - e) < 1e-3 for g, e in zip(got, esp))
        print(f"    {nombre:3s} {tuple(round(g, 3) for g in got)}  esperado "
              f"{tuple(esp)}  {'OK' if ok else '<<< NO COINCIDE'}")

    ceros_base = sum(1 for v in med[BASE][0] if v == 0)
    print(f"\n  consultas con F1@3=0 en la entregada: {ceros_base} (veto: {CERO_VETO})\n")

    orden_q = [g["query_id"] for g in gt_todo]

    def reporte(nombre, ref, med_ref):
        lineas = sum(1 for q in res[nombre] if res[nombre][q] != res[ref][q])
        print(f"  === {nombre}  vs  {ref} ===  {lineas}/50 lineas cambian")
        for j, c in enumerate(COLS):
            d = [x - y for x, y in zip(med[nombre][j], med_ref[j])]
            m, lo, hi = bootstrap_delta(d)
            g = sum(1 for x in d if x > 1e-9)
            p = sum(1 for x in d if x < -1e-9)
            print(f"  {c:9s}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g}g/{p}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in med[nombre][0] if v == 0)
        print(f"  consultas con F1@3=0: {ceros} (entregada: {ceros_base})"
              f"{'  <-- VETO' if ceros > ceros_base else ''}")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, med[nombre][0], med_ref[0])}
        gana = [q for q in orden_q if d_f1[q] > 1e-9]
        pierde = [q for q in orden_q if d_f1[q] < -1e-9]
        print(f"  F1 gana {len(gana)}: {gana}")
        print(f"  F1 pierde {len(pierde)}: {pierde}\n")

    for k in CELDAS:
        if k == BASE:
            continue
        reporte(k, BASE, med[BASE])
        ref = CONTROL.get(k)
        if ref and ref != BASE:
            reporte(k, ref, med[ref])

    for k, v in res.items():
        seguro = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in k)
        with (args.out_dir / f"{seguro}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(v):
                f.write(json.dumps(v[q], ensure_ascii=False) + "\n")
    print(f"resultados por celda -> {args.out_dir}")


if __name__ == "__main__":
    main()
