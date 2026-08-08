#!/usr/bin/env python3
"""E09: normalizar los scores POR CONSULTA antes de sumarlos en la cascada.

HIPOTESIS, escrita antes de medir: la cascada suma cosenos crudos de tres
espacios distintos, y normalizar cada termino sobre el pool de la consulta
antes de sumarlos mejora el ranking.

JUSTIFICACION MECANICA. El docstring de `src/retrieval/rerank.py` afirma
explicitamente que sumar es legitimo porque "los dos terminos son cosenos, asi
que estan en la misma escala". Compartir el intervalo [-1,1] **no** es
compartir distribucion. La familia E5 se entrena con perdida contrastiva y
temperatura baja, y concentra sus cosenos en una franja alta y estrecha;
MiniLM los reparte mucho mas ancho. Sumar una variable de rango 0,20 con otra
de rango 0,50 hace que el termino ancho decida el orden sin importar el peso
nominal: el peso esta comprando VARIANZA, no autoridad.

Eso predice ademas algo ya observado. Que el peso "optimo" se moviera de 0,25
a 0,60 al ampliar el pool (E01) es exactamente lo que pasa cuando el parametro
compensa la escala de un pool cuya composicion cambio, y no una propiedad del
re-puntuador. Normalizar por consulta hace el peso interpretable.

RIESGOS Y MITIGACIONES, fijados ANTES de medir:
  - Criterio de siempre: IC al 90% del delta pareado excluyendo -0,02 en las
    DOS muestras, y ante empate se conserva la suma cruda, que es la entregada.
  - El z-score por consulta depende de la composicion del pool, o sea que
    introduce un acoplamiento entre `k_pool` y el re-puntuado que hoy no
    existe. Si gana, el resultado queda atado a `k_pool=100` y no se extrapola.
  - MITIGACION DE DIRECCION: si normalizar cambia la escala, el peso 0,60 deja
    de ser el optimo bajo la escala nueva. Comparar una variante normalizada
    mal calibrada contra una cruda bien calibrada seria un experimento
    tramposo, asi que **el peso se re-barre dentro de cada variante** y la
    comparacion se hace contra el mejor peso de cada una.
  - Si gana `minmax` (rango) y no `zscore` (varianza), eso apunta a outliers
    del pool y no a la distribucion, y se reporta como tal.

COSTE: una sola pasada de FAISS. Los tres cosenos de cada candidato se
calculan una vez y las nueve celdas salen de recombinarlos. Cero codificacion.

Uso:
    python dev/scripts/barrido_norm_e09.py
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

PROF = 200       # la profundidad entregada (E08 la confirmo como ganadora)
K_POOL = 100     # el pool entregado
AGG = "top5"     # E07
NORMAS = ("cruda", "minmax")
PESOS = (0.60, 1.00)
BASE = ("cruda", 0.60)   # la celda entregada

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")


def normalizar(v: np.ndarray, modo: str) -> np.ndarray:
    """Normaliza un vector de cosenos SOBRE EL POOL DE UNA CONSULTA.

    Los casos degenerados (pool de un solo candidato, o todos con el mismo
    score) devuelven ceros: sin dispersion no hay nada que normalizar y
    cualquier constante da el mismo orden.
    """
    if modo == "cruda":
        return v
    if modo == "zscore":
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)
    if modo == "minmax":
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)
    raise ValueError(modo)


def construir_pools(consultas, cache_idx):
    """Recupera PROF candidatos por consulta y guarda los TRES cosenos crudos.

    Guardarlos por separado es lo que hace barato el barrido: las nueve celdas
    son recombinaciones de estos mismos numeros.
    """
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for n, c in enumerate(consultas, 1):
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=PROF)[:PROF]
        cos = {"primario": np.array([h.score for h in hits], dtype="float64")}
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            cos[sec] = np.array(
                [float(np.dot(qv, idx_s.reconstruct(h.fila))) for h in hits], dtype="float64"
            )
        por_consulta[c["query_id"]] = (hits, cos)
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta


def dispersion(pools) -> None:
    """Imprime la dispersion de cada encoder. Es la premisa del experimento:
    si los tres tuvieran el mismo rango, no habria nada que normalizar y el
    resultado seria un empate por construccion."""
    print("\ndispersion de los cosenos dentro del pool (mediana sobre las 50 consultas):")
    print(f"  {'encoder':38s}{'rango':>9s}{'desv':>9s}{'media':>9s}")
    for nombre in ("primario", *SECUNDARIOS):
        rangos = [c[nombre].max() - c[nombre].min() for _, c in pools.values()]
        desvs = [c[nombre].std() for _, c in pools.values()]
        medias = [c[nombre].mean() for _, c in pools.values()]
        et = f"{nombre} ({MINILM})" if nombre == "primario" else nombre
        print(f"  {et:38.38s}{np.median(rangos):>9.3f}{np.median(desvs):>9.3f}{np.median(medias):>9.3f}")


def celda(pools, norma, peso):
    res = {}
    for qid, (hits, cos) in pools.items():
        total = normalizar(cos["primario"], norma).copy()
        for sec in SECUNDARIOS:
            total += peso * normalizar(cos[sec], norma)
        orden = np.argsort(-total, kind="stable")[:K_POOL]
        recorte = []
        for i, j in enumerate(orden, 1):
            h = hits[j]
            h.score = float(total[j])
            h.rank = i
            recorte.append(h)
        res[qid] = generador.build_result_object(qid, recorte, agg_strategy=AGG)
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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "norm_e09")
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

    print(f"\nrecuperando {PROF} candidatos por consulta y puntuando con los tres...", flush=True)
    pools = construir_pools(consultas, cache_idx)
    dispersion(pools)

    guardadas = {}
    for norma in NORMAS:
        for peso in PESOS:
            res = celda(pools, norma, peso)
            with (args.out_dir / f"{norma}_p{peso:.2f}.jsonl").open("w", encoding="utf-8") as f:
                for q in sorted(res):
                    f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
            guardadas[(norma, peso)] = (
                metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
            )
            print(f"  {norma} peso={peso:.2f} listo", flush=True)

    print(f"\n{'celda':20s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in guardadas:
        et = f"{k[0]}:{k[1]:.2f}" + (" *" if k == BASE else "")
        print(f"{et:20s}" + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la configuracion entregada\n")

    print("deltas pareados contra la entregada, IC al 90% (criterio: bajo > -0.02):")
    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        print(f"\n  === {k[0]} peso={k[1]:.2f} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, bajo, alto = bootstrap_delta(deltas)
            g = sum(1 for d in deltas if d > 1e-9)
            p = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {g}g/{p}p  {'pasa' if bajo > -0.02 else 'no pasa'}")


if __name__ == "__main__":
    main()
