#!/usr/bin/env python3
"""Simula la ground truth VERDADERA por Monte Carlo y estima el puntaje real.

POR QUE EXISTE. La ground truth propia es PARCIAL: ADL no publica la suya, y
un documento no anotado cuenta como irrelevante aunque quiza no lo sea. Por
eso el F1@3 y el NDCG@10 medidos son una COTA INFERIOR y los valores absolutos
se leen mal ("vamos en 0,44" cuando el numero real es probablemente mas alto).
Este script modela las anotaciones que faltan, simula miles de ground truths
plausibles, y reporta la distribucion de lo que el evaluador de verdad nos
daria: la esperanza con su intervalo al 90%.

QUE MODELO. Tres fuentes de anotaciones perdidas, separadas por procedencia
porque tienen mecanismos distintos (la leccion 2 de lecciones_metodologia.md
manda aqui):

1. **Documentos recuperados y nunca juzgados** (solo en las 10 consultas de
   anotacion INDEPENDIENTE, sin campo `pool`): sus candidatos salieron de un
   conteo de palabras clave, no del recuperador, asi que los documentos que el
   sistema entrega y nadie anoto son genuinamente desconocidos. Su
   probabilidad de ser relevantes se calibra EMPIRICAMENTE sobre el propio
   corpus: para cada bin de coseno del primario (MiniLM), la tasa observada de
   documentos relevantes en ese bin (la "curva de calibracion" de las 31
   consultas anotadas por pooling, donde cada documento del pool si se vio).

2. **Error de juicio del anotador** (en las 40 consultas de pooling): el pool
   de 200 candidatos se vio entero y los no marcados se rechazaron, asi que un
   documento recuperado no anotado solo es relevante si el anotador se
   equivoco. Eso es una tasa pequena (`eps_juicio`), no una curva.

3. **Documentos relevantes ocultos** (nunca recuperados ni anotados): no
   entran a ninguna lista, solo inflan el denominador del recall cuando la
   consulta tiene pocos relevantes anotados. Se modelan solo cuando |R| < 3,
   que es donde pueden mover el F1@3 (las 5 consultas de escalon 0,50 del
   plan).

La calibracion es CONSERVADORA: la curva usa la tasa de MARCADOS relevantes,
que a su vez subestima la tasa verdadera (una consulta de pooling pudo perder
relevantes sin marcarlos). El resultado es una cota inferior corregida, no un
piso inventado. Con `--factor-evidencia` se puede ver cuanto sube la
estimacion si la calibracion subestima.

LO QUE TAMBIEN SALE: los candidatos a re-anotar. Para cada documento
recuperado y no anotado, la simulacion da P(relevante) y el aumento de
F1@3/NDCG@10 si se confirma. Re-anotar a mano los de arriba es la prioridad 1
del plan maestro y el unico camino legitimo para SUBIR la metrica medida.

NO MODIFICA NADA. Solo lee resultados.jsonl, la ground truth y el indice.
El escenario de re-anotacion es un escenario, no un cambio al archivo.

Uso:
    python dev/scripts/simular_ground_truth.py
    python dev/scripts/simular_ground_truth.py --escenarios
    python dev/scripts/simular_ground_truth.py --p-ocultos 0.30 --eps-juicio 0.10
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (  # noqa: E402
    DEV_DIR,
    ENCODER_PRIMARY_NAME,
    RESULTADOS_PATH,
)
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402
from eval_mini import f1, ndcg, ndcg_penalizado, techo_f1  # noqa: E402

GT_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"

N_BINS = 10
K_POOL = 200
ITERACIONES = 2000

# Cache del pool (doc_id -> mejor coseno) por consulta. Construirlo tarda ~1
# minuto porque carga el indice; los barridos de parametros no deben pagarlo
# de nuevo. Se invalida solo si cambia el k_pool.
CACHE_POOL = DEV_DIR / "intermedios" / f"simulacion_pool_{K_POOL}.json"


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def construir_pool(consultas, k_pool=K_POOL):
    """Pool del primario (MiniLM) por consulta: hits con score coseno.

    La evidencia de relevancia de un documento es el MAXIMO coseno de sus
    chunks en el pool: es la senal en la que esta construido todo el sistema
    (el recall del pool es 93,2%, medido en E18) y la unica que no necesita
    cargar los dos re-puntuadores de la cascada.

    Se cachea en `intermedios/simulacion_pool_<k>.json` (doc_id -> coseno),
    porque el primer barrido de parametros paga la carga del indice y el
    resto no tiene por que.
    """
    if CACHE_POOL.is_file():
        print(f"evidencia: cache {CACHE_POOL.name} (borrar para reconstruir)")
        return {qid: dict(pd) for qid, pd in json.loads(CACHE_POOL.read_text(encoding="utf-8")).items()}

    encoder = get_encoder(name=ENCODER_PRIMARY_NAME)
    index, metadata = load_index(ENCODER_PRIMARY_NAME)
    print(f"evidencia: {encoder.name}, pool de {k_pool} candidatos por consulta (se cachea)")
    pool = {}
    for c in consultas:
        texto = expandir_consulta(c["text"])
        hits = search(texto, encoder, index, metadata, k=k_pool)
        pool[c["query_id"]] = mejor_coseno_por_documento(hits)
    CACHE_POOL.parent.mkdir(parents=True, exist_ok=True)
    CACHE_POOL.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {CACHE_POOL}")
    return pool


def mejor_coseno_por_documento(pool_hits) -> dict[str, float]:
    """doc_id -> maximo coseno de sus chunks en el pool."""
    mejor: dict[str, float] = {}
    for h in pool_hits:
        if h.doc_id not in mejor or h.score > mejor[h.doc_id]:
            mejor[h.doc_id] = h.score
    return mejor


def calibrar(pool_por_doc: dict[str, dict[str, float]], gt: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Curva empirica P(relevante | coseno) desde las consultas de pooling.

    Se usan SOLO las consultas anotadas por pooling humano (campo `pool`, sin
    `anotador`): para ellas el pool se vio entero, asi que "en el pool y no
    marcado" significa "visto y rechazado", y la tasa de marcados por bin de
    coseno es una estimacion de la probabilidad de relevancia condicional al
    coseno. Los bins son deciles de la distribucion de cosenos del pool.

    Devuelve (bordes, tasas): tasas[i] vale para cosenos en [bordes[i],
    bordes[i+1]).
    """
    etiquetados: list[tuple[float, bool]] = []
    for fila in gt:
        if fila.get("pool") and not fila.get("anotador"):
            relevantes = set(fila["docs_relevantes"])
            for doc_id, cos in pool_por_doc[fila["query_id"]].items():
                etiquetados.append((cos, doc_id in relevantes))
    etiquetados.sort()
    if not etiquetados:
        raise SystemExit("no hay consultas de pooling humano para calibrar la curva")

    cosenos = np.array([c for c, _ in etiquetados])
    bordes = np.quantile(cosenos, np.linspace(0, 1, N_BINS + 1))
    bordes[-1] += 1e-9  # el ultimo bin cierra el rango
    ids_bin = np.digitize(cosenos, bordes) - 1
    tasas = np.zeros(N_BINS)
    for i in range(N_BINS):
        sel = np.where(ids_bin == i)[0]
        if len(sel) == 0:
            continue
        rel = sum(1 for j in sel if etiquetados[int(j)][1])
        tasas[i] = rel / len(sel)
    return bordes, tasas


def tasa_de(bordes: np.ndarray, tasas: np.ndarray, cos: float) -> float:
    idx = int(np.digitize(cos, bordes)) - 1
    idx = min(max(idx, 0), N_BINS - 1)
    return float(tasas[idx])


def phantoms_por_consulta(
    gt: dict[str, dict],
    resultados: dict[str, dict],
    pool_por_doc,
    bordes: np.ndarray,
    tasas: np.ndarray,
    eps: float,
) -> dict[str, list[dict]]:
    """Los documentos recuperados y no anotados, con su P(relevante) y su peso.

    Para cada consulta, candidatos fantasma = documentos del top-3 y de los 10
    fragmentos que no estan en la ground truth. La probabilidad depende de la
    procedencia de la anotacion (ver docstring del modulo):

      - sin `pool` (10 independientes): tasa del bin de coseno, por evidencia;
      - con `pool` (31) o de agentes (9): error de juicio del anotador.

    Devuelve {query_id: [{"doc_id", "p", "top3": bool, "frags": [ranks]}]}.
    """
    mediana_pool = float(np.median([c for d in pool_por_doc.values() for c in d.values()]))
    out: dict[str, list[dict]] = {}
    for qid, fila in gt.items():
        relevantes = set(fila["docs_relevantes"])
        res = resultados.get(qid)
        if res is None:
            continue
        top3 = [d["doc_id"] for d in res.get("documents", [])][:3]
        frags = [f["doc_id"] for f in res.get("fragments", [])]
        independiente = not fila.get("pool")

        vistos: dict[str, dict] = {}
        for doc_id in top3 + frags:
            if doc_id in relevantes or doc_id in vistos:
                continue
            cos = pool_por_doc[qid].get(doc_id, mediana_pool)
            p = tasa_de(bordes, tasas, cos) if independiente else eps
            vistos[doc_id] = {
                "doc_id": doc_id,
                "cos": cos,
                "p": p,
                "top3": doc_id in top3,
                "frags": [i for i, f in enumerate(frags, 1) if f == doc_id],
            }
        out[qid] = list(vistos.values())
    return out


def _ndcg_desde_ganancias(ganancias, k: int = 10) -> float:
    """Mismo ideal fijo que eval_mini: los k cupos llenos de relevante."""
    import math

    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(ganancias, start=1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / idcg


def simulacion(
    gt: list[dict],
    resultados: dict[str, dict],
    phantoms: dict[str, list[dict]],
    iteraciones: int,
    semilla: int,
    p_ocultos: float,
    factor_evidencia: float,
) -> tuple[dict, list[dict]]:
    """Monte Carlo: para cada iteracion, arma una ground truth verdadera.

    Una ground truth simulada = la anotada + (con probabilidad p) cada fantasma
    + (con probabilidad p_ocultos, solo si |R| < 3) un relevante oculto que
    nunca se recupera. El oculto entra a la cuenta del recall (sube el
    denominador) pero no a ninguna lista, que es exactamente su efecto real.

    Devuelve (series, esperanzas) con los promedios por iteracion y, por
    consulta, la esperanza de F1@3 y NDCG@10.
    """
    rnd = random.Random(semilla)
    query_ids = [f["query_id"] for f in gt]
    n = len(query_ids)
    relevantes_base = {f["query_id"]: set(f["docs_relevantes"]) for f in gt}
    independientes = [f["query_id"] for f in gt if not f.get("pool")]

    # El factor de calidad de un fragmento NO cambia entre iteraciones: se
    # precomputa una vez para no pagar la regex de `calidad()` en cada paso
    # del Monte Carlo (era ~0,12 s por iteracion, 4 minutos al total).
    from src.retrieval.calidad_chunk import calidad

    precomputados = {}
    for qid in query_ids:
        res = resultados.get(qid)
        if res is None:
            continue
        frags = res.get("fragments", [])
        precomputados[qid] = {
            "docs": [d["doc_id"] for d in res["documents"][:3]],
            "frags": frags,
            "factores": [calidad(f.get("text", "")) for f in frags],
        }

    series_f1: list[float] = []
    series_ndcg: list[float] = []
    series_ndcg_pen: list[float] = []
    series_f1_10: list[float] = []
    series_ndcg_10: list[float] = []

    esperanza_idx = {qid: {"suma_f1": 0.0, "suma_ndcg": 0.0, "n": 0} for qid in query_ids}

    for _ in range(iteraciones):
        suma_f1 = suma_ndcg = suma_ndcg_pen = 0.0
        f1_10 = ndcg_10 = 0.0
        for qid in query_ids:
            pre = precomputados[qid]
            R = set(relevantes_base[qid])
            for ph in phantoms.get(qid, []):
                if rnd.random() < min(1.0, ph["p"] * factor_evidencia):
                    R.add(ph["doc_id"])
            if len(relevantes_base[qid]) < 3 and rnd.random() < p_ocultos:
                R.add(f"__oculto__:{qid}")
            v_f1 = f1(pre["docs"], R)[2]
            ganancias = [1.0 if f["doc_id"] in R else 0.0 for f in pre["frags"]]
            v_ndcg = _ndcg_desde_ganancias(ganancias)
            v_ndcg_pen = _ndcg_desde_ganancias(
                [g * c for g, c in zip(ganancias, pre["factores"])]
            )
            suma_f1 += v_f1
            suma_ndcg += v_ndcg
            suma_ndcg_pen += v_ndcg_pen
            if qid in independientes:
                f1_10 += v_f1
                ndcg_10 += v_ndcg
            e = esperanza_idx[qid]
            e["suma_f1"] += v_f1
            e["suma_ndcg"] += v_ndcg
            e["n"] += 1
        series_f1.append(suma_f1 / n)
        series_ndcg.append(suma_ndcg / n)
        series_ndcg_pen.append(suma_ndcg_pen / n)
        if independientes:
            m = len(independientes)
            series_f1_10.append(f1_10 / m)
            series_ndcg_10.append(ndcg_10 / m)

    resumen = {
        "f1": np.array(series_f1),
        "ndcg": np.array(series_ndcg),
        "ndcg_pen": np.array(series_ndcg_pen),
        "f1_10": np.array(series_f1_10),
        "ndcg_10": np.array(series_ndcg_10),
    }
    esperanzas = [
        {
            "query_id": qid,
            "f1_esperado": e["suma_f1"] / max(1, e["n"]),
            "ndcg_esperado": e["suma_ndcg"] / max(1, e["n"]),
        }
        for qid, e in esperanza_idx.items()
    ]
    return resumen, esperanzas


def ic90(serie: np.ndarray) -> tuple[float, float, float]:
    if serie.size == 0:
        return 0.0, 0.0, 0.0
    return float(serie.mean()), float(np.percentile(serie, 5)), float(np.percentile(serie, 95))


def medir(gt, resultados, metrica):
    """Media de una metrica sobre la ground truth dada (parcial o confirmada)."""
    total = 0.0
    m = 0
    for fila in gt:
        if not fila["docs_relevantes"]:
            continue
        res = resultados.get(fila["query_id"])
        if res is None:
            continue
        R = set(fila["docs_relevantes"])
        if metrica == "f1":
            total += f1([d["doc_id"] for d in res["documents"][:3]], R)[2]
        elif metrica == "ndcg":
            total += ndcg(res.get("fragments", []), R)
        else:
            total += ndcg_penalizado(res.get("fragments", []), R)
        m += 1
    return total / max(1, m)


def medir_subset(gt, resultados, metrica, subset):
    return medir([f for f in gt if f["query_id"] in subset], resultados, metrica)


def escenario_confirmar(gt, resultados, phantoms, top_n: int, factor_evidencia: float) -> None:
    """Que pasaria con las metricas MEDIDAS si se re-anotaran a mano y se
    confirmaran los candidatos mas probables. Es un escenario: no toca la
    ground truth real."""
    candidatos = [
        (qid, ph["doc_id"], ph["p"] * factor_evidencia)
        for qid, phs in phantoms.items()
        for ph in phs
        if ph["top3"] or ph["frags"]
    ]
    candidatos.sort(key=lambda t: -t[2])
    plausibles = [c for c in candidatos if c[2] >= 0.20]

    print("\nescenario de re-anotacion (confirmar a mano los candidatos mas probables):")
    print(f"  {'confirmados':>11s} {'F1@3':>7s} {'NDCG@10':>8s} {'NDCGp':>7s}")
    for n in [1, 3, 5, 10, len(plausibles)]:
        if n == 0:
            continue
        elegidos = plausibles[:n]
        gt2 = copy.deepcopy(gt)
        por2 = {f["query_id"]: f for f in gt2}
        for qid, doc_id, _ in elegidos:
            if doc_id not in por2[qid]["docs_relevantes"]:
                por2[qid]["docs_relevantes"].append(doc_id)
        print(
            f"  {n:>11d} {medir(gt2, resultados, 'f1'):7.3f} "
            f"{medir(gt2, resultados, 'ndcg'):8.3f} {medir(gt2, resultados, 'ndcg_pen'):7.3f}"
        )
    print(
        f"  ({len(plausibles)} candidatos con P(relevante) >= 0.20 en este modelo; "
        "confirmar es trabajo de anotacion, no un truco de medir)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resultados", type=Path, default=RESULTADOS_PATH)
    parser.add_argument("--ground-truth", type=Path, default=GT_PATH)
    parser.add_argument("--consultas", type=Path, default=CONSULTAS_PATH)
    parser.add_argument("--k-pool", type=int, default=K_POOL)
    parser.add_argument("--iteraciones", type=int, default=ITERACIONES)
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument(
        "--eps-juicio",
        type=float,
        default=0.05,
        help="Probabilidad de que un documento del pool VISTO y no marcado sea "
        "relevante de todos modos (error de juicio del anotador). Solo aplica a "
        "las consultas anotadas por pooling.",
    )
    parser.add_argument(
        "--p-ocultos",
        type=float,
        default=0.15,
        help="Probabilidad por consulta de que existan documentos relevantes nunca "
        "recuperados ni anotados (solo cuando |R| < 3, donde pueden bajar el F1@3).",
    )
    parser.add_argument(
        "--factor-evidencia",
        type=float,
        default=1.0,
        help="Multiplica la P(relevante) de los documentos no juzgados de las 10 "
        "consultas independientes. La calibracion subestima la tasa verdadera "
        "(mide marcados, no relevantes de verdad), asi que 1.0 es conservador.",
    )
    parser.add_argument(
        "--escenarios",
        action="store_true",
        help="Ademas del estimado, muestra el escenario de re-anotacion a mano.",
    )
    args = parser.parse_args()

    gt = cargar_jsonl(args.ground_truth)
    resultados = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    consultas = cargar_jsonl(args.consultas)

    gt_sim = [f for f in gt if f["docs_relevantes"]]
    pool_por_doc = construir_pool(consultas, k_pool=args.k_pool)

    bordes, tasas = calibrar(pool_por_doc, gt)
    phantoms = phantoms_por_consulta(
        {f["query_id"]: f for f in gt_sim},
        resultados,
        pool_por_doc,
        bordes,
        tasas,
        args.eps_juicio,
    )
    resumen, esperanzas = simulacion(
        gt_sim,
        resultados,
        phantoms,
        iteraciones=args.iteraciones,
        semilla=args.semilla,
        p_ocultos=args.p_ocultos,
        factor_evidencia=args.factor_evidencia,
    )

    indep = [f["query_id"] for f in gt_sim if not f.get("pool")]
    m_f1 = medir(gt_sim, resultados, "f1")
    m_ndcg = medir(gt_sim, resultados, "ndcg")
    m_pen = medir(gt_sim, resultados, "ndcg_pen")

    print("\nsimulacion de la ground truth verdadera (Monte Carlo)")
    print("=" * 60)
    print(
        f"modelo: {sum(len(v) for v in phantoms.values())} documentos recuperados-no-anotados; "
        f"eps_juicio={args.eps_juicio:.2f}, p_ocultos={args.p_ocultos:.2f}, "
        f"factor_evidencia={args.factor_evidencia:.2f}\n"
    )

    print("medido (ground truth parcial -- cota inferior):")
    print(f"  F1@3     {m_f1:.3f}")
    print(f"  NDCG@10  {m_ndcg:.3f}")
    print(f"  NDCG@10p {m_pen:.3f}")
    print(f"  techo    {techo_f1(gt_sim):.3f}\n")

    mf, lo_f, hi_f = ic90(resumen["f1"])
    mn, lo_n, hi_n = ic90(resumen["ndcg"])
    mp, lo_p, hi_p = ic90(resumen["ndcg_pen"])

    print("simulado (ground truth verdadera estimada -- esperanza [IC al 90%]):")
    print(f"  F1@3     {mf:.3f}  [{lo_f:.3f}, {hi_f:.3f}]")
    print(f"  NDCG@10  {mn:.3f}  [{lo_n:.3f}, {hi_n:.3f}]")
    print(f"  NDCG@10p {mp:.3f}  [{lo_p:.3f}, {hi_p:.3f}]\n")

    if indep:
        mf10, lo_f10, hi_f10 = ic90(resumen["f1_10"])
        mn10, lo_n10, hi_n10 = ic90(resumen["ndcg_10"])
        print("sobre las {} consultas independientes (el numero honesto):".format(len(indep)))
        print(f"  F1@3     {medir_subset(gt_sim, resultados, 'f1', indep):.3f} -> {mf10:.3f}  [{lo_f10:.3f}, {hi_f10:.3f}]")
        print(f"  NDCG@10  {medir_subset(gt_sim, resultados, 'ndcg', indep):.3f} -> {mn10:.3f}  [{lo_n10:.3f}, {hi_n10:.3f}]\n")

    print("como leerlo:")
    print(f"  P(F1@3 verdadero > {m_f1:.3f}): {(resumen['f1'] > m_f1).mean():.2f}")
    print(f"  P(NDCG@10 verdadero > {m_ndcg:.3f}): {(resumen['ndcg'] > m_ndcg).mean():.2f}")
    print(
        "  La estimacion es conservadora: la curva calibra sobre MARCADOS, asi que\n"
        "  el verdadero suele estar por encima del extremo alto del intervalo.\n"
    )

    tabla = sorted(
        (
            (qid, ph["doc_id"], ph["cos"], ph["p"], ph["top3"], len(ph["frags"]))
            for qid, phs in phantoms.items()
            for ph in phs
            if ph["top3"] or ph["frags"]
        ),
        key=lambda t: (-t[3], -t[2]),
    )
    print("candidatos a re-anotar a mano (no confirmados; por P(relevante)):")
    print(f"  {'consulta':>8s} {'doc_id':<18s} {'coseno':>6s} {'P(rel)':>6s} {'top3':>4s} {'frags':>5s}")
    for qid, doc_id, cos, p, top3, nf in tabla[:20]:
        print(f"  {qid:>8s} {doc_id:<18s} {cos:6.3f} {p:6.2f} {'si' if top3 else 'no':>4s} {nf:5d}")
    print(
        f"\n  {len(tabla)} candidatos en total; confirmar a mano los de arriba es el camino\n"
        "  legitimo para subir la metrica MEDIDA (no solo la estimada)."
    )

    if args.escenarios:
        escenario_confirmar(gt_sim, resultados, phantoms, top_n=20, factor_evidencia=args.factor_evidencia)


if __name__ == "__main__":
    main()
