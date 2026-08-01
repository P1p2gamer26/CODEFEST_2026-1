#!/usr/bin/env python3
"""F1@3 aproximado contra un mini ground truth anotado a mano.

ADL no publica el ground truth, asi que sin esto las decisiones de diseno
(un encoder o dos, fusion RRF si o no, grafo si o no) se toman a ojo. El
ground truth propio cubre hoy 41 de las 50 consultas; el valor absoluto NO
es la nota oficial (la anotacion es parcial: los documentos no anotados
cuentan como irrelevantes aunque quiza no lo sean, asi que el F1 real sera
mayor o igual que este).

**El promedio no decide.** Con ~40 consultas cada una pesa 0.025, asi que
dos que cambien de lado por azar mueven la media mas que un efecto real.
Para elegir entre dos configuraciones usar --comparar-con, que cuenta en
cuantas consultas gana cada una y aplica una prueba de signos.

NDCG@10 de fragmentos queda fuera a proposito: exigiria anotar relevancia
graduada fragmento por fragmento, y para elegir entre configuraciones el
F1@3 a nivel documento ya ordena igual.

Formato del ground truth (una linea por consulta):
    {"query_id": "q020", "docs_relevantes": ["F2-SWF-012", "F2-SWF-031"]}

Uso:
    python scripts/eval_mini.py
    python scripts/eval_mini.py --resultados intermedios/resultados_e5.jsonl
"""

import argparse
import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, N_DOCUMENTS_PER_QUERY, RESULTADOS_PATH  # noqa: E402

GROUND_TRUTH_MINI_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def f1(recuperados: list[str], relevantes: set[str]) -> tuple[float, float, float]:
    """Precision, recall y F1 de una consulta segun la sec. 10.2.1 del PDF.

    El denominador del recall es min(|D*|, 3), NO |D*|: la especificacion lo
    limita asi para no penalizar los casos con mas documentos relevantes que
    cupos disponibles. Dividir por |D*| a secas subestima el recall y puede
    ordenar mal dos configuraciones que se comparan entre si.
    """
    aciertos = len(set(recuperados) & relevantes)
    if aciertos == 0:
        return 0.0, 0.0, 0.0
    p = aciertos / len(recuperados)
    r = aciertos / min(len(relevantes), len(recuperados))
    return p, r, 2 * p * r / (p + r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultados", type=Path, default=RESULTADOS_PATH)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_MINI_PATH)
    parser.add_argument("--k", type=int, default=N_DOCUMENTS_PER_QUERY)
    parser.add_argument(
        "--sin-pooling",
        action="store_true",
        help="Usa solo las consultas anotadas de forma independiente (sin campo 'pool'). "
        "OBLIGATORIO al comparar dos encoders entre si: una consulta anotada sobre los "
        "candidatos que propuso el encoder X favorece a X, porque los documentos que X "
        "nunca recupero jamas pudieron marcarse como relevantes.",
    )
    parser.add_argument(
        "--comparar-con",
        type=Path,
        default=None,
        help="Otro resultados.jsonl. En vez de solo promediar, cuenta en cuantas "
        "consultas gana cada uno. Es la forma correcta de decidir entre dos "
        "configuraciones: con ~30 consultas cada una pesa 0.03 en la media, asi que "
        "dos que cambien de lado por azar mueven el promedio mas que un efecto real.",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth} -- hay que anotarlo a mano primero")
        sys.exit(2)

    resultados = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    gt = cargar_jsonl(args.ground_truth)

    if args.sin_pooling:
        antes = len(gt)
        gt = [f for f in gt if not f.get("pool")]
        print(f"solo anotacion independiente: {len(gt)} de {antes} consultas\n")

    suma = 0.0
    print(f"{'consulta':10s} {'P':>6s} {'R':>6s} {'F1':>6s}  aciertos")
    for fila in gt:
        qid = fila["query_id"]
        relevantes = set(fila["docs_relevantes"])
        res = resultados.get(qid)
        if res is None:
            print(f"{qid:10s} {'--':>6s} {'--':>6s} {'--':>6s}  (sin resultado)")
            continue
        docs = [d["doc_id"] for d in res["documents"][: args.k]]
        p, r, valor = f1(docs, relevantes)
        suma += valor
        print(f"{qid:10s} {p:6.2f} {r:6.2f} {valor:6.2f}  {sorted(set(docs) & relevantes)}")

    print(f"\nF1@{args.k} promedio sobre {len(gt)} consultas: {suma / len(gt):.3f}")

    if args.comparar_con:
        otros = {r["query_id"]: r for r in cargar_jsonl(args.comparar_con)}
        a = args.resultados.name
        b = args.comparar_con.name
        gana_a = gana_b = empate = 0
        suma_b = 0.0
        for fila in gt:
            qid = fila["query_id"]
            relevantes = set(fila["docs_relevantes"])
            if qid not in resultados or qid not in otros:
                continue
            va = f1([d["doc_id"] for d in resultados[qid]["documents"][: args.k]], relevantes)[2]
            vb = f1([d["doc_id"] for d in otros[qid]["documents"][: args.k]], relevantes)[2]
            suma_b += vb
            if abs(va - vb) < 1e-9:
                empate += 1
            elif va > vb:
                gana_a += 1
            else:
                gana_b += 1
        print(f"\n{b}: F1@{args.k} promedio {suma_b / len(gt):.3f}")
        print(f"\n{a} gana en {gana_a}, {b} gana en {gana_b}, empatan {empate}")
        difieren = gana_a + gana_b
        if difieren == 0:
            print("  -> las dos configuraciones dan exactamente lo mismo")
            return
        # Prueba de signos: bajo la hipotesis de que las dos configuraciones son
        # equivalentes, cada consulta que difiere es una moneda al aire. p es la
        # probabilidad de ver un reparto al menos tan desigual por puro azar.
        ganador, mayor = (a, gana_a) if gana_a > gana_b else (b, gana_b)
        p = 2 * sum(math.comb(difieren, i) for i in range(mayor, difieren + 1)) / 2**difieren
        p = min(1.0, p)
        print(f"  prueba de signos sobre las {difieren} que difieren: p = {p:.3f}")
        if p > 0.05:
            print(
                f"  -> NO concluyente. Entregar la configuracion mas simple, y no "
                f"cambiarla apoyandose en la diferencia de promedios."
            )
        else:
            print(f"  -> {ganador} gana de forma consistente, no solo en promedio")


if __name__ == "__main__":
    main()
