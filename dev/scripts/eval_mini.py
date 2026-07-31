#!/usr/bin/env python3
"""F1@3 aproximado contra un mini ground truth anotado a mano.

ADL no publica el ground truth, asi que sin esto las decisiones de diseno
(un encoder o dos, fusion RRF si o no, grafo si o no) se toman a ojo. Con
~10 consultas anotadas alcanza para comparar configuraciones entre si; el
valor absoluto NO es la nota oficial (la anotacion es parcial: los
documentos no anotados cuentan como irrelevantes aunque quiza no lo sean,
asi que el F1 real sera mayor o igual que este).

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
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth} -- hay que anotarlo a mano primero")
        sys.exit(2)

    resultados = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    gt = cargar_jsonl(args.ground_truth)

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


if __name__ == "__main__":
    main()
