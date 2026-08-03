#!/usr/bin/env python3
"""Funde el consenso del panel de agentes en ground_truth_mini.jsonl.

Cubre las 9 consultas que ningun anotador humano pudo marcar. Decision
explicita del usuario, tomada CON la medicion a la vista: el panel reproduce
al humano con F1 0.23 (ver README.md de esta carpeta). Estas etiquetas NO son
equivalentes a las humanas y por eso van marcadas con `anotador`, para que
eval_mini pueda separarlas siempre.

Umbral: >=2 votos de 4. Es el mejor balance medido en la calibracion
(P 0.28 / F1 0.30) y exige que coincidan al menos dos sesgos distintos.

Uso:
    python dev/eval/panel_agentes/fundir_consenso.py            # simulacion
    python dev/eval/panel_agentes/fundir_consenso.py --escribir
"""

import argparse
import json
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
GT = AQUI.parent / "ground_truth_mini.jsonl"
SESGOS = ("estricto", "inclusivo", "dominio", "literal")
VOTO_MINIMO = 2
# Las 9 que ningun humano pudo marcar. El panel solo puede tocar estas: pisar
# una etiqueta humana con una de agente seria cambiar el patron de medida.
OBJETIVO = {"q001", "q007", "q008", "q011", "q012", "q015", "q028", "q038", "q048"}


def cargar(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escribir", action="store_true", help="Sin esto solo simula.")
    ap.add_argument("--voto-minimo", type=int, default=VOTO_MINIMO)
    args = ap.parse_args()

    paneles = [{f["query_id"]: set(f["docs_relevantes"]) for f in cargar(AQUI / f"panel_{s}.jsonl")} for s in SESGOS]
    por_qid = {f["query_id"]: f for f in cargar(GT)}

    humanas_tocadas = OBJETIVO & {q for q, f in por_qid.items() if f.get("docs_relevantes") and not f.get("anotador")}
    if humanas_tocadas:
        raise SystemExit(f"ABORTA: {sorted(humanas_tocadas)} ya tienen etiqueta humana no vacia")

    nuevas = 0
    for qid in sorted(OBJETIVO):
        votos = Counter(d for p in paneles for d in p.get(qid, set()))
        docs = sorted(d for d, v in votos.items() if v >= args.voto_minimo)
        print(f"{qid}: {len(docs)} docs  {docs}")
        if not docs:
            continue
        por_qid[qid] = {
            "query_id": qid,
            "nota": f"consenso de {len(SESGOS)} anotadores-agente, >={args.voto_minimo} votos",
            "anotador": "panel-agentes",
            "pool": "paraphrase-multilingual-MiniLM-L12-v2+bm25",
            "votos": {d: votos[d] for d in docs},
            "docs_relevantes": docs,
        }
        nuevas += 1

    print(f"\nconsultas con consenso: {nuevas} de {len(OBJETIVO)}")
    if not args.escribir:
        print("simulacion -- volver a correr con --escribir")
        return

    with open(GT, "w", encoding="utf-8") as f:
        for qid in sorted(por_qid):
            f.write(json.dumps(por_qid[qid], ensure_ascii=False) + "\n")
    humanas = sum(1 for v in por_qid.values() if not v.get("anotador") and v["docs_relevantes"])
    print(f"escrito {GT}: {len(por_qid)} consultas ({humanas} humanas + {nuevas} del panel)")


if __name__ == "__main__":
    main()
