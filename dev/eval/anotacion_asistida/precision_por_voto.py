"""Precision de las marcas segun cuantas rondas votaron.

La pregunta que responde: el acuerdo entre agentes, senala acierto? Si la
unanimidad fuera confiable, 4/4 tendria precision alta aunque 1/4 no.
"""

import collections
import json
from pathlib import Path

S = Path(__file__).parent
NOMBRES = ("estricto", "inclusivo", "dominio", "literal")


def cargar(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


P = [{f["query_id"]: set(f["docs_relevantes"]) for f in cargar(S / f"panel_{n}.jsonl")} for n in NOMBRES]
H = {f["query_id"]: set(f["docs_relevantes"]) for f in cargar("dev/eval/ground_truth_mini.jsonl")}
calib = [q for q in P[0] if q in H]

por_voto = collections.defaultdict(lambda: [0, 0])
for q in calib:
    votos = collections.Counter(d for p in P for d in p.get(q, set()))
    for d, n in votos.items():
        por_voto[n][1] += 1
        por_voto[n][0] += d in H[q]

print(f"calibracion: {calib}\n")
print("votos   aciertos/marcas   precision")
for n in sorted(por_voto, reverse=True):
    a, t = por_voto[n]
    print(f" {n}/4        {a:2d}/{t:<3d}          {a / t:.2f}")

union = {q: set().union(*[p.get(q, set()) for p in P]) for q in calib}
invisibles = sum(len(H[q] - union[q]) for q in calib)
total_h = sum(len(H[q]) for q in calib)
print(f"\ndocumentos que el humano marco y NINGUN agente vio: {invisibles} de {total_h}")
print("(techo de recall del metodo: no puede acertar lo que nunca marca)")
