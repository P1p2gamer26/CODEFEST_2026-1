"""Mide al panel de agentes contra las etiquetas humanas.

Uso: python calibrar_panel.py voto_minimo panel_*.jsonl

Las 6 consultas de CALIBRACION estan anotadas a mano y son INDEPENDIENTES
(sus candidatos no los propuso el recuperador), asi que sirven de patron.
Las 9 restantes no tienen etiqueta y solo se reportan.

El criterio de aceptacion se fija ANTES de ver el resultado, como manda
docs/lecciones_metodologia.md:

  el consenso del panel se adopta como ground truth SOLO si su F1 contra el
  humano supera 0.50 en las 6 de calibracion.

0.50 no es arbitrario: el recuperador que se quiere evaluar saca 0.344. Un
anotador que acierta menos que eso mide parecido a otro modelo, no acierto.
El anotador-agente unico ya medido dio 0.28 y por eso se descarto.
"""

import json
import sys
from collections import Counter
from pathlib import Path

SIN_ANOTAR = {"q001", "q007", "q008", "q011", "q012", "q015", "q028", "q038", "q048"}
GT = Path("dev/eval/ground_truth_mini.jsonl")
UMBRAL_ADOPCION = 0.50


def cargar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def prf(pred: set, real: set):
    if not pred and not real:
        return 1.0, 1.0, 1.0
    if not pred or not real:
        return 0.0, 0.0, 0.0
    ac = len(pred & real)
    if not ac:
        return 0.0, 0.0, 0.0
    p, r = ac / len(pred), ac / len(real)
    return p, r, 2 * p * r / (p + r)


def main():
    voto_min = int(sys.argv[1])
    paneles = {Path(p).stem: {f["query_id"]: set(f["docs_relevantes"]) for f in cargar_jsonl(p)} for p in sys.argv[2:]}
    humano = {f["query_id"]: set(f["docs_relevantes"]) for f in cargar_jsonl(GT)}
    calib = sorted(set(humano) & set.union(*[set(p) for p in paneles.values()]))

    print(f"panelistas: {list(paneles)}\nconsultas de calibracion: {calib}\n")

    # cada anotador por separado
    print(f"{'anotador':28s} {'P':>6s} {'R':>6s} {'F1':>6s}  marcas/consulta")
    for nombre, marcas in paneles.items():
        ps = rs = fs = 0.0
        for q in calib:
            p, r, f = prf(marcas.get(q, set()), humano[q])
            ps, rs, fs = ps + p, rs + r, fs + f
        n = len(calib)
        densidad = sum(len(marcas.get(q, set())) for q in marcas) / max(len(marcas), 1)
        print(f"{nombre:28s} {ps/n:6.2f} {rs/n:6.2f} {fs/n:6.2f}  {densidad:.1f}")

    # consenso
    consenso = {}
    todas = set().union(*[set(p) for p in paneles.values()])
    for q in todas:
        votos = Counter(d for marcas in paneles.values() for d in marcas.get(q, set()))
        consenso[q] = ({d for d, v in votos.items() if v >= voto_min}, votos)

    ps = rs = fs = 0.0
    for q in calib:
        p, r, f = prf(consenso[q][0], humano[q])
        ps, rs, fs = ps + p, rs + r, fs + f
    n = len(calib)
    print(f"\n{'CONSENSO (>=' + str(voto_min) + ' votos)':28s} {ps/n:6.2f} {rs/n:6.2f} {fs/n:6.2f}")

    print("\ndetalle de calibracion:")
    for q in calib:
        pred, votos = consenso[q]
        print(f"  {q}  humano={sorted(humano[q])}")
        print(f"       panel ={sorted(pred)}  votos={dict(votos.most_common(6))}")

    f1 = fs / n
    print(f"\n{'=' * 60}")
    if f1 >= UMBRAL_ADOPCION:
        print(f"ADOPTABLE: F1 {f1:.2f} >= {UMBRAL_ADOPCION}. El consenso puede usarse como ground truth.")
    else:
        print(f"NO ADOPTABLE: F1 {f1:.2f} < {UMBRAL_ADOPCION}. El panel no reproduce al humano;")
        print("usarlo como ground truth mediria parecido entre modelos, no acierto.")
    print("=" * 60)

    print("\nlas 9 sin etiqueta humana (solo informativo):")
    for q in sorted(SIN_ANOTAR & todas):
        pred, votos = consenso[q]
        print(f"  {q}  consenso={sorted(pred) or 'VACIA'}  votos={dict(votos.most_common(8))}")


if __name__ == "__main__":
    main()
