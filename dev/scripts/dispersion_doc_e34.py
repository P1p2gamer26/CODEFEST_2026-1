#!/usr/bin/env python3
"""E34, paso 0: la DISTRIBUCION de la dispersion de los scores AGREGADOS de
documento. Es la puerta de entrada del experimento.

E29 murio porque la dispersion del top-12 de CHUNKS resulto continua y ningun
umbral inducia una particion. La magnitud de aca es OTRA -- los scores ya
agregados a documento, sobre el pool YA filtrado por fenomeno (E32, umbral
0.8) -- y nadie la ha mirado. Si tambien sale continua, E34 esta muerto antes
de medir y eso es el resultado.

No carga FAISS: lee pools_entregados.json.

    .venv/Scripts/python.exe dev/scripts/dispersion_doc_e34.py
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval.aggregate import (  # noqa: E402
    aggregate_documents,
    filtrar_por_fenomeno_dominante,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

AGG = "top5"
M_TOPE = 5
UMBRAL_FENOMENO = 0.8   # E32, lo que hoy hay en la entrega


def main() -> None:
    config, pools = cargar_pools()
    print("config del volcado:", config, "\n")

    filas = []
    for qid in sorted(pools):
        hits = filtrar_por_fenomeno_dominante(hits_desde_pool(pools[qid]),
                                              umbral=UMBRAL_FENOMENO)
        base = aggregate_documents(hits, top_n=len(hits), strategy=AGG)
        s = [d.score for d in base]
        n_chunks = defaultdict(int)
        for h in hits:
            n_chunks[h.doc_id] += 1
        sat = [d for d in base if n_chunks[d.doc_id] >= M_TOPE]
        # Tres lecturas de "empate", todas relativas para ser comparables
        # entre consultas (los scores absolutos dependen de la consulta).
        gap34 = (s[2] - s[3]) / s[2] if len(s) > 3 and s[2] > 0 else float("nan")
        disp5 = (s[0] - s[4]) / s[0] if len(s) > 4 and s[0] > 0 else float("nan")
        # dispersion DENTRO del conjunto saturado, que es sobre el que E24 actua
        ss = [d.score for d in sat]
        disp_sat = (ss[0] - ss[-1]) / ss[0] if len(ss) > 1 and ss[0] > 0 else float("nan")
        filas.append((qid, gap34, disp5, disp_sat, len(sat)))

    for nombre, col in (("gap relativo doc3->doc4", 1),
                        ("dispersion relativa top-5", 2),
                        ("dispersion dentro de los saturados", 3)):
        v = sorted(f[col] for f in filas if f[col] == f[col])
        print(f"=== {nombre} ===  n={len(v)}")
        print("  min %.4f  p10 %.4f  p25 %.4f  mediana %.4f  p75 %.4f  p90 %.4f  max %.4f"
              % (v[0], v[len(v)//10], v[len(v)//4], statistics.median(v),
                 v[3*len(v)//4], v[9*len(v)//10], v[-1]))
        # el salto mas grande entre valores consecutivos: si hay particion,
        # tiene que verse aca.
        saltos = sorted(((v[i+1] - v[i], i) for i in range(len(v)-1)), reverse=True)
        print("  mayores saltos entre valores consecutivos:")
        for d, i in saltos[:3]:
            print(f"    +{d:.4f} en {v[i]:.4f} -> {v[i+1]:.4f}  (parte {i+1}/{len(v)-i-1})")
        print("  valores ordenados:", " ".join(f"{x:.3f}" for x in v))
        print()

    print("documentos saturados por consulta: media %.1f, min %d, max %d"
          % (sum(f[4] for f in filas)/len(filas),
             min(f[4] for f in filas), max(f[4] for f in filas)))


if __name__ == "__main__":
    main()
