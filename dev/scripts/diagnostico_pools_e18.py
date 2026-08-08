#!/usr/bin/env python3
"""E18: a que profundidad aparece el primer documento relevante, por encoder.

HIPOTESIS, escrita antes de medir: las consultas que fallan comparten una
firma diagnosticable, y esa firma dice cual de las dos palancas de la ronda
(E16 doble consulta, E17 union de encoders) puede servir.

JUSTIFICACION MECANICA. las notas del proyecto declara agotado el eje "reordenar lo que el
pool ya trajo" y deja abierta la construccion del pool. Pero nunca se midio lo
mas basico: **el techo del pool**. Si a profundidad 200 el pool ya contiene
todos los documentos relevantes, entonces reordenar era efectivamente lo unico
que quedaba y las dos palancas de la ronda estan muertas antes de correrse. Si
no los contiene, esto dice cuanta profundidad haria falta y con que encoder.

Se mide con tres consultas por pregunta -- cruda, expandida, y con gte -- para
separar tres causas que hoy estan mezcladas: el glosario, el idioma y el
encoder.

NO cambia nada de la entrega. Es diagnostico (regla 6 del handoff).

Uso:
    python dev/scripts/diagnostico_pools_e18.py [--prof 5000]
"""

import argparse
import statistics
import sys
from pathlib import Path

import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

from eval_mini import cargar_jsonl  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
GTE = "gte-multilingual-base"

# Las tres ramas que se comparan. La segunda es lo que hace hoy la entrega.
RAMAS = ("minilm-crudo", "minilm-expandido", "gte-expandido")

CORTES = (100, 200, 500, 1000)


def primera_aparicion(hits, relevantes):
    """Rank (1-indexado) del primer chunk de cada doc_id relevante; None si no sale."""
    vistos = {}
    for i, h in enumerate(hits, 1):
        d = h.doc_id
        if d in relevantes and d not in vistos:
            vistos[d] = i
    return {d: vistos.get(d) for d in relevantes}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prof", type=int, default=5000,
                    help="profundidad maxima explorada (default 5000)")
    args = ap.parse_args()

    gt = {g["query_id"]: g for g in cargar_jsonl(GT) if g["docs_relevantes"]}
    consultas = [c for c in cargar_jsonl(CONSULTAS) if c["query_id"] in gt]
    print(f"{len(consultas)} consultas evaluables, profundidad {args.prof}\n")

    idx_m, metadata = load_index(MINILM)
    idx_g = faiss.read_index(str(encoder_dir(GTE) / "index.faiss"))
    enc_m, enc_g = get_encoder(name=MINILM), get_encoder(name=GTE)
    print(f"  MiniLM {idx_m.ntotal:,} / gte {idx_g.ntotal:,} vectores\n", flush=True)

    filas = {}   # query_id -> rama -> {doc_id: rank o None}
    disp = {}    # query_id -> dispersion del top-12 de la rama entregada
    for n, c in enumerate(consultas, 1):
        qid, rel = c["query_id"], set(gt[qid := c["query_id"]]["docs_relevantes"])
        crudo, expandido = c["text"], expandir_consulta(c["text"])
        por_rama = {}
        for rama, texto, enc, idx in (
            ("minilm-crudo", crudo, enc_m, idx_m),
            ("minilm-expandido", expandido, enc_m, idx_m),
            ("gte-expandido", expandido, enc_g, idx_g),
        ):
            hits = search(texto, enc, idx, metadata, k=args.prof)
            por_rama[rama] = primera_aparicion(hits, rel)
            if rama == "minilm-expandido":
                top = [h.score for h in hits[:12]]
                disp[qid] = max(top) - min(top)
        filas[qid] = por_rama
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()

    # --- 1. Recall del pool por profundidad -------------------------------
    print(f"\n{'rama':20s}" + "".join(f"{'@' + str(k):>9s}" for k in CORTES)
          + f"{'@' + str(args.prof):>9s}" + f"{'nunca':>9s}")
    for rama in RAMAS:
        total = sum(len(filas[q][rama]) for q in filas)
        cuenta = []
        for k in CORTES:
            cuenta.append(sum(1 for q in filas for r in filas[q][rama].values()
                              if r is not None and r <= k))
        hasta = sum(1 for q in filas for r in filas[q][rama].values() if r is not None)
        print(f"{rama:20s}" + "".join(f"{c / total:>9.3f}" for c in cuenta)
              + f"{hasta / total:>9.3f}" + f"{total - hasta:>9d}")
    print(f"\n  fraccion de los {total} pares (consulta, documento relevante) cuyo "
          f"documento aparece\n  con al menos un chunk dentro de la profundidad. "
          f"La entrega usa 200.")

    # --- 2. Lo que solo ve gte, y al reves --------------------------------
    K = 200
    def dentro(q, rama, k=K):
        return {d for d, r in filas[q][rama].items() if r is not None and r <= k}

    solo_gte = solo_mini = ambos = ninguno = 0
    ganadas_por_gte = []
    for q in filas:
        m, g = dentro(q, "minilm-expandido"), dentro(q, "gte-expandido")
        solo_gte += len(g - m)
        solo_mini += len(m - g)
        ambos += len(m & g)
        ninguno += len(set(filas[q]["minilm-expandido"]) - m - g)
        if g - m:
            ganadas_por_gte.append((q, sorted(g - m)))
    print(f"\ndocumentos relevantes dentro de la profundidad {K}:")
    print(f"  los dos encoders     {ambos}")
    print(f"  solo MiniLM          {solo_mini}")
    print(f"  solo gte             {solo_gte}   <- lo que E17 podria rescatar")
    print(f"  ninguno              {ninguno}   <- fuera del alcance de E17")
    for q, ds in ganadas_por_gte:
        print(f"    {q}: {', '.join(ds)}")

    # --- 3. Lo que aporta o quita el glosario -----------------------------
    gana = [(q, sorted(dentro(q, "minilm-expandido") - dentro(q, "minilm-crudo")))
            for q in filas]
    pierde = [(q, sorted(dentro(q, "minilm-crudo") - dentro(q, "minilm-expandido")))
              for q in filas]
    ng = sum(len(x[1]) for x in gana)
    np_ = sum(len(x[1]) for x in pierde)
    print(f"\nglosario a profundidad {K}: mete {ng} documentos y saca {np_}")
    for etiqueta, lista in (("mete", gana), ("saca", pierde)):
        for q, ds in lista:
            if ds:
                print(f"  {etiqueta} {q}: {', '.join(ds)}")
    print("  el 'saca' es lo que E16 (union de los dos pools) recuperaria por "
          "construccion.")

    # --- 4. Dispersion contra acierto -------------------------------------
    ciegas = [q for q in filas
              if not any(r is not None and r <= K
                         for r in filas[q]["minilm-expandido"].values())]
    resto = [q for q in filas if q not in ciegas]
    print(f"\ndispersion del top-12 (rama entregada):")
    for etiqueta, grupo in (("pool ciego", ciegas), ("pool con relevante", resto)):
        if grupo:
            vals = [disp[q] for q in grupo]
            print(f"  {etiqueta:20s} n={len(grupo):2d}  mediana "
                  f"{statistics.median(vals):.3f}  rango "
                  f"{min(vals):.3f}-{max(vals):.3f}")
    print(f"  consultas ciegas: {' '.join(sorted(ciegas))}")


if __name__ == "__main__":
    main()
