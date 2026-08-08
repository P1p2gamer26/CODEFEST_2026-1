#!/usr/bin/env python3
"""E19: donde caen los documentos relevantes en el ranking agregado.

HIPOTESIS, escrita antes de medir: los documentos relevantes que el pool si
trae y la respuesta no entrega caen CERCA del top-3, o sea que el fallo es de
calibracion fina de la agregacion y no un descarte estructural.

JUSTIFICACION MECANICA. E18 cierra que el 93.2% de los documentos relevantes
esta en el pool a profundidad 200 y que ninguna consulta tiene pool ciego,
mientras el F1@3 es 0.440 sobre un techo de 0.906. La mitad de lo alcanzable
se pierde entre el pool y los 3 cupos: eso es agregacion. Pero nadie miro
DONDE caen. E07 barrio la estrategia y E14 propone un piso de longitud, los
dos a ciegas sobre la forma del fallo. Posiciones 4-8 y posiciones 30+ piden
intervenciones opuestas.

NO cambia nada de la entrega (regla 6 del handoff).

Uso:
    python dev/scripts/diagnostico_rank_doc_e19.py
"""

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

from eval_mini import cargar_jsonl  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

# Exactamente los de la entrega.
PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60


def main() -> None:
    gt = {g["query_id"]: set(g["docs_relevantes"])
          for g in cargar_jsonl(GT) if g["docs_relevantes"]}
    consultas = [c for c in cargar_jsonl(CONSULTAS) if c["query_id"] in gt]
    print(f"{len(consultas)} consultas evaluables\n")

    cache, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache[nombre] = idx
        else:
            cache[nombre] = faiss.read_index(str(encoder_dir(nombre) / "index.faiss"))
    enc_p = get_encoder(name=MINILM)
    print(f"  {cache[MINILM].ntotal:,} vectores por indice\n", flush=True)

    ranks = []              # posicion del relevante en el ranking agregado
    perfil_rel, perfil_win = [], []   # (n_chunks, mejor_score) del relevante y del ganador
    fuera_del_pool = 0
    detalle = []
    for n, c in enumerate(consultas, 1):
        qid = c["query_id"]
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, cache[MINILM], metadata, k=PROF)[:PROF]
        for sec in SECUNDARIOS:
            enc_s, idx_s = get_encoder(name=sec), cache[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        pool = sorted(hits, key=lambda h: -h.score)[:K_POOL]

        docs = aggregate_documents(pool, top_n=len(pool), strategy=AGG)
        pos = {d.doc_id: i for i, d in enumerate(docs, 1)}
        por_doc = defaultdict(list)
        for h in pool:
            por_doc[h.doc_id].append(h.score)

        top3 = [d.doc_id for d in docs[:3]]
        for d in top3:
            perfil_win.append((len(por_doc[d]), max(por_doc[d])))

        for rel in sorted(gt[qid]):
            r = pos.get(rel)
            if r is None:
                fuera_del_pool += 1
                continue
            ranks.append(r)
            perfil_rel.append((len(por_doc[rel]), max(por_doc[rel])))
            if r > 3:
                detalle.append((qid, rel, r, len(por_doc[rel]), max(por_doc[rel])))
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()

    # --- 1. Histograma de la posicion --------------------------------------
    print(f"\nposicion del documento relevante en el ranking agregado "
          f"({len(ranks)} pares dentro del pool, {fuera_del_pool} fuera):\n")
    tramos = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12),
              (13, 20), (21, 30), (31, 10**6)]
    for lo, hi in tramos:
        cuenta = sum(1 for r in ranks if lo <= r <= hi)
        etiqueta = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**5 else f"{lo}-{hi}")
        barra = "#" * cuenta
        print(f"  {etiqueta:>6s}  {cuenta:3d}  {cuenta / len(ranks):5.1%}  {barra}")
    dentro = sum(1 for r in ranks if r <= 3)
    print(f"\n  entregados (posicion <= 3): {dentro} de {len(ranks)} "
          f"({dentro / len(ranks):.1%})")
    print(f"  mediana de los que NO entran: "
          f"{statistics.median([r for r in ranks if r > 3]):.0f}")

    # --- 2. Cuanto ganaria ensanchar el cupo (no se puede, pero acota) ------
    print("\n  si el reto permitiera N cupos en vez de 3 (cota del reordenamiento):")
    for k in (3, 4, 5, 6, 8, 10):
        print(f"    @{k:<3d} {sum(1 for r in ranks if r <= k) / len(ranks):6.1%}")

    # --- 3. Perfil: por que gana el que gana --------------------------------
    print("\nperfil (chunks aportados al pool, mejor score de chunk):")
    for etiqueta, perfil in (("los 3 que ocupan cupo", perfil_win),
                             ("los relevantes", perfil_rel)):
        ch = [p[0] for p in perfil]
        sc = [p[1] for p in perfil]
        print(f"  {etiqueta:24s} n={len(perfil):3d}  chunks mediana "
              f"{statistics.median(ch):4.1f} (media {statistics.mean(ch):4.1f})  "
              f"mejor score mediana {statistics.median(sc):.3f}")
    perdedores = [(c, s) for _, _, _, c, s in detalle]
    if perdedores:
        ch = [p[0] for p in perdedores]
        sc = [p[1] for p in perdedores]
        print(f"  {'los relevantes que PIERDEN':24s} n={len(perdedores):3d}  "
              f"chunks mediana {statistics.median(ch):4.1f} "
              f"(media {statistics.mean(ch):4.1f})  "
              f"mejor score mediana {statistics.median(sc):.3f}")

    # --- 4. Cuantos chunks aporta el que gana, contra el que pierde --------
    tope = int(AGG[3:]) if AGG.startswith("top") else None
    if tope:
        satur_win = sum(1 for c, _ in perfil_win if c >= tope) / len(perfil_win)
        satur_perd = (sum(1 for c, _ in perdedores if c >= tope) / len(perdedores)
                      if perdedores else 0.0)
        print(f"\n  saturan el tope de {AGG} (aportan >= {tope} chunks al pool):")
        print(f"    los que ocupan cupo   {satur_win:6.1%}")
        print(f"    los relevantes que pierden {satur_perd:6.1%}")
        print("  si el primero es mucho mayor, la agregacion premia inundar el "
              "pool\n  y el piso de longitud de E14 apunta al sintoma equivocado.")

    # --- 5. Los peores casos, para mirarlos a mano -------------------------
    print("\nlos 12 relevantes que caen mas lejos:")
    for qid, rel, r, nch, sc in sorted(detalle, key=lambda x: -x[2])[:12]:
        print(f"  {qid}  {rel:18s} pos {r:3d}  {nch:2d} chunks  mejor {sc:.3f}")

    c = Counter(qid for qid, _, r, _, _ in detalle if r > 20)
    if c:
        print(f"\n  consultas con algun relevante mas alla de la posicion 20: "
              f"{' '.join(sorted(c))}")


if __name__ == "__main__":
    main()
