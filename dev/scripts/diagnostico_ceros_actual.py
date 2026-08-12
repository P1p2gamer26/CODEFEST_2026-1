#!/usr/bin/env python3
"""Diagnostico de las 11 consultas con F1@3 = 0.00 sobre la entrega actual.

diagnostico_ceros.py (el viejo) mide otro sistema: k_pool=60, agregacion sum,
sin glosario y sin cascada. Este reproduce EXACTAMENTE el camino entregado
(d200, cascada MiniLM->gte+e5 peso 0.60, k_pool=100, top5, glosario) usando el
arnes de E19, y clasifica cada consulta en:

  (a) fallo de pool        ningun chunk del documento relevante entra a d200
  (b) fallo de agregacion  esta en el pool pero el ranking agregado lo hunde
  (c) ground truth dudoso  la consulta tiene etiqueta anotacion-asistida

Diagnostico puro: no toca Entrega/ ni propone nada.

Uso:
    .venv/Scripts/python.exe dev/scripts/diagnostico_ceros_actual.py
"""

import sys
from collections import defaultdict
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

PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60
PROF_HONDA = 5000   # hasta donde se busca el relevante cuando no esta en el pool

CEROS = ("q003 q005 q007 q011 q014 q015 q034 q037 q044 q046 q047").split()


def main() -> None:
    gt = {g["query_id"]: g for g in cargar_jsonl(GT) if g["docs_relevantes"]}
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    ceros = sys.argv[1:] or CEROS

    cache, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache[nombre] = idx
        else:
            cache[nombre] = faiss.read_index(str(encoder_dir(nombre) / "index.faiss"))
    enc_p = get_encoder(name=MINILM)
    # cuantos chunks tiene cada doc_id en el indice entero (cota de lo alcanzable)
    chunks_por_doc = defaultdict(int)
    for m in metadata:
        chunks_por_doc[m["doc_id"]] += 1
    print(f"{cache[MINILM].ntotal:,} vectores por indice, "
          f"{len(chunks_por_doc):,} documentos\n", flush=True)

    resumen = []
    for qid in ceros:
        texto_crudo = consultas[qid]
        texto = expandir_consulta(texto_crudo)
        rel = set(gt[qid]["docs_relevantes"])
        anotador = gt[qid].get("anotador", "humano")

        hondos = search(texto, enc_p, cache[MINILM], metadata, k=PROF_HONDA)
        hits = [h for h in hondos[:PROF]]
        # la cascada muta h.score; se re-puntua sobre copias del corte a d200
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

        # primera aparicion de cada relevante en la busqueda honda (sin cascada)
        primera = {}
        for i, h in enumerate(hondos, 1):
            if h.doc_id in rel and h.doc_id not in primera:
                primera[h.doc_id] = i

        top3 = [d.doc_id for d in docs[:3]]
        en_pool = [d for d in rel if d in pos]
        print(f"=== {qid}  ({anotador}, {len(rel)} relevantes)")
        print(f"    {texto_crudo[:110]}")
        if texto != texto_crudo:
            print(f"    expandida: {texto[len(texto_crudo):].strip()[:90]}")
        print(f"    entregados: {top3}")
        for d in sorted(rel):
            p = pos.get(d)
            if p is not None:
                print(f"      {d:20s} POOL   pos_agregada {p:3d}  "
                      f"{len(por_doc[d]):2d} chunks en pool  "
                      f"mejor {max(por_doc[d]):.3f}  "
                      f"({chunks_por_doc.get(d, 0)} chunks en el indice)")
            else:
                pr = primera.get(d)
                estado = f"1er chunk en rank {pr}" if pr else f"nunca en top-{PROF_HONDA}"
                nidx = chunks_por_doc.get(d, 0)
                print(f"      {d:20s} FUERA  {estado}  "
                      f"({nidx} chunks en el indice{'' if nidx else ' -- AUSENTE'})")
        clase = ("agregacion" if en_pool else "pool")
        if anotador != "humano":
            clase += " / gt-dudoso"
        resumen.append((qid, anotador, len(rel), len(en_pool),
                        min([pos[d] for d in en_pool], default=None), clase))
        print()

    print(f"{'qid':6s} {'anotador':14s} {'rel':>4s} {'en pool':>8s} "
          f"{'mejor pos':>10s}  clasificacion")
    for qid, an, nrel, nen, mp in [(r[0], r[1], r[2], r[3], r[4]) for r in resumen]:
        clase = [r[5] for r in resumen if r[0] == qid][0]
        print(f"{qid:6s} {an:14s} {nrel:4d} {nen:8d} "
              f"{(str(mp) if mp else '-'):>10s}  {clase}")


if __name__ == "__main__":
    main()
