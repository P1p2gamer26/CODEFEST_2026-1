#!/usr/bin/env python3
"""Mide la expansion de consulta por glosario bilingue (src/retrieval/glosario.py).

Compara la configuracion entregada contra ella misma con la consulta expandida,
sobre los mismos indices. Lo unico que cambia es el TEXTO que se vectoriza, asi
que las consultas sin ningun termino del glosario dan resultados identicos y
solo se mueven las que lo tienen.

JUSTIFICACION MECANICA, escrita antes de ver los numeros: los terminos de
dominio en espanol de las consultas son entre 30 y 2.000 veces mas raros en el
corpus que su forma inglesa, y cinco estan ausentes (ver el docstring de
glosario.py, contado sobre los 128.526 chunks del indice entregado).

REGLA DE DECISION FIJADA ANTES DE MEDIR: se adopta si el IC al 90% del delta
pareado excluye una perdida de 0,02, decidiendo por F1@3 (el cambio mueve el
ranking de documentos) y mirando el NDCG@10 como segunda lectura. Se reporta
por separado sobre las 41 humanas, porque las 9 asistidas reproducen al humano
con F1 0.23 y dos de las consultas que el glosario toca estan entre ellas.

Uso:
    python dev/scripts/barrido_glosario.py
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta, terminos_expandidos  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, veredicto_bootstrap  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = [(GTE, 0.25), (E5, 0.25)]
RERANK_DEPTH = 200
K_POOL = 60


def responder(consultas, cache_idx, expandir: bool):
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    resultados = {}
    for c in consultas:
        texto = expandir_consulta(c["text"]) if expandir else c["text"]
        hits = search(texto, enc_p, idx_p, metadata, k=RERANK_DEPTH)[:RERANK_DEPTH]
        for sec, peso in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += peso * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:K_POOL]
        for i, h in enumerate(hits, 1):
            h.rank = i
        resultados[c["query_id"]] = generador.build_result_object(c["query_id"], hits)
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "glosario")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consultas = cargar_jsonl(CONSULTAS)
    tocadas = {c["query_id"] for c in consultas if terminos_expandidos(c["text"])}
    print(f"el glosario toca {len(tocadas)} de {len(consultas)} consultas: {sorted(tocadas)}\n")
    for c in consultas:
        if c["query_id"] in tocadas:
            print(f"  {c['query_id']}: + {' | '.join(terminos_expandidos(c['text']))}")

    gt_todo = [f for f in cargar_jsonl(GT) if f["docs_relevantes"]]
    gt_hum = [f for f in gt_todo if not f.get("anotador")]
    # Solo las consultas que el glosario toca Y tienen ground truth: es donde
    # el cambio puede hacer algo. Las demas diluyen el delta hacia cero.
    gt_tocadas = [f for f in gt_todo if f["query_id"] in tocadas]

    cache_idx = {}
    metadata = None
    for nombre in (MINILM, GTE, E5):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
    print()

    tablas = {}
    for etiqueta, expandir in (("base", False), ("glosario", True)):
        res = responder(consultas, cache_idx, expandir)
        tablas[etiqueta] = res
        salida = args.out_dir / f"{etiqueta}.jsonl"
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        print(f"  escrito {salida.name}", flush=True)

    A, B = tablas["glosario"], tablas["base"]
    for etiqueta, gt in (
        (f"todas ({len(gt_todo)})", gt_todo),
        (f"humanas ({len(gt_hum)})", gt_hum),
        (f"solo las que toca ({len(gt_tocadas)})", gt_tocadas),
    ):
        print(f"\n=== {etiqueta} ===")
        for nombre, fn in (
            ("F1@3", lambda r, g: f1([d["doc_id"] for d in r["documents"][:3]], set(g["docs_relevantes"]))[2]),
            ("NDCG@10", lambda r, g: ndcg(r["fragments"], set(g["docs_relevantes"]))),
        ):
            va = [fn(A[g["query_id"]], g) for g in gt]
            vb = [fn(B[g["query_id"]], g) for g in gt]
            media, bajo, alto = bootstrap_delta([x - y for x, y in zip(va, vb)])
            gana = sum(1 for x, y in zip(va, vb) if x > y + 1e-9)
            pierde = sum(1 for x, y in zip(va, vb) if x < y - 1e-9)
            print(f"  {nombre}: {sum(vb)/len(vb):.3f} -> {sum(va)/len(va):.3f}  "
                  f"(gana {gana}, pierde {pierde}, empata {len(va)-gana-pierde})")
            print("  " + veredicto_bootstrap(media, bajo, alto).strip())


if __name__ == "__main__":
    main()
