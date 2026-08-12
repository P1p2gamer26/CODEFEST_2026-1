#!/usr/bin/env python3
"""E42: normalizar el score agregado por el tamano del documento.

E41 midio que los ceros son de RANKING, no de pool ciego: el documento
relevante tiene el mejor chunk del pool (top1 1.667-1.751) pero aporta 2-9
chunks, y el ganador aporta 8-38. Con top5 el ganador satura el tope y gana
por margenes de 0.0087-0.073. topM es un TOPE y ya esta saturado en los dos
(por eso top8 y top12 dieron identico en E33); esto es un NORMALIZADOR, que
es otra operacion.

Grilla pre-registrada y CERRADA: alpha en 0.00 0.02 0.05 0.10 0.20, con el
denominador en n_pool y n_corpus. No se extiende hacia arriba aunque el borde
gane: con alpha grande esto degenera en `mean`, que E41 midio en 0.0567 con
43 ceros.

Sin FAISS: lee dev/intermedios/pools_entregados.json (volcar_pools.py).

    .venv/Scripts/python.exe dev/scripts/barrido_norm_doc_e42.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR, ROOT_DIR  # noqa: E402
from src.retrieval.aggregate import DocumentHit, filtrar_por_fenomeno_dominante  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
METADATA = (ROOT_DIR / "Entrega" / "base_vectorial"
            / "encoder_paraphrase-multilingual-MiniLM-L12-v2" / "metadata.jsonl")
CACHE_CONTEOS = DEV_DIR / "intermedios" / "conteo_chunks_por_doc.json"

UMBRAL_FENOMENO = 0.8   # E32, configuracion entregada
M = 5                   # top5, configuracion entregada
ALPHAS = (0.00, 0.02, 0.05, 0.10, 0.20)
DENOMINADORES = ("n_pool", "n_corpus")


def agregar_normalizado(hits, top_n=3, m=M, alpha=0.0, denominador="n_pool",
                        conteos_corpus=None):
    """topM dividido por el tamano del documento elevado a alpha.

    alpha=0.0 devuelve exactamente topM (el divisor es 1.0), asi que la celda
    base tiene que reproducir la entrega digito a digito.
    """
    por_doc = defaultdict(list)
    for h in hits:
        por_doc[h.doc_id].append(h.score)

    agg = {}
    for doc_id, scores in por_doc.items():
        bruto = sum(sorted(scores, reverse=True)[:m])
        if alpha == 0.0:
            agg[doc_id] = bruto
            continue
        if denominador == "n_pool":
            tamano = len(scores)
        elif denominador == "n_corpus":
            tamano = (conteos_corpus or {}).get(doc_id, len(scores))
        else:
            raise ValueError(f"denominador desconocido: {denominador}")
        agg[doc_id] = bruto / (max(tamano, 1) ** alpha)

    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [DocumentHit(rank=i, doc_id=d, score=s)
            for i, (d, s) in enumerate(ranked, start=1)]


def _firma(path: Path) -> list:
    """(tamano, mtime) de metadata.jsonl. Si el indice se reconstruye, el
    archivo cambia de tamano o de fecha, y eso basta para invalidar el
    cache sin tener que hashear 150 MB (el mismo problema que obligo a
    poner el hash del texto en el checkpoint de codificacion, ver
    las notas del proyecto punto de "corridas largas": un cache que no valida su
    fuente sirve numeros viejos en silencio)."""
    st = path.stat()
    return [st.st_size, st.st_mtime]


def conteos_del_corpus(path=METADATA, cache=CACHE_CONTEOS):
    """Cuantos chunks tiene cada documento en TODO el corpus (no en el pool).

    Una pasada por metadata.jsonl (150 MB) y se cachea, pero el cache se
    invalida si metadata.jsonl cambio de tamano o de mtime desde que se
    genero (ver _firma).
    """
    firma = _firma(path)
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("firma") == firma:
            return d["conteos"]
    c = Counter()
    with open(path, encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                c[json.loads(linea)["doc_id"]] += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"firma": firma, "conteos": c}), encoding="utf-8")
    return dict(c)


def resultado(qid, hits, texto_consulta, alpha, denominador, conteos):
    """Camino online aplanado con los defaults entregados (E32 + cupo 10)."""
    hits = filtrar_por_fenomeno_dominante(hits, umbral=UMBRAL_FENOMENO)
    doc_hits = agregar_normalizado(hits, top_n=3, m=M, alpha=alpha,
                                   denominador=denominador, conteos_corpus=conteos)
    top_ids = [d.doc_id for d in doc_hits]
    toks = frozenset(tokens_de(texto_consulta))
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks)
    )
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [{"rank": f["rank"], "chunk_id": f["chunk_id"],
                       "doc_id": f["doc_id"], "text": f["text"]} for f in frags],
    }


def metricas(res, gt):
    """Devuelve (f1_por_consulta, ndcg_por_consulta, ndcgp_por_consulta)."""
    a, b, c = [], [], []
    for g in gt:
        r = res.get(g["query_id"])
        if not r:
            continue
        rel = set(g["docs_relevantes"])
        a.append(f1([d["doc_id"] for d in r["documents"][:3]], rel)[2])
        b.append(ndcg(r["fragments"], rel))
        c.append(ndcg_penalizado(r["fragments"], rel))
    return a, b, c


def ceros(res, gt):
    """Consultas con F1@3 = 0. Es el veto pre-registrado: hoy son 11."""
    fuera = []
    for g in gt:
        r = res.get(g["query_id"])
        if not r or not g["docs_relevantes"]:
            continue
        if f1([d["doc_id"] for d in r["documents"][:3]], set(g["docs_relevantes"]))[2] == 0.0:
            fuera.append(g["query_id"])
    return fuera


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path,
                    default=DEV_DIR / "experimentos" / "e42_resultados.json")
    args = ap.parse_args()

    _, pools = cargar_pools()
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    gt = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    # Criterio real de "independiente" (no una conjetura): sin campo 'pool' Y
    # sin campo 'anotador'. Es exactamente lo que usa --sin-pooling de
    # eval_mini.py combinado con excluir anotacion-asistida, y lo que ya usa
    # barrido_orden_e22_e23.py::gt_indep.
    indep = {g["query_id"] for g in gt if not g.get("pool") and not g.get("anotador")}
    conteos = conteos_del_corpus()

    filas, series = [], {}
    for denom in DENOMINADORES:
        for alpha in ALPHAS:
            # alpha=0.0 es la misma celda base en los dos denominadores (el
            # divisor es 1.0 sin importar que se divida): calcularla una sola
            # vez, no saltarse el resto de la grilla del segundo denominador.
            if alpha == 0.0 and denom != DENOMINADORES[0]:
                continue
            res = {qid: resultado(qid, hits_desde_pool(p), consultas[qid],
                                  alpha, denom, conteos)
                   for qid, p in pools.items()}
            f50, n50, p50 = metricas(res, gt)
            gi = [g for g in gt if g["query_id"] in indep]
            fi, ni, _ = metricas(res, gi)
            z = ceros(res, gt)
            clave = f"{denom}:a{alpha:.2f}"
            series[clave] = {"f50": f50, "n50": n50, "p50": p50, "fi": fi, "ni": ni}
            filas.append({"celda": clave, "F1(50)": sum(f50) / len(f50),
                          "ND(50)": sum(n50) / len(n50), "NDp(50)": sum(p50) / len(p50),
                          "F1(ind)": sum(fi) / len(fi) if fi else 0.0,
                          "ND(ind)": sum(ni) / len(ni) if ni else 0.0,
                          "ceros": len(z), "qids_cero": z})

    base = series["n_pool:a0.00"]
    print(f"{'celda':<20}{'F1(50)':>9}{'ND(50)':>9}{'NDp(50)':>9}"
          f"{'F1(ind)':>9}{'ND(ind)':>9}{'ceros':>7}")
    for f in filas:
        print(f"{f['celda']:<20}{f['F1(50)']:>9.4f}{f['ND(50)']:>9.4f}"
              f"{f['NDp(50)']:>9.4f}{f['F1(ind)']:>9.4f}{f['ND(ind)']:>9.4f}"
              f"{f['ceros']:>7d}")
    print("\nIC al 90% del delta pareado contra la celda base (alpha 0):")
    for clave, s in series.items():
        if clave.endswith("a0.00"):
            continue
        for etiqueta, x, y in (("F1(50)", s["f50"], base["f50"]),
                               ("ND(50)", s["n50"], base["n50"]),
                               ("F1(ind)", s["fi"], base["fi"]),
                               ("ND(ind)", s["ni"], base["ni"])):
            deltas = [xi - yi for xi, yi in zip(x, y)]
            media, lo, hi = bootstrap_delta(deltas)
            print(f"  {clave:<18}{etiqueta:<9}{media:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    args.salida.write_text(json.dumps(filas, indent=1, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\n-> {args.salida}")


if __name__ == "__main__":
    main()
