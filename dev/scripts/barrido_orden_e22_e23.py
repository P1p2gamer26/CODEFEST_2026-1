#!/usr/bin/env python3
"""E22 (idioma por encima del top-3) y E23 (cobertura lexica como 4o criterio).

Los dos son REORDENAMIENTOS del pool ya re-puntuado: no tocan vectores. Por eso
este arnes lee `dev/intermedios/pools_entregados.json` (volcar_pools.py) y NO
carga ningun indice FAISS -- 987 MB de RAM que este experimento no necesita.

Mide sobre `dev/src/` (aggregate_documents + ordenar_para_fragmentos +
enforce_word_limit), replicando el camino de `build_result_object` con los
defaults entregados (agg top5, cupo_alineado = 10 => rama simple).

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_orden_e22_e23.py
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.calidad_chunk import fraccion_aparato  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import Hit  # noqa: E402
from src.retrieval.truncate import IDIOMAS_LEGIBLES, UMBRAL_APARATO, enforce_word_limit  # noqa: E402

from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
AGG = "top5"

# Misma definicion de "palabra de contenido" que la puerta de entrada
# (puertas_e22_e23.py), fijada ANTES de medir. No se retoca segun el resultado.
STOPWORDS = frozenset("""
para como cual cuales cuanto cuantos donde desde entre sobre segun hacia hasta
ante bajo cabe contra durante mediante salvo tras esta este estos estas esos
esas aquel aquella ellos ellas nosotros vosotros pero sino aunque porque
cuando mientras siempre nunca tambien tanto tanta menos mismo misma otro otra
otros otras cada todo toda todos todas algun alguna algunos algunas ningun
ninguna mucho mucha muchos muchas poco poca pocos pocas
that this these those with from have been they their there where which what
when while about into over under between during through after before more
most such than then them your ours will would could should because being
qué cómo cuál dónde cuáles cuántos
""".split())

# Celdas. Cada una es la tupla de criterios de ordenacion, en orden.
CELDAS = {
    "entregada": ("top", "idioma", "aparato"),
    "e22-idioma-1o": ("idioma", "top", "aparato"),
    "e23-cobertura-4o": ("top", "idioma", "aparato", "cob_binaria"),
    "e23-cobertura-graduada": ("top", "idioma", "aparato", "cob_graduada"),
    # No pre-registrada: las dos adoptables juntas. Se mide porque es lo que
    # se aplicaria si ambas entran, y componerlas no es obvio a priori.
    "e22+e23": ("idioma", "top", "aparato", "cob_binaria"),
}
BASE = "entregada"
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)")


def normalizar(texto: str) -> set:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]{4,}", t) if w not in STOPWORDS}


def hits_desde_pool(pool: list[dict]) -> list[Hit]:
    return [
        Hit(rank=c["rank"], score=c["score"], chunk_id=c["chunk_id"], doc_id=c["doc_id"],
            fuente=c["fuente"], texto=c["texto"], formato=c["formato"],
            fenomeno=c["fenomeno"], idioma=c["idioma"], fila=c["fila"])
        for c in pool
    ]


def ordenar(hits, criterios, top_ids, toks_consulta):
    top = set(top_ids or ())

    def clave(h):
        partes = []
        for c in criterios:
            if c == "top":
                partes.append(1 if (top and h.doc_id not in top) else 0)
            elif c == "idioma":
                partes.append(0 if h.idioma in IDIOMAS_LEGIBLES else 1)
            elif c == "aparato":
                partes.append(1 if fraccion_aparato(h.texto) >= UMBRAL_APARATO else 0)
            elif c == "cob_binaria":
                partes.append(0 if (normalizar(h.texto) & toks_consulta) else 1)
            elif c == "cob_graduada":
                partes.append(-len(normalizar(h.texto) & toks_consulta))
        return tuple(partes)

    return sorted(hits, key=clave)


def resultado(qid, hits, criterios, toks):
    doc_hits = aggregate_documents(hits, top_n=3, strategy=AGG)
    top_ids = [d.doc_id for d in doc_hits]
    frags = enforce_word_limit(ordenar(hits, criterios, top_ids, toks))
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [
            {"rank": f["rank"], "chunk_id": f["chunk_id"], "doc_id": f["doc_id"], "text": f["text"]}
            for f in frags
        ],
    }


def metricas(res, gt):
    out = [[], [], []]
    for g in gt:
        r = res.get(g["query_id"])
        if not r:
            continue
        rel = set(g["docs_relevantes"])
        out[0].append(f1([d["doc_id"] for d in r["documents"][:3]], rel)[2])
        out[1].append(ndcg(r["fragments"], rel))
        out[2].append(ndcg_penalizado(r["fragments"], rel))
    return out


def diagnostico(res, pools, toks_por_q):
    """Las dos cifras que el NDCG no ve: ilegibles y cobertura cero."""
    idioma = {c["chunk_id"]: c["idioma"] for p in pools.values() for c in p}
    ileg = cero = alineados = 0
    for q, r in res.items():
        top3 = {d["doc_id"] for d in r["documents"][:3]}
        for fr in r["fragments"]:
            if idioma.get(fr["chunk_id"], "es") not in IDIOMAS_LEGIBLES:
                ileg += 1
            if not (normalizar(fr["text"]) & toks_por_q[q]):
                cero += 1
            if fr["doc_id"] in top3:
                alineados += 1
    return ileg, cero, alineados


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "orden_e22_e23")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    toks = {q: normalizar(expandir_consulta(consultas[q])) for q in pools}

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes\n")

    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}

    guardadas, diag, archivos = {}, {}, {}
    for celda, criterios in CELDAS.items():
        res = {q: resultado(q, h, criterios, toks[q]) for q, h in hits_por_q.items()}
        archivos[celda] = res
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep)
        diag[celda] = diagnostico(res, pools, toks)

    print(f"{'celda':24s}" + "".join(f"{c:>9s}" for c in COLS)
          + f"{'ilegib':>8s}{'cob=0':>8s}{'alin':>7s}")
    for k in guardadas:
        i, c, a = diag[k]
        print(f"{k + (' *' if k == BASE else ''):24s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k])
              + f"{i:>8d}{c:>8d}{a:>7d}")
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476 (regla de E09)\n")

    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        # cuantas lineas y fragmentos cambian respecto de la entregada
        lineas = sum(1 for q in archivos[k]
                     if archivos[k][q]["fragments"] != archivos[BASE][q]["fragments"]
                     or archivos[k][q]["documents"] != archivos[BASE][q]["documents"])
        frags = sum(1 for q in archivos[k]
                    for a, b in zip(archivos[k][q]["fragments"], archivos[BASE][q]["fragments"])
                    if a["chunk_id"] != b["chunk_id"] or a["text"] != b["text"])
        print(f"  === {k} ===  {lineas} lineas cambian, {frags}/500 fragmentos difieren en su posicion")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, bajo, alto = bootstrap_delta(deltas)
            g = sum(1 for d in deltas if d > 1e-9)
            p = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {g}g/{p}p  "
                  f"{'pasa' if bajo > -0.02 else 'NO pasa'}")
        # veto compartido: consultas con F1@3 = 0
        ceros = sum(1 for v in guardadas[k][0] if v == 0)
        print(f"  consultas con F1@3=0: {ceros} (entregada: {sum(1 for v in base[0] if v == 0)})\n")


if __name__ == "__main__":
    main()
