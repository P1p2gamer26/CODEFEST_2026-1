#!/usr/bin/env python3
"""E45: el desempate por grafo (commit 8bfb9d6) bajo E42 (normalizacion por
tamano de documento, ver barrido_norm_doc_e42.py).

PREGUNTA: E42 cambia el ranking de documentos de varias consultas. El grafo
hoy es inerte porque casi no hay empates de score que romper. ¿Sigue siendo
inerte con E42 activo, o las dos cosas interactuan?

Rejilla 2x2: base / grafo / e42 / e42+grafo. Sin FAISS ni torch: lee el pool
ya re-puntuado de dev/intermedios/pools_entregados.json (volcar_pools.py) y
reusa agregar_normalizado/conteos_del_corpus de barrido_norm_doc_e42.py (NO
se copian). La logica de desempate por grafo SI se copia de
Entrega/generador.py (funciones graph_search/desempatar_con_grafo y sus
dependencias de NER) porque ese archivo es autocontenido y no se puede
importar -- mantener sincronizada con generador.py si cambia.

    .venv/Scripts/python.exe dev/scripts/barrido_grafo_e42_e45.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR, ROOT_DIR  # noqa: E402
from src.retrieval.aggregate import filtrar_por_fenomeno_dominante  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_norm_doc_e42 import agregar_normalizado, conteos_del_corpus  # noqa: E402
from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
GRAFO_PATH = ROOT_DIR / "Entrega" / "base_vectorial" / "grafo" / "grafo.graphml"
RESULTADOS_ENTREGA = ROOT_DIR / "Entrega" / "resultados.jsonl"

UMBRAL_FENOMENO = 0.8
M = 5
ALPHA_E42 = 0.02
DENOM_E42 = "n_corpus"


# --- copiado de Entrega/generador.py (desempate por grafo, sec. 8.5) ---
# Mantener sincronizado si generador.py cambia esta parte.

SPACY_MODEL_BY_LANG = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
}
NER_DEFAULT_LANG = "es"
NER_EXCLUDED_LABELS = {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}


@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class GraphHit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str


@lru_cache(maxsize=None)
def _get_ner_pipeline(lang: str):
    import spacy

    model_name = SPACY_MODEL_BY_LANG.get(lang, SPACY_MODEL_BY_LANG[NER_DEFAULT_LANG])
    return spacy.load(model_name)


def extract_entities(text, lang):
    if not text or not text.strip():
        return []
    nlp = _get_ner_pipeline(lang or NER_DEFAULT_LANG)
    doc = nlp(text)
    return [
        Entity(ent.text.strip(), ent.label_, ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.text.strip() and ent.label_ not in NER_EXCLUDED_LABELS
    ]


def _normalize(text):
    return text.strip().lower()


def graph_search(query, graph, lang=None, k=10):
    query_entities = {_normalize(e.text) for e in extract_entities(query, lang)}
    if not query_entities:
        return []
    # El indice de nodos se construye UNA vez y se cachea en el propio grafo.
    # generador.py lo rehace por consulta, que en una sola corrida da igual;
    # aca hay 4 celdas x 50 consultas = 200 reconstrucciones de un dict de
    # 224.101 entradas, y con spaCy cargado eso revienta con MemoryError.
    node_by_normalized = getattr(graph, "_idx_normalizado", None)
    if node_by_normalized is None:
        node_by_normalized = {_normalize(n): n for n in graph.nodes}
        graph._idx_normalizado = node_by_normalized
    matched_nodes = [node_by_normalized[e] for e in query_entities if e in node_by_normalized]
    if not matched_nodes:
        return []
    evidence_count = defaultdict(int)
    for node in matched_nodes:
        for _, _, data in graph.out_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1
        for _, _, data in graph.in_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1
    ranked = sorted(evidence_count.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        GraphHit(rank=i, score=float(count), doc_id=doc_id, chunk_id=chunk_id)
        for i, ((doc_id, chunk_id), count) in enumerate(ranked, start=1)
    ]


def desempatar_con_grafo(hits, graph_hits):
    if not graph_hits:
        return hits
    evidencia = {gh.chunk_id: gh.score for gh in graph_hits}
    return sorted(hits, key=lambda h: (h.score, evidencia.get(h.chunk_id, 0.0)), reverse=True)


# --- fin de lo copiado de generador.py ---


def resultado(qid, hits, texto_consulta, con_grafo, graph, con_e42, conteos):
    """Camino online aplanado con los defaults entregados (E32 + cupo 10),
    mas los interruptores de E42 (normalizacion) y grafo (desempate)."""
    if con_grafo and graph is not None:
        graph_hits = graph_search(texto_consulta, graph, lang=hits[0].idioma if hits else None,
                                   k=100)
        hits = desempatar_con_grafo(hits, graph_hits)

    hits = filtrar_por_fenomeno_dominante(hits, umbral=UMBRAL_FENOMENO)
    if con_e42:
        doc_hits = agregar_normalizado(hits, top_n=3, m=M, alpha=ALPHA_E42,
                                       denominador=DENOM_E42, conteos_corpus=conteos)
    else:
        doc_hits = agregar_normalizado(hits, top_n=3, m=M, alpha=0.0)

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
    fuera = []
    for g in gt:
        r = res.get(g["query_id"])
        if not r or not g["docs_relevantes"]:
            continue
        if f1([d["doc_id"] for d in r["documents"][:3]], set(g["docs_relevantes"]))[2] == 0.0:
            fuera.append(g["query_id"])
    return fuera


def lineas_distintas(base, otra):
    """qids donde cambian los 3 doc_id o los 10 chunk_id, en orden, vs. base."""
    cambios = []
    for qid, rb in base.items():
        ro = otra[qid]
        docs_b = [d["doc_id"] for d in rb["documents"]]
        docs_o = [d["doc_id"] for d in ro["documents"]]
        chunks_b = [f["chunk_id"] for f in rb["fragments"]]
        chunks_o = [f["chunk_id"] for f in ro["fragments"]]
        if docs_b != docs_o or chunks_b != chunks_o:
            cambios.append(qid)
    return cambios


def verificar_fidelidad(res_base, consultas):
    """La celda base tiene que reproducir Entrega/resultados.jsonl byte a byte
    en doc_id y chunk_id, en orden, las 50 consultas. Es la puerta: si falla,
    el arnes esta mintiendo y no se mira ninguna celda mas."""
    entrega = {json.loads(l)["query_id"]: json.loads(l)
               for l in RESULTADOS_ENTREGA.read_text(encoding="utf-8").splitlines() if l.strip()}
    fallos = []
    for qid in consultas:
        r = res_base.get(qid)
        e = entrega.get(qid)
        if r is None or e is None:
            fallos.append((qid, "falta"))
            continue
        docs_r = [d["doc_id"] for d in r["documents"]]
        docs_e = [d["doc_id"] for d in e["documents"]]
        chunks_r = [f["chunk_id"] for f in r["fragments"]]
        chunks_e = [f["chunk_id"] for f in e["fragments"]]
        if docs_r != docs_e or chunks_r != chunks_e:
            fallos.append((qid, "diverge"))
    return fallos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path,
                    default=DEV_DIR / "experimentos" / "e45_resultados.json")
    args = ap.parse_args()

    _, pools = cargar_pools()
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    gt = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    indep = {g["query_id"] for g in gt if not g.get("pool") and not g.get("anotador")}
    conteos = conteos_del_corpus()

    import networkx as nx
    graph = nx.read_graphml(GRAFO_PATH)
    print(f"grafo cargado: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas")

    hits_por_q = {qid: hits_desde_pool(p) for qid, p in pools.items()}

    CELDAS = {
        "base": (False, False),
        "grafo": (True, False),
        "e42": (False, True),
        "e42+grafo": (True, True),
    }

    resultados_por_celda = {}
    for nombre, (con_grafo, con_e42) in CELDAS.items():
        res = {
            qid: resultado(qid, hits, consultas[qid], con_grafo, graph, con_e42, conteos)
            for qid, hits in hits_por_q.items()
        }
        resultados_por_celda[nombre] = res

    # --- puerta de fidelidad: innegociable ---
    fallos = verificar_fidelidad(resultados_por_celda["base"], list(consultas))
    if fallos:
        print(f"PUERTA DE FIDELIDAD FALLIDA: {len(fallos)} consultas divergen de la entrega:")
        for qid, motivo in fallos:
            print(f"  {qid}: {motivo}")
        sys.exit(1)
    print(f"puerta de fidelidad: 50/50 OK (base reproduce Entrega/resultados.jsonl)")

    filas = {}
    series = {}
    for nombre, res in resultados_por_celda.items():
        f50, n50, p50 = metricas(res, gt)
        gi = [g for g in gt if g["query_id"] in indep]
        fi, ni, pi = metricas(res, gi)
        z = ceros(res, gt)
        cambios = lineas_distintas(resultados_por_celda["base"], res)
        series[nombre] = {"f50": f50, "n50": n50, "p50": p50, "fi": fi, "ni": ni, "pi": pi}
        filas[nombre] = {
            "F1(50)": sum(f50) / len(f50), "ND(50)": sum(n50) / len(n50),
            "NDp(50)": sum(p50) / len(p50),
            "F1(ind)": sum(fi) / len(fi) if fi else 0.0,
            "ND(ind)": sum(ni) / len(ni) if ni else 0.0,
            "NDp(ind)": sum(pi) / len(pi) if pi else 0.0,
            "ceros": len(z), "qids_cero": z,
            "lineas_cambiadas_vs_base": len(cambios), "qids_cambiados": cambios,
        }

    print(f"\n{'celda':<14}{'F1(50)':>9}{'ND(50)':>9}{'NDp(50)':>9}"
          f"{'F1(ind)':>9}{'ND(ind)':>9}{'ceros':>7}{'cambios':>9}")
    for nombre, f in filas.items():
        print(f"{nombre:<14}{f['F1(50)']:>9.4f}{f['ND(50)']:>9.4f}{f['NDp(50)']:>9.4f}"
              f"{f['F1(ind)']:>9.4f}{f['ND(ind)']:>9.4f}{f['ceros']:>7d}"
              f"{f['lineas_cambiadas_vs_base']:>9d}")

    print("\nqids que cambian vs. base, por celda:")
    for nombre, f in filas.items():
        if nombre != "base":
            print(f"  {nombre}: {f['qids_cambiados']}")

    print("\ne42+grafo vs. e42 solo (la comparacion que importa):")
    cambios_interaccion = lineas_distintas(resultados_por_celda["e42"], resultados_por_celda["e42+grafo"])
    print(f"  lineas distintas: {len(cambios_interaccion)} -> {cambios_interaccion}")

    print("\nIC al 90% del delta pareado, cada celda contra base:")
    base_s = series["base"]
    for nombre, s in series.items():
        if nombre == "base":
            continue
        for etiqueta, x, y in (("F1(50)", s["f50"], base_s["f50"]),
                               ("ND(50)", s["n50"], base_s["n50"]),
                               ("F1(ind)", s["fi"], base_s["fi"]),
                               ("ND(ind)", s["ni"], base_s["ni"])):
            deltas = [xi - yi for xi, yi in zip(x, y)]
            media, lo, hi = bootstrap_delta(deltas)
            print(f"  {nombre:<12}{etiqueta:<9}{media:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print("\nIC al 90%: e42+grafo vs. e42 solo:")
    s_e42, s_comb = series["e42"], series["e42+grafo"]
    for etiqueta, x, y in (("F1(50)", s_comb["f50"], s_e42["f50"]),
                           ("ND(50)", s_comb["n50"], s_e42["n50"]),
                           ("F1(ind)", s_comb["fi"], s_e42["fi"]),
                           ("ND(ind)", s_comb["ni"], s_e42["ni"])):
        deltas = [xi - yi for xi, yi in zip(x, y)]
        media, lo, hi = bootstrap_delta(deltas)
        print(f"  {etiqueta:<9}{media:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    args.salida.write_text(json.dumps(filas, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.salida}")


if __name__ == "__main__":
    main()
