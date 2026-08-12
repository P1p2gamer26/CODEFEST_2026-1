#!/usr/bin/env python3
"""E30: desempatar por ENTIDADES COMPARTIDAS entre los documentos saturados.

HIPOTESIS (pre-registrada, cola.jsonl id E30): entre los documentos que saturan
el tope de `top5` -- donde E19/E20 probaron que el score de chunk no discrimina
(1.647 contra 1.672, 1.5%) -- ordenar por cuantas entidades de la consulta
menciona el documento sube el F1@3.

MECANICA. El diagnostico del 9 ago sobre las 50: se acierta al menos un
documento en 39 (78%), se falla la COLECCION en solo 4 (8%) y en 7 (14%) la
coleccion es correcta y el documento no. El fallo dominante es elegir entre
HERMANOS de la misma serie, y para eso hace falta una senal EXTERNA a los
scores. El grafo entregado tiene 224.101 nodos de entidades con `doc_id` en
las aristas y nunca se uso para esto.

RIESGO PRE-REGISTRADO. El desempate actua SOLO entre saturados, como
PERMUTACION dentro de ese grupo: los documentos no saturados conservan su
posicion exacta. Aplicarlo a todo el ranking es E24 otra vez, y E24 perdio.

Sin FAISS: lee `pools_entregados.json` (volcar_pools.py) y el cache de
entidades (mapa_entidades_grafo.py).

    .venv/Scripts/python.exe dev/scripts/barrido_entidades_e30.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.graph.ner import extract_entities  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.calidad_chunk import fraccion_aparato  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    IDIOMAS_LEGIBLES,
    UMBRAL_APARATO,
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from mapa_entidades_grafo import MIN_LARGO, cargar as cargar_entidades, normalizar  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
AGG = "top5"
TOPE_TOP5 = 5  # saturar = aportar >= 5 chunks al pool, la definicion de E19

CELDAS = ("entregada", "e30-graduada", "e30-binaria")
BASE = "entregada"
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

# Las 7 de "hermano equivocado" que lista diagnostico_colecciones.py sobre la
# entrega actual. Se rellenan al arrancar (no se fijan a mano) para que el
# analisis siga siendo correcto si la entrega cambia.


def entidades_consulta(texto: str, vocab: set) -> set:
    """Entidades de la consulta con el MISMO extractor con el que se construyo
    el grafo (spaCy es_core_news_sm, src/graph/ner.py), normalizadas igual.
    Solo se conservan las que existen en el grafo: el resto no puede desempatar
    nada."""
    ents = {normalizar(e.text) for e in extract_entities(texto, "es")}
    return {e for e in ents if len(e) >= MIN_LARGO and e in vocab}


def solapamiento(ents: set, docs_por_ent: dict) -> dict:
    """doc_id -> cuantas entidades de la consulta menciona."""
    c: Counter = Counter()
    for e in ents:
        c.update(docs_por_ent.get(e, ()))
    return c


def documentos(hits, solape, graduada):
    """Ranking de documentos con el desempate por entidades aplicado SOLO como
    permutacion dentro del grupo saturado."""
    doc_hits = aggregate_documents(hits, top_n=len(hits), strategy=AGG)
    conteo: Counter = Counter(h.doc_id for h in hits)
    idx_sat = [i for i, d in enumerate(doc_hits) if conteo[d.doc_id] >= TOPE_TOP5]
    if solape and len(idx_sat) > 1:
        sats = [doc_hits[i] for i in idx_sat]
        def clave(d):
            n = solape.get(d.doc_id, 0)
            return (-(min(n, 1) if not graduada else n), -d.score)
        for pos, d in zip(idx_sat, sorted(sats, key=clave)):
            doc_hits[pos] = d
    for i, d in enumerate(doc_hits[:3], 1):
        d.rank = i
    return doc_hits[:3]


def ordenar_fragmentos(hits, top_ids, texto_consulta=None):
    """Delega en la funcion REAL de src/retrieval/truncate.py.

    Reimplementar aqui los criterios fue un error que costo una medicion: este
    arnes tenia copiados los TRES criterios viejos (top-3, idioma, aparato) y
    seguia dando 0.490/0.476 de base cuando el sistema ya daba 0.506/0.491,
    porque E22 subio el idioma por encima del top-3 y E23 anadio la cobertura
    lexica. El F1@3 no se veia afectado -- el orden de fragmentos no toca los
    documentos -- pero las lecturas de NDCG median un sistema que ya no existe.

    La logica ya esta duplicada a proposito entre `src/` y `Entrega/generador.py`
    (punto 14 de las notas del proyecto) y hay un test que vigila ESE par. Una tercera copia
    dentro de un barrido no la vigila nadie: por eso aqui se llama, no se copia.
    """
    toks = frozenset(tokens_de(texto_consulta)) if texto_consulta else None
    return ordenar_para_fragmentos(
        hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks
    )


def resultado(qid, hits, solape, graduada, texto_consulta=None):
    doc_hits = documentos(hits, solape, graduada)
    top_ids = [d.doc_id for d in doc_hits]
    frags = enforce_word_limit(ordenar_fragmentos(hits, top_ids, texto_consulta))
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [{"rank": f["rank"], "chunk_id": f["chunk_id"],
                       "doc_id": f["doc_id"], "text": f["text"]} for f in frags],
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


def coleccion(doc_id):
    return doc_id.rsplit("-", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "entidades_e30")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    docs_por_ent = cargar_entidades()
    vocab = set(docs_por_ent)
    print(f"{len(vocab):,} entidades en el grafo\n")

    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}
    ents_por_q = {q: entidades_consulta(expandir_consulta(consultas[q]), vocab) for q in pools}
    solape_por_q = {q: solapamiento(e, docs_por_ent) for q, e in ents_por_q.items()}

    # ---- PUERTA DE ENTRADA (obligatoria, antes de mirar ninguna metrica) ----
    con_ent = [q for q, e in ents_por_q.items() if e]
    crudas = {q: {normalizar(x.text) for x in extract_entities(expandir_consulta(consultas[q]), "es")}
              for q in pools}
    print("=== PUERTA DE ENTRADA ===")
    print(f"  consultas con >=1 entidad reconocida por spaCy : "
          f"{sum(1 for q in crudas if crudas[q])}/50")
    print(f"  consultas con >=1 entidad TAMBIEN en el grafo  : {len(con_ent)}/50")
    print(f"  entidades por consulta (mediana de las que las tienen): "
          f"{sorted(len(ents_por_q[q]) for q in con_ent)[len(con_ent)//2] if con_ent else 0}")

    n_sat = n_sat_ent = n_permuta = 0
    for q, hits in hits_por_q.items():
        conteo = Counter(h.doc_id for h in hits)
        base = aggregate_documents(hits, top_n=len(hits), strategy=AGG)
        sat = [d.doc_id for d in base if conteo[d.doc_id] >= TOPE_TOP5]
        n_sat += len(sat)
        n_sat_ent += sum(1 for d in sat if solape_por_q[q].get(d, 0))
        top3 = [d.doc_id for d in base[:3]]
        if documentos(hits, solape_por_q[q], True)[:3] != base[:3]:
            pass
        if [d.doc_id for d in documentos(hits, solape_por_q[q], True)] != top3:
            n_permuta += 1
    print(f"  documentos saturados en total (50 consultas)   : {n_sat}")
    print(f"  ...con alguna entidad de su consulta           : {n_sat_ent} "
          f"({100*n_sat_ent/max(n_sat,1):.0f}%)")
    print(f"  consultas cuyo top-3 CAMBIA (celda graduada)   : {n_permuta}/50\n")

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes, {len(gt_hum)} humanas\n")

    # Consulta YA EXPANDIDA por el glosario: es la que se midio en E23.
    expandidas = {c["query_id"]: expandir_consulta(c["text"]) for c in cargar_jsonl(CONSULTAS)}

    archivos, guardadas = {}, {}
    for celda in CELDAS:
        sol = (lambda q: {}) if celda == BASE else (lambda q: solape_por_q[q])
        grad = celda == "e30-graduada"
        res = {q: resultado(q, h, sol(q), grad, expandidas[q]) for q, h in hits_por_q.items()}
        archivos[celda] = res
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                            + metricas(res, gt_hum))

    print(f"{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in CELDAS:
        print(f"{k + (' *' if k == BASE else ''):16s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la entregada; tiene que dar 0.440/0.506/0.491 y 0.400/0.436/0.429 (E09)\n")

    # las 7 de hermano equivocado, calculadas sobre la fila base
    gtm = {g["query_id"]: set(g["docs_relevantes"]) for g in gt_todo}
    hermano = []
    for g in gt_todo:
        q = g["query_id"]
        dev = [d["doc_id"] for d in archivos[BASE][q]["documents"][:3]]
        if set(dev) & gtm[q]:
            continue
        if {coleccion(d) for d in dev} & {coleccion(d) for d in gtm[q]}:
            hermano.append(q)
    print(f"=== HERMANO EQUIVOCADO: {len(hermano)} consultas -> {hermano}\n")

    base = guardadas[BASE]
    orden_q = [g["query_id"] for g in gt_todo]
    for k in CELDAS:
        if k == BASE:
            continue
        lineas = sum(1 for q in archivos[k]
                     if archivos[k][q] != archivos[BASE][q])
        docs = sum(1 for q in archivos[k]
                   for a, b in zip(archivos[k][q]["documents"], archivos[BASE][q]["documents"])
                   if a["doc_id"] != b["doc_id"])
        print(f"  === {k} ===  {lineas} lineas y {docs}/150 documentos cambian")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            g_ = sum(1 for d in deltas if d > 1e-9)
            p_ = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g_}g/{p_}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in guardadas[k][0] if v == 0)
        print(f"  VETO consultas con F1@3=0: {ceros} (entregada: {sum(1 for v in base[0] if v == 0)})")
        d_h = {q: d for q, d in zip(orden_q, [x - y for x, y in zip(guardadas[k][0], base[0])])}
        print("  las 7 de hermano equivocado: "
              + ", ".join(f"{q} {d_h.get(q, 0):+.2f}" for q in hermano))
        movidas = [q for q in hermano if abs(d_h.get(q, 0)) > 1e-9]
        print(f"  -> se mueven {len(movidas)}/{len(hermano)}: {movidas}\n")


if __name__ == "__main__":
    main()
