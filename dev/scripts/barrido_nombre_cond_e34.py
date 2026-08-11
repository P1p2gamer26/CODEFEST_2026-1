#!/usr/bin/env python3
"""E34: el nombre del documento como desempate CONDICIONAL.

E24 midio el nombre como desempate INCONDICIONAL entre los documentos que
saturan el tope de top5: rescato q003 y q037 (las primeras consultas en cero
que movio ninguna palanca del proyecto) y perdio donde ya se acertaba mucho
(q041 1.00->0.67, q020 y q032 0.67->0.33). Su cierre dejo viva la variante
condicional: disparar SOLO cuando el ranking agregado de documentos esta
empatado.

PUERTA DE ENTRADA (dispersion_doc_e34.py): la dispersion relativa del top-5
agregado SI induce una particion -- 18 consultas por debajo de 0.051 y 32 por
encima de 0.190, con un hueco vacio de 0.14 en medio. No es el caso continuo
que mato a E29.

UMBRALES, fijados ANTES de medir y sin grilla:
  - cond-disp5 : dispara si (s1-s5)/s1 < 0.10  -> el punto medio del hueco.
                 Es LA celda que decide.
  - cond-gap34 : dispara si (s3-s4)/s3 < 0.10  -> CONTROL. Su particion cae en
                 41/50, o sea casi incondicional: se espera que reproduzca a
                 E24. Esta para comprobar el veto (c), no para adoptarse.

BASE: la entrega ACTUAL, que ya incluye E32 (post-filtrado por fenomeno,
umbral 0.8). Tiene que reproducir 0.455/0.516/0.499 y 0.433/0.474/0.467.

VETOS pre-registrados:
  (a) que no baje ninguna consulta que hoy vale F1@3 = 1.00 (hundio a E24);
  (b) que no suban las consultas que hoy valen F1@3 = 0;
  (c) si dispara en casi todas, es E24 otra vez y se dice.

No toca `Entrega/`. No carga FAISS (solo el encoder MiniLM, para los nombres).

    .venv/Scripts/python.exe dev/scripts/barrido_nombre_cond_e34.py
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import (  # noqa: E402
    DocumentHit,
    aggregate_documents,
    filtrar_por_fenomeno_dominante,
)
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from barrido_orden_e22_e23 import hits_desde_pool  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from nombres_doc_e24 import MIN_TOKENS, normalizar  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
AGG, M_TOPE, UMBRAL_FENOMENO = "top5", 5, 0.8

BASE = "entregada"
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)",
        "F1(hum)", "ND(hum)", "NDp(hum)")

# Celdas: (nombre del detector, umbral). Dos valores discretos, no una grilla.
CELDAS = {
    "cond-disp5": ("disp5", 0.10),
    "cond-gap34": ("gap34", 0.10),          # control, ~incondicional
    "e24-incondicional": (None, None),      # reproduccion de E24 sobre la base nueva
}


def dispersiones(base_docs):
    """Las dos lecturas de empate, relativas para ser comparables entre consultas."""
    s = [d.score for d in base_docs]
    gap34 = (s[2] - s[3]) / s[2] if len(s) > 3 and s[2] > 0 else 1.0
    disp5 = (s[0] - s[4]) / s[0] if len(s) > 4 and s[0] > 0 else 1.0
    return {"gap34": gap34, "disp5": disp5}


def permutar_saturados(base_docs, n_chunks, cos):
    """El criterio de E24: los saturados se permutan ENTRE SI, en las posiciones
    que ya ocupan. Nada mas se mueve."""
    idx = [i for i, d in enumerate(base_docs) if n_chunks[d.doc_id] >= M_TOPE]
    vals = [cos[d.doc_id] for i, d in enumerate(base_docs)
            if i in set(idx) and d.doc_id in cos]
    neutro = statistics.median(vals) if vals else 0.0
    clave = {base_docs[i].doc_id: cos.get(base_docs[i].doc_id, neutro) for i in idx}
    orden = sorted(idx, key=lambda i: -clave[base_docs[i].doc_id])  # estable
    salida = list(base_docs)
    for destino, origen in zip(idx, orden):
        salida[destino] = base_docs[origen]
    return salida


def resultado(qid, hits, texto_consulta, cos=None, detector=None, umbral=None):
    hits = filtrar_por_fenomeno_dominante(hits, umbral=UMBRAL_FENOMENO)
    base_docs = aggregate_documents(hits, top_n=len(hits), strategy=AGG)
    disparo = False
    if cos is not None:
        disparo = detector is None or dispersiones(base_docs)[detector] < umbral
        if disparo:
            n_chunks = defaultdict(int)
            for h in hits:
                n_chunks[h.doc_id] += 1
            base_docs = permutar_saturados(base_docs, n_chunks, cos)
    doc_hits = [DocumentHit(rank=i, doc_id=d.doc_id, score=d.score)
                for i, d in enumerate(base_docs[:3], 1)]
    top_ids = [d.doc_id for d in doc_hits]
    toks = frozenset(tokens_de(texto_consulta))
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks))
    return {
        "query_id": qid, "disparo": disparo,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=DEV_DIR / "intermedios" / "nombre_cond_e34")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}
    expandidas = {c["query_id"]: expandir_consulta(c["text"])
                  for c in cargar_jsonl(CONSULTAS)}

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    gt_ag = [g for g in gt_todo if g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas, "
          f"{len(gt_ag)} de agente\n")

    # --- nombres: solo los documentos que aparecen en algun pool ---
    fuente_de = {}
    for p in pools.values():
        for c in p:
            fuente_de.setdefault(c["doc_id"], c["fuente"])
    informativos = {d: normalizar(f) for d, f in fuente_de.items()
                    if len(normalizar(f).split()) >= MIN_TOKENS}
    print(f"{len(fuente_de)} documentos en los pools, "
          f"{len(informativos)} con nombre informativo "
          f"({len(informativos)/len(fuente_de):.1%})")
    docs = sorted(informativos)
    enc = get_encoder(name=MINILM)
    vecs = np.asarray(enc.encode_passages([informativos[d] for d in docs]),
                      dtype="float32")
    cos_por_q = {}
    for q, texto in expandidas.items():
        qv = np.asarray(enc.encode_query(texto), dtype="float32")
        cos_por_q[q] = dict(zip(docs, map(float, vecs @ qv)))
    del enc, vecs

    celdas, archivos = {}, {}

    def registrar(nombre, cos_ok, detector=None, umbral=None):
        res = {q: resultado(q, h, expandidas[q],
                            cos=cos_por_q[q] if cos_ok else None,
                            detector=detector, umbral=umbral)
               for q, h in hits_por_q.items()}
        archivos[nombre] = res
        with (args.out_dir / f"{nombre}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        celdas[nombre] = (metricas(res, gt_todo) + metricas(res, gt_indep)
                          + metricas(res, gt_hum))
        malos = [q for q, r in res.items()
                 if len(r["documents"]) != 3 or len(r["fragments"]) != 10]
        if malos:
            print(f"  !! {nombre}: {malos}")
        return res

    registrar(BASE, False)
    for nombre, (det, umb) in CELDAS.items():
        registrar(nombre, True, detector=det, umbral=umb)

    print(f"\n{'celda':20s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in celdas:
        print(f"{k + (' *' if k == BASE else ''):20s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in celdas[k]))
    print("\n  * = base; tiene que dar 0.455/0.516/0.499 y 0.433/0.474/0.467\n")

    base = celdas[BASE]
    orden_q = [g["query_id"] for g in gt_todo]
    f1_base = dict(zip(orden_q, base[0]))
    unos = [q for q in orden_q if f1_base[q] >= 1.0 - 1e-9]
    ceros = [q for q in orden_q if f1_base[q] <= 1e-9]
    print(f"consultas con F1@3=1.00 hoy ({len(unos)}): {unos}")
    print(f"consultas con F1@3=0.00 hoy ({len(ceros)}): {ceros}\n")

    base_ag = metricas(archivos[BASE], gt_ag)
    for k in celdas:
        if k == BASE:
            continue
        n_disp = sum(1 for r in archivos[k].values() if r["disparo"])
        docs_cambian = sum(1 for q in archivos[k]
                           for a, b in zip(archivos[k][q]["documents"],
                                           archivos[BASE][q]["documents"])
                           if a["doc_id"] != b["doc_id"])
        print(f"  === {k} ===  dispara en {n_disp}/50 consultas, "
              f"{docs_cambian}/150 documentos cambian")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(celdas[k][j], base[j])]
            media, lo, hi = bootstrap_delta(deltas)
            g_ = sum(1 for d in deltas if d > 1e-9)
            p_ = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g_}g/{p_}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ag = metricas(archivos[k], gt_ag)
        print(f"  DESGLOSE agente(9): F1 {sum(base_ag[0])/len(base_ag[0]):.3f} -> "
              f"{sum(ag[0])/len(ag[0]):.3f}   humanas(41): "
              f"{sum(base[6])/len(base[6]):.3f} -> {sum(celdas[k][6])/len(celdas[k][6]):.3f}")
        d_f1 = {q: x - y for q, x, y in zip(orden_q, celdas[k][0], base[0])}
        gana = [f"{q}({f1_base[q]:.2f}->{f1_base[q]+d_f1[q]:.2f})"
                for q in orden_q if d_f1[q] > 1e-9]
        pierde = [f"{q}({f1_base[q]:.2f}->{f1_base[q]+d_f1[q]:.2f})"
                  for q in orden_q if d_f1[q] < -1e-9]
        print(f"  F1 gana {len(gana)}: {gana}")
        print(f"  F1 pierde {len(pierde)}: {pierde}")
        v_a = [q for q in unos if d_f1[q] < -1e-9]
        v_b = [q for q in ceros if d_f1[q] > 1e-9]
        print(f"  VETO (a) baja alguna de F1=1.00: {v_a or 'no'}")
        print(f"  VETO (b) sube alguna de F1=0.00: {v_b or 'no'}  <- se quiere que SI")
        print(f"  VETO (c) dispara en casi todas: "
              f"{'SI (' + str(n_disp) + '/50)' if n_disp >= 40 else 'no'}")
        for q in ("q003", "q037", "q041", "q020", "q032"):
            print(f"    {q}: {f1_base.get(q, float('nan')):.3f} -> "
                  f"{f1_base.get(q, float('nan')) + d_f1.get(q, 0):.3f}")
        print()


if __name__ == "__main__":
    main()
