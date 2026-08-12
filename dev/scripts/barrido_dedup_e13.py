#!/usr/bin/env python3
"""E13: deduplicar fragmentos casi identicos entre los 10 entregados.

HIPOTESIS, pre-registrada en dev/experimentos/cola.jsonl: entre los 10
fragmentos entregados hay pares de texto casi identico, y reemplazarlos por el
siguiente candidato del pool mejora el NDCG@10 real; ademas es un defecto del
dato, no una mejora de metrica.

JUSTIFICACION MECANICA (escrita antes de medir): el chunker solapa oraciones
entre chunks consecutivos POR DISENO (CHUNK_OVERLAP_SENTENCES) y esta medido
que el 32% de los pares consecutivos comparte sus ultimas 25 palabras. E22
alinea el 98% de los fragmentos con el top-3, o sea que dos de los 10 sean
chunks vecinos del mismo documento es probable por construccion. Un cupo
gastado en texto ya entregado vale cero en NDCG@10.

RIESGO, de la cola: (1) el umbral es un parametro continuo -- por eso solo
tres valores gruesos y ante empate gana el que menos fragmentos mueve; (2) el
reemplazo puede venir de fuera del top-3 y deshacer E22, hay que reportarlo;
(3) si gana solo en las 9 de etiqueta asistida, se rechaza.

Es un REORDENAMIENTO del pool ya re-puntuado: no toca vectores ni FAISS.

Regla de E09: la fila base tiene que reproducir 0.4547 / 0.5157 / 0.4990.
NO toca Entrega/.

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_dedup_e13.py
"""

import re
import unicodedata

SHINGLE = 5  # palabras por shingle; fijado antes de medir, no se retoca


def _shingles(texto: str) -> set:
    """Conjunto de n-gramas de SHINGLE palabras, normalizado sin acentos."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    pal = re.findall(r"[a-z0-9]+", t)
    if len(pal) < SHINGLE:
        return {" ".join(pal)} if pal else set()
    return {" ".join(pal[i:i + SHINGLE]) for i in range(len(pal) - SHINGLE + 1)}


def solapamiento(a: str, b: str) -> float:
    """Jaccard sobre shingles de 5 palabras. 1.0 = identico, 0.0 = disjunto."""
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedup(frags: list, umbral: float, reserva: list) -> list:
    """Reemplaza cada fragmento que solape >= umbral con uno ya aceptado.

    Devuelve SIEMPRE la misma cantidad de fragmentos que entraron (sec. 9.2
    exige 10). Si no hay reemplazo limpio en la reserva, conserva el duplicado.
    """
    salida = []
    cola = list(reserva)
    for f in frags:
        if any(solapamiento(f["text"], v["text"]) >= umbral for v in salida):
            rep = None
            while cola:
                cand = cola.pop(0)
                if all(solapamiento(cand["text"], v["text"]) < umbral for v in salida):
                    rep = cand
                    break
            salida.append(rep if rep is not None else f)
        else:
            salida.append(f)
    return salida


# --- a partir de aca, el barrido (Step 5) ---

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.aggregate import aggregate_documents, filtrar_por_fenomeno_dominante  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import Hit  # noqa: E402
from src.retrieval.truncate import enforce_word_limit, ordenar_para_fragmentos, tokens_de  # noqa: E402

from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
AGG = "top5"
# E32, ya adoptado y entregado: post-filtrado del pool por fenomeno dominante
# antes de agregar a documento. Sin esto la fila "base" de este arnes NO
# reproduce la entrega (regla de E09).
UMBRAL_FENOMENO = 0.8
# Tres valores gruesos, fijados antes de medir. Ante empate gana el mas alto
# (el que menos fragmentos mueve).
CELDAS = {"base": None, "u0.5": 0.5, "u0.7": 0.7, "u0.9": 0.9}
# Cuantos candidatos extra de reserva se ofrecen al reemplazo (mas alla de
# los 10 entregados), tomados del mismo pool ya ordenado.
RESERVA_MAX = 40


def hits_desde_pool(pool: list) -> list:
    return [
        Hit(rank=c["rank"], score=c["score"], chunk_id=c["chunk_id"], doc_id=c["doc_id"],
            fuente=c["fuente"], texto=c["texto"], formato=c["formato"],
            fenomeno=c["fenomeno"], idioma=c["idioma"], fila=c["fila"])
        for c in pool
    ]


def resultado(qid, hits, texto_consulta, umbral):
    hits = filtrar_por_fenomeno_dominante(hits, umbral=UMBRAL_FENOMENO)
    doc_hits = aggregate_documents(hits, top_n=3, strategy=AGG)
    top_ids = [d.doc_id for d in doc_hits]
    toks = tokens_de(texto_consulta)
    ordenados = ordenar_para_fragmentos(hits, top_ids, tokens_consulta=toks)
    # Igual que el camino entregado (barrido_fenomeno_topm_e32_e33.py): se le
    # pasa la lista completa, no un recorte a 10 -- enforce_word_limit puede
    # necesitar mas de 10 hits de entrada si alguno se sub-fragmenta o se
    # deduplica por texto exacto internamente.
    frags = enforce_word_limit(ordenados)
    if umbral is not None:
        usados = {f["chunk_id"] for f in frags}
        resto = [h for h in ordenados if h.chunk_id not in usados]
        reserva = enforce_word_limit(resto, max_fragments=RESERVA_MAX)
        frags = dedup(frags, umbral, reserva)
        for i, f in enumerate(frags, start=1):
            f["rank"] = i
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


def main():
    config, pools = cargar_pools()
    print("config del volcado:", config)
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    gt = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    # Las 10 independientes: sin campo "pool" (no anotadas por pooling) y sin
    # etiqueta asistida. Mismo criterio que barrido_orden_e22_e23.py.
    gt_ind = [g for g in gt if not g.get("pool") and not g.get("anotador")]

    todo = {}
    for nombre, umbral in CELDAS.items():
        res = {}
        for qid, pool in pools.items():
            texto = expandir_consulta(consultas[qid])
            res[qid] = resultado(qid, hits_desde_pool(pool), texto, umbral)
        todo[nombre] = res
        f1s, nd, ndp = metricas(res, gt)
        ceros = sum(1 for x in f1s if x == 0.0)
        movidos = 0
        if umbral is not None:
            for qid, r in res.items():
                base = {f["chunk_id"] for f in todo["base"][qid]["fragments"]}
                movidos += len(base - {f["chunk_id"] for f in r["fragments"]})
        print(f"{nombre:8s} F1 {sum(f1s)/len(f1s):.4f}  ND {sum(nd)/len(nd):.4f}  "
              f"NDp {sum(ndp)/len(ndp):.4f}  ceros {ceros}  frags movidos {movidos}")

    # Deltas pareados contra la base, en las dos muestras.
    base_m = metricas(todo["base"], gt)
    base_i = metricas(todo["base"], gt_ind)
    for nombre in CELDAS:
        if nombre == "base":
            continue
        for etiq, muestra, ref in (("50", gt, base_m), ("ind", gt_ind, base_i)):
            cel = metricas(todo[nombre], muestra)
            for j, met in enumerate(("F1", "ND", "NDp")):
                deltas = [x - y for x, y in zip(cel[j], ref[j])]
                media, lo, hi = bootstrap_delta(deltas)
                print(f"  {nombre:8s} {met}({etiq}) {media:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Efecto sobre la alineacion con el top-3 (riesgo 2 pre-registrado):
    # cuantos de los 10 fragmentos entregados vienen de un doc_id fuera del
    # top-3 de documentos declarado en la misma respuesta.
    for nombre in CELDAS:
        res = todo[nombre]
        fuera = sum(
            1 for r in res.values()
            for f in r["fragments"]
            if f["doc_id"] not in {d["doc_id"] for d in r["documents"][:3]}
        )
        print(f"  {nombre:8s} fragmentos fuera del top-3: {fuera}/500")

    Path(DEV_DIR / "intermedios" / "e13_lecturas.json").write_text(
        json.dumps({k: list(v.items()) for k, v in todo.items()}, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
