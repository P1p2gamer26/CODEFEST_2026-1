#!/usr/bin/env python3
"""E11: el ORDEN de los criterios dentro de `ordenar_para_fragmentos`.

HIPOTESIS, escrita antes de medir: subir el gate de bibliografia de TERCER a
SEGUNDO criterio -- por encima de la prioridad de idioma -- mejora el NDCG@10
penalizado, porque en el orden actual el gate casi nunca llega a actuar.

JUSTIFICACION MECANICA. Hoy se ordena por (1) pertenecer al top-3 de
documentos, (2) idioma legible, (3) no ser bibliografia. Un criterio en tercer
lugar solo desempata entre hits que ya empataron en los dos primeros, o sea
entre fragmentos del MISMO grupo de documento y el MISMO idioma. Como la
alineacion ya concentra el 98% de los fragmentos en el top-3 y el corpus
relevante se reparte en pocos idiomas, el gate se aplica dentro de bloques
chicos, y su efecto medido fue coherente con eso: NDCG binario -0,001 y
penalizado +0,007. La hipotesis es que el techo lo pone la POSICION del
criterio y no su poder de deteccion.

RIESGOS Y MITIGACIONES, fijados ANTES de medir:
  - La metrica de decision es `ndcg_penalizado`, unica de las nuestras que ve
    bibliografia; el binario es casi ciego a este cambio y **no se puede usar
    para rescatar un resultado que el penalizado no sostiene**.
  - Bajar el idioma a tercer criterio puede devolver al top-10 los fragmentos
    ilegibles (ko/ru/ar/zh). **Nuestro NDCG no mide eso.** Por eso se CUENTAN
    aparte, y si suben el cambio se rechaza aunque el penalizado mejore: es la
    leccion 7, un defecto del dato no se cambia por decimales de una metrica
    que no lo ve.
  - Ante empate se conserva el orden entregado.
  - Regla de E09: la fila base tiene que reproducir 0.440 / 0.490 / 0.476.

COSTE: una sola pasada de FAISS; las celdas son reordenamientos del mismo pool.

Uso:
    python dev/scripts/barrido_orden_e11.py
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
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
SECUNDARIOS = (GTE, E5)

PROF, K_POOL, AGG, PESO = 200, 100, "top5", 0.60

# Cada celda es el orden de los criterios. "top" siempre va primero: la
# alineacion es lo unico ya adoptado y medido, y moverla seria otro experimento.
CELDAS = {
    "entregada": ("top", "idioma", "aparato"),
    "biblio-2o": ("top", "aparato", "idioma"),
    "solo-biblio": ("top", "aparato"),
    "posicion": ("top", "idioma", "aparato", "posicion"),
}
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")

_ordenar_original = generador.ordenar_para_fragmentos


def construir_pools(consultas, cache_idx):
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    por_consulta = {}
    for n, c in enumerate(consultas, 1):
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=PROF)[:PROF]
        for sec in SECUNDARIOS:
            enc_s = get_encoder(name=sec)
            idx_s, _ = cache_idx[sec]
            qv = enc_s.encode_query(texto)
            for h in hits:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits.sort(key=lambda h: -h.score)
        recorte = hits[:K_POOL]
        for i, h in enumerate(recorte, 1):
            h.rank = i
        por_consulta[c["query_id"]] = recorte
        print(f"    {n}/{len(consultas)}", end="\r", flush=True)
    print()
    return por_consulta


def parche_orden(criterios):
    """Reemplazo de `ordenar_para_fragmentos` con los criterios en otro orden.

    `posicion` usa `fila`: los tres indices comparten el orden de filas y
    dentro de un documento las filas van en orden de aparicion, asi que una
    fila menor ES una posicion mas temprana. No hace falta parsear el chunk_id.
    """

    def ordenar(hits, doc_ids_prioritarios=None, priorizar_idioma=True, degradar_aparato=True):
        top = set(doc_ids_prioritarios or ())

        def clave(h):
            partes = []
            for c in criterios:
                if c == "top":
                    partes.append(1 if (top and h.doc_id not in top) else 0)
                elif c == "idioma":
                    partes.append(0 if h.idioma in generador.IDIOMAS_LEGIBLES else 1)
                elif c == "aparato":
                    partes.append(
                        1 if generador.fraccion_aparato(h.texto) >= generador.UMBRAL_APARATO else 0
                    )
                elif c == "posicion":
                    partes.append(h.fila)
            return tuple(partes)

        return sorted(hits, key=clave)

    return ordenar


def celda_resultados(pools, celda):
    generador.ordenar_para_fragmentos = (
        _ordenar_original if celda == BASE else parche_orden(CELDAS[celda])
    )
    try:
        return {
            qid: generador.build_result_object(qid, hits, agg_strategy=AGG)
            for qid, hits in pools.items()
        }
    finally:
        generador.ordenar_para_fragmentos = _ordenar_original


def contar_ilegibles(res, pools) -> int:
    """La cifra que el NDCG no ve y que puede vetar el cambio (riesgo nº 2)."""
    idioma = {h.chunk_id: h.idioma for hits in pools.values() for h in hits}
    return sum(
        1
        for r in res.values()
        for fr in r["fragments"]
        if idioma.get(fr["chunk_id"], "es") not in generador.IDIOMAS_LEGIBLES
    )


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "orden_e11")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas\n")

    cache_idx, metadata = {}, None
    for nombre in (MINILM, *SECUNDARIOS):
        d = encoder_dir(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        print(f"  {nombre}: {cache_idx[nombre][0].ntotal:,} vectores", flush=True)

    print(f"\nrecuperando {PROF} y re-puntuando (peso={PESO})...", flush=True)
    pools = construir_pools(consultas, cache_idx)

    guardadas, ilegibles = {}, {}
    for celda in CELDAS:
        res = celda_resultados(pools, celda)
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        ilegibles[celda] = contar_ilegibles(res, pools)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS) + f"{'ilegibles':>11s}")
    for k in guardadas:
        print(f"{k + (' *' if k == BASE else ''):16s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k])
              + f"{ilegibles[k]:>11d}")
    print("\n  * = la entregada; tiene que dar 0.440 / 0.490 / 0.476")
    print("  ilegibles = fragmentos de 500 en idioma que el evaluador no lee.")
    print("  Si suben respecto de la entregada, el cambio se rechaza aunque el")
    print("  NDCG penalizado mejore (leccion 7: nuestro NDCG no mide esto).\n")

    print("deltas pareados contra la entregada, IC al 90% (criterio: bajo > -0.02).")
    print("LA METRICA DE DECISION ES NDp, fijada antes de medir:")
    base = guardadas[BASE]
    for k in guardadas:
        if k == BASE:
            continue
        print(f"\n  === {k} ===")
        for j, nombre in enumerate(COLS):
            deltas = [x - y for x, y in zip(guardadas[k][j], base[j])]
            media, bajo, alto = bootstrap_delta(deltas)
            g = sum(1 for d in deltas if d > 1e-9)
            p = sum(1 for d in deltas if d < -1e-9)
            print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {g}g/{p}p  {'pasa' if bajo > -0.02 else 'no pasa'}")


if __name__ == "__main__":
    main()
