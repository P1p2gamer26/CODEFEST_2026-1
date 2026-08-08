#!/usr/bin/env python3
"""E10: el gate de bibliografia tambien en la AGREGACION A DOCUMENTO.

HIPOTESIS, escrita antes de medir: descontar del sumatorio `top5` los chunks
que son aparato bibliografico mejora el F1@3, porque hoy esos chunks suman
score entero y pueden meter en el top-3 un documento que solo coincide en su
bibliografia.

JUSTIFICACION MECANICA. `src/retrieval/calidad_chunk.py` existe desde el 4 de
agosto y se usa en UN solo punto: como tercer criterio de orden de los
FRAGMENTOS. `aggregate_documents` no lo consulta, asi que un chunk de
referencias aporta a `top5` exactamente lo mismo que un chunk que responde. Y
el aparato bibliografico es justo donde mas denso es el vocabulario de dominio:
una lista de referencias de un informe sobre otra cosa acumula los terminos de
la consulta sin contener ninguna respuesta. Es ademas el unico experimento de
la tanda que ataca F1@3 en vez de NDCG, y **no ensancha el pool** -- solo
cambia como se puntua lo que el pool ya trajo, asi que escapa al corolario de
E08.

RIESGOS Y MITIGACIONES, fijados ANTES de medir:
  - El detector es una heuristica tipografica sin modelo (sec. 8.3) y puede
    marcar tablas, CSV y los registros de PubMed del AI Index, que son
    documentos legitimos. **Antes de puntuar nada se cuenta cuantos chunks del
    pool marca y de que formato son**; si marca CSV/XLSX en masa, el
    experimento mide otra cosa y se detiene ahi.
  - Excluir del sumatorio puede dejar un documento con CERO chunks puntuables
    y sacarlo del ranking entero: eso es filtrar, no re-puntuar, y es mas
    agresivo que la hipotesis. Por eso la grilla incluye descuentos parciales y
    el veredicto mira si el ganador es `excluir` o `descontar`.
  - Criterio de siempre: IC al 90% del delta pareado excluyendo -0,02 en las
    DOS muestras, y ante empate se conserva la entregada.
  - Regla nueva de E09: la fila base tiene que reproducir digito a digito el
    0.440 / 0.490 / 0.476 que da `eval_mini.py` sobre `Entrega/`, o el barrido
    no se lee.

COMO SE AISLA EL CAMBIO. Solo se toca la agregacion a documento: se envuelve
`generador.aggregate_documents` para que descuente antes de sumar. El orden de
los fragmentos sigue usando los scores originales, asi que lo que se mide es
la hipotesis y no un descuento global.

COSTE: una sola pasada de FAISS, cero codificacion.

Uso:
    python dev/scripts/barrido_biblio_doc_e10.py
"""

import argparse
import collections
import dataclasses
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
from src.retrieval.calidad_chunk import fraccion_aparato  # noqa: E402
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

PROF = 200
K_POOL = 100
AGG = "top5"
PESO = 0.60
UMBRAL = 0.5      # fraccion de aparato a partir de la cual el chunk "es" bibliografia

# (etiqueta, factor). None = la entregada, sin gate. "prop" = descuento
# proporcional por calidad, sin umbral.
CELDAS = ("entregada", "excluir", "desc0.50", "desc0.25", "prop")
BASE = "entregada"

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")

_agregar_original = generador.aggregate_documents


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


def auditar(pools, fraccs) -> None:
    """La mitigacion nº 1: ver QUE marca el detector antes de creerle a nada."""
    por_formato = collections.Counter()
    marcados = collections.Counter()
    for hits in pools.values():
        for h in hits:
            por_formato[h.formato] += 1
            if fraccs[h.chunk_id] >= UMBRAL:
                marcados[h.formato] += 1
    total = sum(por_formato.values())
    tot_m = sum(marcados.values())
    print(f"\nauditoria del detector (umbral {UMBRAL}): {tot_m}/{total} chunks del pool marcados "
          f"({100 * tot_m / total:.1f}%)")
    print(f"  {'formato':12s}{'en pool':>10s}{'marcados':>10s}{'%':>8s}")
    for fmt, n in por_formato.most_common():
        m = marcados[fmt]
        print(f"  {fmt:12.12s}{n:>10d}{m:>10d}{100 * m / n:>7.1f}%")
    print("  Si un formato tabular (csv/xlsx) sale marcado en masa, el experimento\n"
          "  mide otra cosa y hay que parar aca.")


def parche_agregacion(celda, fraccs):
    """Devuelve un reemplazo de `aggregate_documents` que descuenta antes de
    sumar. Solo afecta al ranking de DOCUMENTOS: los hits originales que usan
    los fragmentos no se tocan (se agrega sobre copias)."""

    def agregar(hits, top_n=3, strategy="sum"):
        ajustados = []
        for h in hits:
            f = fraccs[h.chunk_id]
            if celda == "excluir":
                if f >= UMBRAL:
                    continue
                factor = 1.0
            elif celda == "desc0.50":
                factor = 0.5 if f >= UMBRAL else 1.0
            elif celda == "desc0.25":
                factor = 0.25 if f >= UMBRAL else 1.0
            elif celda == "prop":
                factor = 1.0 - f
            else:
                factor = 1.0
            ajustados.append(dataclasses.replace(h, score=h.score * factor))
        return _agregar_original(ajustados, top_n=top_n, strategy=strategy)

    return agregar


def celda_resultados(pools, celda, fraccs):
    generador.aggregate_documents = (
        _agregar_original if celda == BASE else parche_agregacion(celda, fraccs)
    )
    try:
        return {
            qid: generador.build_result_object(qid, hits, agg_strategy=AGG)
            for qid, hits in pools.items()
        }
    finally:
        generador.aggregate_documents = _agregar_original


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
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "biblio_e10")
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

    # Una sola pasada del detector por chunk: el mismo chunk aparece en varios pools.
    fraccs = {}
    for hits in pools.values():
        for h in hits:
            if h.chunk_id not in fraccs:
                fraccs[h.chunk_id] = fraccion_aparato(h.texto)
    auditar(pools, fraccs)

    guardadas = {}
    for celda in CELDAS:
        res = celda_resultados(pools, celda, fraccs)
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        print(f"  {celda} listo", flush=True)

    print(f"\n{'celda':20s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in guardadas:
        print(f"{k + (' *' if k == BASE else ''):20s}"
              + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  * = la configuracion entregada; tiene que dar 0.440 / 0.490 / 0.476\n")

    print("deltas pareados contra la entregada, IC al 90% (criterio: bajo > -0.02):")
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
