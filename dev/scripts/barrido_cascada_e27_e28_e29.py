#!/usr/bin/env python3
"""E27 (recorte entre re-puntuadores), E28 (pesos asimetricos), E29 (disparo
condicional por dispersion del top-12).

Los tres son variantes de la ARQUITECTURA de la cascada, no reordenamientos:
cambian que candidatos entran al pool y con que score. Aun asi corren sin
FAISS, porque `pools_entregados.json --con-similitudes` trae los 200
candidatos crudos del primario con el coseno de CADA encoder. El score
entregado se reconstruye exacto:

    score = cos_minilm + 0.60*cos_gte + 0.60*cos_e5  -> top 100 -> agg top5

DECLARACION DE COBERTURA DE DATOS: los campos de texto/idioma/formato no estan
en `crudos`, asi que se leen UNA vez por streaming de `metadata.jsonl` para las
6.314 filas distintas que aparecen en los 50 pools crudos (cache
`dev/intermedios/meta_crudos.json`). O sea que **cualquier candidato de los 200
puede subir al pool y sale con su texto real**: ninguna celda esta limitada por
falta de metadata.

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_cascada_e27_e28_e29.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import Hit  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
POOLS = DEV_DIR / "intermedios" / "pools_entregados.json"
META = DEV_DIR / "intermedios" / "meta_crudos.json"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
GTE = "gte-multilingual-base"
E5 = "multilingual-e5-base"
K_POOL = 100
AGG = "top5"
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)")
BASE = "entregada"


CAMPOS = ("chunk_id", "doc_id", "fuente", "texto", "formato", "fenomeno", "idioma")


def construir_cache(crudos):
    """Lee metadata.jsonl POR STREAMING para las filas de los pools crudos.
    La fila es el numero de linea; se verifica contra el chunk_id volcado."""
    from src.config import ENTREGA_DIR  # import local: solo hace falta aca

    need = {c["fila"]: c["chunk_id"] for q in crudos for c in crudos[q]}
    ruta = ENTREGA_DIR / "base_vectorial" / f"encoder_{MINILM}" / "metadata.jsonl"
    out = {}
    with ruta.open(encoding="utf-8") as f:
        for i, linea in enumerate(f):
            if i in need:
                r = json.loads(linea)
                assert r["chunk_id"] == need[i], f"fila {i} desalineada"
                out[str(i)] = {k: r[k] for k in CAMPOS}
    META.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"cache de metadata: {len(out)} filas -> {META}")
    return out


def cargar():
    d = json.loads(POOLS.read_text(encoding="utf-8"))
    meta = (
        json.loads(META.read_text(encoding="utf-8"))
        if META.exists()
        else construir_cache(d["crudos"])
    )
    return d["config"], d["crudos"], d["similitudes"], meta


def pool_variante(crudos, sims, orden, cortes, pesos):
    """Aplica los re-puntuadores en `orden`, recortando a `cortes[i]` DESPUES
    de cada uno. Devuelve los K_POOL mejores como (fila, score).

    Con un solo corte (= la entregada) el recorte es unico y previo al pool
    final, que es exactamente lo que hace `generador.py`.
    """
    vivos = [(i, sims[MINILM][i]) for i in range(len(crudos))]
    for enc, corte in zip(orden, cortes):
        vivos = [(i, s + pesos[enc] * sims[enc][i]) for i, s in vivos]
        vivos.sort(key=lambda t: -t[1])
        vivos = vivos[:corte]
    return vivos[:K_POOL]


def hits_de(vivos, crudos, meta):
    out = []
    for rank, (i, score) in enumerate(vivos, 1):
        c = crudos[i]
        m = meta[str(c["fila"])]
        out.append(Hit(rank=rank, score=score, chunk_id=m["chunk_id"], doc_id=m["doc_id"],
                       fuente=m["fuente"], texto=m["texto"], formato=m["formato"],
                       fenomeno=m["fenomeno"], idioma=m["idioma"], fila=c["fila"]))
    return out


def resultado(qid, hits, texto_expandido):
    toks = frozenset(tokens_de(texto_expandido))
    doc_hits = aggregate_documents(hits, top_n=3, strategy=AGG)
    top_ids = [d.doc_id for d in doc_hits]
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks)
    )
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [{"rank": f["rank"], "chunk_id": f["chunk_id"], "doc_id": f["doc_id"],
                       "text": f["text"]} for f in frags],
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


def dispersion(sims, n=12):
    """Rango del top-n del PRIMARIO. Plano y bajo = el termino discriminante
    se perdio (CLAUDE.md). Los crudos vienen en orden del primario."""
    s = sims[MINILM][:n]
    return max(s) - min(s)


PESOS_SIM = {GTE: 0.60, E5: 0.60}
PESOS_GTE = {GTE: 0.90, E5: 0.30}   # misma autoridad total (1.20), repartida
PESOS_E5 = {GTE: 0.30, E5: 0.90}


def celdas(crudos, sims, meta, disp):
    """Devuelve {nombre: {qid: [Hit]}}. Las celdas se declaran aca, no en un
    diccionario global, porque E29 necesita el detector por consulta."""
    qs = list(crudos)
    out = {}

    def build(nombre, fn):
        out[nombre] = {q: hits_de(fn(q), crudos[q], meta) for q in qs}

    # base: un solo recorte, ambos re-puntuadores sobre los 200
    build(BASE, lambda q: pool_variante(crudos[q], sims[q], (GTE, E5), (200, 200), PESOS_SIM))

    # --- E27: embudo, recorte intermedio ---
    for n in (120, 150):
        build(f"e27-gte->{n}->e5",
              lambda q, n=n: pool_variante(crudos[q], sims[q], (GTE, E5), (n, 200), PESOS_SIM))
        build(f"e27-e5->{n}->gte",
              lambda q, n=n: pool_variante(crudos[q], sims[q], (E5, GTE), (n, 200), PESOS_SIM))

    # --- E28: pesos asimetricos, autoridad total constante ---
    build("e28-gte>e5 (0.90/0.30)",
          lambda q: pool_variante(crudos[q], sims[q], (GTE, E5), (200, 200), PESOS_GTE))
    build("e28-gte<e5 (0.30/0.90)",
          lambda q: pool_variante(crudos[q], sims[q], (GTE, E5), (200, 200), PESOS_E5))

    # --- E29: disparo condicional. Solo donde el top-12 del primario es plano
    # se aplica la variante gte-pesado; el resto queda igual a la entregada.
    for umbral in (0.05, 0.08):
        disparan = [q for q in qs if disp[q] < umbral]
        out[f"e29-umbral{umbral}"] = {
            q: hits_de(pool_variante(crudos[q], sims[q], (GTE, E5), (200, 200),
                                     PESOS_GTE if q in disparan else PESOS_SIM),
                       crudos[q], meta)
            for q in qs
        }
        print(f"  e29-umbral{umbral}: dispara en {len(disparan)} consultas -> "
              f"{' '.join(sorted(disparan)) or '(ninguna)'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "cascada_e27_e29")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, crudos, sims, meta = cargar()
    print("config del volcado:", config)
    assert config["peso"] == 0.60 and config["profundidad"] == 200 and config["k_pool"] == K_POOL

    consultas = {c["query_id"]: expandir_consulta(c["text"]) for c in cargar_jsonl(CONSULTAS)}
    disp = {q: dispersion(sims[q]) for q in crudos}
    print("dispersion del top-12: min %.3f  mediana %.3f  max %.3f"
          % (min(disp.values()), sorted(disp.values())[len(disp) // 2], max(disp.values())))

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_ind = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_ind)} independientes\n")

    hits = celdas(crudos, sims, meta, disp)
    res = {k: {q: resultado(q, h, consultas[q]) for q, h in v.items()} for k, v in hits.items()}
    med = {k: metricas(v, gt_todo) + metricas(v, gt_ind) for k, v in res.items()}

    print("\n" + f"{'celda':26s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in med:
        print(f"{k + (' *' if k == BASE else ''):26s}"
              + "".join(f"{sum(v) / len(v):>9.3f}" for v in med[k]))
    print("\n  * = la entregada; tiene que dar 0.440 / 0.506 / 0.491 (regla de E09)\n")

    base = med[BASE]
    ceros_base = sum(1 for v in base[0] if v == 0)
    for k in med:
        if k == BASE:
            continue
        lineas = sum(1 for q in res[k] if res[k][q] != res[BASE][q])
        print(f"  === {k} ===  {lineas}/50 lineas cambian")
        for j, nombre in enumerate(COLS):
            d = [x - y for x, y in zip(med[k][j], base[j])]
            m, lo, hi = bootstrap_delta(d)
            g = sum(1 for x in d if x > 1e-9)
            p = sum(1 for x in d if x < -1e-9)
            print(f"  {nombre:9s}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {g}g/{p}p  "
                  f"{'pasa' if lo > -0.02 else 'NO pasa'}")
        ceros = sum(1 for v in med[k][0] if v == 0)
        print(f"  consultas con F1@3=0: {ceros} (entregada: {ceros_base})"
              f"{'  <-- VETO' if ceros > ceros_base else ''}\n")

    for k, v in res.items():
        seguro = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in k)
        with (args.out_dir / f"{seguro}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(v):
                f.write(json.dumps(v[q], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
