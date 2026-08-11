#!/usr/bin/env python3
"""E26 -- extender cada fragmento con oraciones del chunk contiguo.

Los 500 fragmentos entregados usan p50 = 168 palabras de las 250 que la sec.
9.2.1 permite, y NINGUNO llega al tope: se regala un tercio del presupuesto.
La sec. 9.2.1 autoriza concatenar con el fragmento contiguo mientras no se
pase de 250. La sec. 10.2.1 juzga la relevancia sobre el campo `text` con
escala GRADUADA, asi que un pasaje del documento correcto cuya oracion que
responde cayo en el chunk vecino cobra r=1 en vez de r=2.

POR QUE NO ES LO QUE YA SE REVIRTIO. La version vieja concatenaba el VECINO
ENTERO y fallaba por dos propiedades de esa implementacion: el chunker solapa
una oracion entre chunks consecutivos (CHUNK_OVERLAP_SENTENCES=1) y el texto
salia DUPLICADO, y 164+164 = 328 > 250 dejaba entrar solo pares cortos. Aca se
extiende POR ORACIONES y con deduplicacion: lo primero es imposible por
construccion, lo segundo es irrelevante porque se toman las que quepan.

La extension se aplica DESPUES de elegir los 10 fragmentos, asi que no puede
mover ni los documentos ni el orden -- solo anade texto.

No carga ningun indice FAISS. Lee el pool volcado y `metadata.jsonl` por
streaming.

Uso:
    .venv/Scripts/python.exe dev/scripts/barrido_extender_e26.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.chunking.sentence_split import split_sentences  # noqa: E402
from src.config import DEV_DIR, ROOT_DIR, MAX_FRAGMENT_WORDS  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.calidad_chunk import fraccion_aparato  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import Hit  # noqa: E402
from src.retrieval.truncate import (  # noqa: E402
    IDIOMAS_LEGIBLES,
    UMBRAL_APARATO,
    _clave_dedup,
    enforce_word_limit,
    ordenar_para_fragmentos,
    tokens_de,
)

from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from volcar_pools import cargar_pools  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
METADATA = (
    ROOT_DIR / "Entrega" / "base_vectorial"
    / "encoder_paraphrase-multilingual-MiniLM-L12-v2" / "metadata.jsonl"
)
AGG = "top5"
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)")


# --- vecinos ---------------------------------------------------------------

def vecinos_de(chunk_id: str) -> list[str]:
    """chunk_id = <doc_id>-c<posicion:04d>. Siguiente primero, luego anterior:
    el texto que continua se lee mejor pegado que el que precede."""
    base, _, pos = chunk_id.rpartition("-c")
    if not pos.isdigit():
        return []
    p = int(pos)
    return [f"{base}-c{p + 1:04d}"] + ([f"{base}-c{p - 1:04d}"] if p else [])


def leer_textos(ids: set[str], path: Path) -> dict[str, str]:
    """Streaming: metadata.jsonl son 150 MB y solo hacen falta ~10k lineas."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for linea in fh:
            # filtro barato antes de parsear: json.loads de 128k lineas es caro
            i = linea.find('"chunk_id": "')
            if i < 0:
                continue
            cid = linea[i + 13 : linea.find('"', i + 13)]
            if cid in ids:
                out[cid] = json.loads(linea)["texto"]
    return out


def extender(frag: dict, idioma: str, textos: dict[str, str], max_words: int,
             sin_aparato: bool = False) -> dict:
    """Anade oraciones del chunk contiguo hasta el tope, sin repetir ninguna.

    Solo se extiende un fragmento que sea el chunk ENTERO: si `enforce_word_limit`
    tuvo que partirlo, el chunk ya llenaba el presupuesto por si solo.
    """
    palabras = frag["text"].split()
    hueco = max_words - len(palabras)
    if hueco <= 0:
        return frag

    # Dedup por CONTENCION de subcadena sobre el texto ya acumulado, no por
    # igualdad de oracion. Un set de oraciones NO alcanza y se comprobo
    # midiendo: el segmentador no es idempotente cruzando el borde del chunk.
    # En F2-UNOOSA-030 la oracion solapada sale como '23.' + 'Las directrices
    # reflejan...' dentro del chunk y como '23. Las directrices reflejan...'
    # entera en el vecino, asi que la igualdad exacta no la ve y el parrafo
    # salia DOS VECES -- exactamente el fallo que mato a la version de vecino
    # entero. La contencion lo cubre en los dos sentidos porque
    # `enforce_word_limit` une las oraciones con un espacio simple.
    acumulado = _clave_dedup(frag["text"])
    anadidas = []
    for vid in vecinos_de(frag["chunk_id"]):
        texto = textos.get(vid)
        if not texto:
            continue
        for sent in split_sentences(texto, idioma):
            n = len(sent.split())
            clave = _clave_dedup(sent)
            if n > hueco or not clave or clave in acumulado:
                continue
            # Variante NO pre-registrada, medida porque el sistema ya filtra
            # los fragmentos por aparato bibliografico (gate de calidad_chunk)
            # y es incoherente inyectarles aparato al extenderlos.
            if sin_aparato and fraccion_aparato(sent) >= UMBRAL_APARATO:
                continue
            acumulado += " " + clave
            anadidas.append(sent)
            hueco -= n
        if hueco <= 0:
            break

    if not anadidas:
        return frag
    return {**frag, "text": " ".join(palabras + anadidas), "ampliado": len(anadidas)}


# --- arnes -----------------------------------------------------------------

def hits_desde_pool(pool: list[dict]) -> list[Hit]:
    return [
        Hit(rank=c["rank"], score=c["score"], chunk_id=c["chunk_id"], doc_id=c["doc_id"],
            fuente=c["fuente"], texto=c["texto"], formato=c["formato"],
            fenomeno=c["fenomeno"], idioma=c["idioma"], fila=c["fila"])
        for c in pool
    ]


def resultado(qid, hits, toks, textos, ampliar, sin_aparato=False):
    doc_hits = aggregate_documents(hits, top_n=3, strategy=AGG)
    top_ids = [d.doc_id for d in doc_hits]
    idioma = {h.chunk_id: h.idioma for h in hits}
    frags = enforce_word_limit(
        ordenar_para_fragmentos(hits, doc_ids_prioritarios=top_ids, tokens_consulta=toks)
    )
    if ampliar:
        frags = [
            extender(f, idioma.get(f["chunk_id"], "es"), textos, MAX_FRAGMENT_WORDS,
                     sin_aparato)
            for f in frags
        ]
    return {
        "query_id": qid,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [
            {"rank": f["rank"], "chunk_id": f["chunk_id"], "doc_id": f["doc_id"],
             "text": f["text"], "ampliado": f.get("ampliado", 0)}
            for f in frags
        ],
        "_idioma": idioma,
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


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def diagnostico(res, toks_por_q):
    largos, cero, ileg, ampliados, dup, excedidos = [], 0, 0, 0, 0, 0
    for q, r in res.items():
        for fr in r["fragments"]:
            n = len(fr["text"].split())
            largos.append(n)
            if n > MAX_FRAGMENT_WORDS:
                excedidos += 1
            if not (tokens_de(fr["text"]) & toks_por_q[q]):
                cero += 1
            if r["_idioma"].get(fr["chunk_id"], "es") not in IDIOMAS_LEGIBLES:
                ileg += 1
            if fr.get("ampliado"):
                ampliados += 1
            # control anti-duplicacion: ninguna oracion dos veces
            sents = [_clave_dedup(s) for s in split_sentences(fr["text"], "es")]
            sents = [s for s in sents if s]
            if len(sents) != len(set(sents)):
                dup += 1
    return {
        "p50": pct(largos, 0.5), "p90": pct(largos, 0.9), "max": max(largos),
        "total_palabras": sum(largos), "cob0": cero, "ilegibles": ileg,
        "ampliados": ampliados, "dup": dup, "excedidos": excedidos,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "extender_e26")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config, pools = cargar_pools()
    print("config del volcado:", config)
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    toks = {q: frozenset(tokens_de(expandir_consulta(consultas[q]))) for q in pools}

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} independientes")

    hits_por_q = {q: hits_desde_pool(p) for q, p in pools.items()}

    quiero = {v for p in pools.values() for c in p for v in vecinos_de(c["chunk_id"])}
    print(f"buscando {len(quiero)} chunks vecinos en metadata.jsonl (streaming)...")
    textos = leer_textos(quiero, METADATA)
    print(f"  encontrados {len(textos)}\n")

    guardadas, diag, archivos = {}, {}, {}
    for celda, ampliar, sa in (("entregada", False, False), ("e26-extendida", True, False),
                               ("e26-sin-aparato", True, True)):
        res = {q: resultado(q, h, toks[q], textos, ampliar, sa) for q, h in hits_por_q.items()}
        archivos[celda] = res
        with (args.out_dir / f"{celda}.jsonl").open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps({k: v for k, v in res[q].items() if k != "_idioma"},
                                   ensure_ascii=False) + "\n")
        guardadas[celda] = metricas(res, gt_todo) + metricas(res, gt_indep)
        diag[celda] = diagnostico(res, toks)

    print(f"{'celda':16s}" + "".join(f"{c:>9s}" for c in COLS))
    for k in guardadas:
        print(f"{k:16s}" + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[k]))
    print("\n  entregada tiene que dar 0.440 / 0.506 / 0.491 (regla de E09)\n")

    print(f"{'celda':16s}{'p50':>6s}{'p90':>6s}{'max':>6s}{'palabras':>10s}"
          f"{'ampl':>6s}{'cob=0':>7s}{'ileg':>6s}{'DUP':>5s}{'>250':>6s}")
    for k, d in diag.items():
        print(f"{k:16s}{d['p50']:>6d}{d['p90']:>6d}{d['max']:>6d}{d['total_palabras']:>10d}"
              f"{d['ampliados']:>6d}{d['cob0']:>7d}{d['ilegibles']:>6d}{d['dup']:>5d}{d['excedidos']:>6d}")

    base = guardadas["entregada"]
    for celda in [k for k in guardadas if k != "entregada"]:
      print(f"\n=== {celda} vs. entregada ===")
      for j, nombre in enumerate(COLS):
        deltas = [x - y for x, y in zip(guardadas[celda][j], base[j])]
        media, bajo, alto = bootstrap_delta(deltas)
        g = sum(1 for d in deltas if d > 1e-9)
        p = sum(1 for d in deltas if d < -1e-9)
        print(f"  {nombre:9s}: {media:+.3f} [{bajo:+.3f}, {alto:+.3f}]  {g}g/{p}p  "
              f"{'pasa' if bajo > -0.02 else 'NO pasa'}")


    # muestra para lectura humana: los 10 con mas texto anadido
    ext = archivos["e26-extendida"]
    todos = [(q, i, fr) for q, r in ext.items() for i, fr in enumerate(r["fragments"])]
    todos.sort(key=lambda t: -(len(t[2]["text"].split())
                               - len(archivos["entregada"][t[0]]["fragments"][t[1]]["text"].split())))
    muestra = args.out_dir / "muestra.md"
    with muestra.open("w", encoding="utf-8") as f:
        for q, i, fr in todos[:10]:
            antes = archivos["entregada"][q]["fragments"][i]["text"]
            f.write(f"## {q} rank {fr['rank']} — {fr['chunk_id']}\n\n"
                    f"**Consulta:** {consultas[q]}\n\n"
                    f"**Original ({len(antes.split())} palabras):**\n\n> {antes}\n\n"
                    f"**Anadido ({len(fr['text'].split()) - len(antes.split())} palabras, "
                    f"{fr['ampliado']} oraciones):**\n\n> {fr['text'][len(antes):].strip()}\n\n---\n\n")
    print(f"\nmuestra de 10 ampliados -> {muestra}")


if __name__ == "__main__":
    main()
