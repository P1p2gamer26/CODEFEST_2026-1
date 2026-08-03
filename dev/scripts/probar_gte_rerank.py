#!/usr/bin/env python3
"""Prueba barata de gte-multilingual-base como re-puntuador de la cascada.

Construir su indice completo son ~4,8 h de CPU. Re-puntuar solo necesita los
vectores de los ~200 candidatos por consulta, o sea ~10.000 chunks: ~25 min.
Si gte no re-puntua mejor que e5-base, no vale la pena el indice completo
(leccion 9 de docs/lecciones_metodologia.md: los experimentos baratos van
primero).

**Corre en TRES FASES separadas, cada una en su propio proceso.** Esta maquina
tiene 7,6 GB de RAM y el indice + la metadata + un modelo de 305 M no caben
juntos. Cada fase escribe a disco y muere; la siguiente lee lo que dejo.

    python dev/scripts/probar_gte_rerank.py pool    # MiniLM: pools -> disco
    python dev/scripts/probar_gte_rerank.py vectores # gte: textos -> vectores
    python dev/scripts/probar_gte_rerank.py evaluar  # re-puntuar y medir

La fase `vectores` es reanudable: graba cada 512 chunks y al relanzarse
retoma. Es la unica larga.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from src.config import DEV_DIR, ENCODER_GTE_NAME, ENCODER_PRIMARY_NAME  # noqa: E402

TRABAJO = DEV_DIR / "intermedios" / "gte_rerank"
POOL_PATH = TRABAJO / "pools.jsonl"
VEC_PATH = TRABAJO / "vectores.npy"
PROGRESO_PATH = TRABAJO / "vectores.progreso"
CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"

RERANK_DEPTH = 200
LOTE = 512


def cargar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------
# Fase 1: pools con MiniLM. Carga el indice grande y lo suelta al terminar.
# --------------------------------------------------------------------------
def fase_pool(args):
    from src.embedding.build_index import load_index
    from src.embedding.encoders import get_encoder
    from src.retrieval.search import search

    TRABAJO.mkdir(parents=True, exist_ok=True)
    encoder = get_encoder(name=ENCODER_PRIMARY_NAME)
    index, metadata = load_index(ENCODER_PRIMARY_NAME)
    print(f"MiniLM: {index.ntotal} vectores", flush=True)

    consultas = cargar_jsonl(CONSULTAS)
    with open(POOL_PATH, "w", encoding="utf-8") as f:
        for c in consultas:
            hits = search(c["text"], encoder, index, metadata, k=RERANK_DEPTH)
            f.write(
                json.dumps(
                    {
                        "query_id": c["query_id"],
                        "text": c["text"],
                        "candidatos": [
                            {
                                "chunk_id": h.chunk_id,
                                "doc_id": h.doc_id,
                                "fila": h.fila,
                                "score": h.score,
                                "texto": h.texto,
                                "idioma": h.idioma,
                                "fuente": h.fuente,
                                "formato": h.formato,
                                "fenomeno": h.fenomeno,
                            }
                            for h in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"  {c['query_id']}: {len(hits)} candidatos", flush=True)

    unicos = {c["chunk_id"] for p in cargar_jsonl(POOL_PATH) for c in p["candidatos"]}
    print(f"\npools escritos: {POOL_PATH}\nchunks unicos a codificar: {len(unicos)}", flush=True)


# --------------------------------------------------------------------------
# Fase 2: vectores de gte. Solo el modelo en memoria, nada del indice.
# --------------------------------------------------------------------------
def _chunks_unicos():
    vistos, orden, textos = set(), [], []
    for p in cargar_jsonl(POOL_PATH):
        for c in p["candidatos"]:
            if c["chunk_id"] not in vistos:
                vistos.add(c["chunk_id"])
                orden.append(c["chunk_id"])
                textos.append(c["texto"])
    return orden, textos


def fase_vectores(args):
    from src.embedding.encoders import get_encoder

    orden, textos = _chunks_unicos()
    print(f"chunks a codificar: {len(orden)}", flush=True)

    hechos = 0
    if PROGRESO_PATH.exists() and VEC_PATH.exists():
        prog = json.loads(PROGRESO_PATH.read_text(encoding="utf-8"))
        # Igual que el checkpoint de build_index: valida CONTENIDO, no solo
        # cantidad. Si el pool cambio, el cache se descarta solo.
        if prog.get("total") == len(orden) and prog.get("ultimo") == orden[prog["hechos"] - 1]:
            hechos = prog["hechos"]
            print(f"reanudando desde {hechos}", flush=True)

    encoder = get_encoder(name=ENCODER_GTE_NAME)
    print(f"gte cargado, dim {encoder.dim}", flush=True)
    vectores = np.load(VEC_PATH) if hechos else np.zeros((len(orden), encoder.dim), dtype="float32")
    if vectores.shape != (len(orden), encoder.dim):
        vectores = np.zeros((len(orden), encoder.dim), dtype="float32")
        hechos = 0

    t0 = time.perf_counter()
    inicio_corrida = hechos
    for i in range(hechos, len(orden), LOTE):
        lote = textos[i : i + LOTE]
        vectores[i : i + len(lote)] = encoder.encode_passages(lote)
        np.save(VEC_PATH, vectores)
        PROGRESO_PATH.write_text(
            json.dumps({"hechos": i + len(lote), "total": len(orden), "ultimo": orden[i + len(lote) - 1]}),
            encoding="utf-8",
        )
        anterior, hechos = hechos, i + len(lote)
        transcurrido = time.perf_counter() - t0
        # El ritmo se mide sobre lo hecho EN ESTA corrida (hechos - inicio),
        # no sobre el total acumulado: al reanudar desde un checkpoint, el
        # tiempo transcurrido no incluye lo que se codifico en la corrida
        # anterior y la estimacion saldria absurdamente optimista.
        #
        # La version anterior de esta linea dividia por una constante (1024)
        # en vez de por lo realmente codificado, asi que el "faltan" CRECIA
        # a cada lote: 10, 18, 24, 29 min. Un ETA que sube es peor que no
        # tener ETA, porque parece informacion.
        del anterior
        hechos_aqui = hechos - inicio_corrida
        ritmo = hechos_aqui / transcurrido if transcurrido > 0 else 0
        faltan = (len(orden) - hechos) / ritmo if ritmo > 0 else 0
        print(
            f"codificados {hechos}/{len(orden)} "
            f"({transcurrido / 60:.1f} min, {ritmo:.1f} chunks/s, faltan ~{faltan / 60:.0f} min)",
            flush=True,
        )
    print("vectores completos", flush=True)


# --------------------------------------------------------------------------
# Fase 3: re-puntuar y medir. Sin modelos, solo aritmetica.
# --------------------------------------------------------------------------
def fase_evaluar(args):
    from eval_mini import cargar_jsonl as _cj  # noqa: F401
    from src.retrieval.aggregate import aggregate_documents
    from src.retrieval.search import Hit

    orden, _ = _chunks_unicos()
    fila_de = {cid: i for i, cid in enumerate(orden)}
    vectores = np.load(VEC_PATH)

    from src.embedding.encoders import get_encoder

    encoder = get_encoder(name=ENCODER_GTE_NAME)

    pools = cargar_jsonl(POOL_PATH)
    salida = TRABAJO / f"res_gte_w{args.peso}.jsonl"
    resultados = []
    for p in pools:
        qv = encoder.encode_query(p["text"])
        hits = []
        for c in p["candidatos"]:
            v = vectores[fila_de[c["chunk_id"]]]
            # Mismo esquema que src/retrieval/rerank.py: el secundario SUMA
            # con peso, no reemplaza. Conserva el recall del primario.
            nuevo = c["score"] + args.peso * float(np.dot(qv, v))
            hits.append(
                Hit(
                    rank=0,
                    score=nuevo,
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    fuente=c["fuente"],
                    texto=c["texto"],
                    formato=c["formato"],
                    fenomeno=c["fenomeno"],
                    idioma=c["idioma"],
                    fila=c["fila"],
                )
            )
        hits.sort(key=lambda h: -h.score)
        hits = hits[:60]  # k_pool de la entrega
        for i, h in enumerate(hits, 1):
            h.rank = i

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))
        import generador  # noqa: E402

        resultados.append(generador.build_result_object(p["query_id"], hits))

    with open(salida, "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"escrito {salida}", flush=True)
    print("\nmedir con:")
    print(f"  python dev/scripts/eval_mini.py --resultados {salida} --solo-humanas --comparar-con Entrega/resultados.jsonl")
    print(f"  python dev/scripts/victorias_ndcg.py {salida} Entrega/resultados.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fase", choices=["pool", "vectores", "evaluar"])
    ap.add_argument("--peso", type=float, default=0.25, help="Peso del re-puntuador (el de la entrega).")
    args = ap.parse_args()
    {"pool": fase_pool, "vectores": fase_vectores, "evaluar": fase_evaluar}[args.fase](args)


if __name__ == "__main__":
    main()
