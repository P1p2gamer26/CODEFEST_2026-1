#!/usr/bin/env python3
"""Puede una arquitectura de DOS encoders superar a uno solo?

La fusion RRF simetrica a nivel de chunk ya se midio y pierde (0,306 -> 0,268).
La razon esta medida: MiniLM y e5 comparten apenas el 11,3% de los documentos
del top-3 y el 6,2% de los fragmentos del top-10, y RRF con ese desacuerdo no
fusiona, intercala. Este script prueba disenos que NO dependen del acuerdo:

  cascada:W   e5 re-puntua los candidatos de MiniLM en vez de aportar los
              suyos (src/retrieval/rerank.py). Conserva el recall del primario.
  doc_rrf     RRF de los rankings de DOCUMENTO, no de chunks: la senal de
              acuerdo es casi el doble (11,3% contra 6,2%).
  doc_rrf_pes RRF de documentos dando la mitad de autoridad a e5.

Lo que decide es el conteo de victorias por consulta con prueba de signos, no
el promedio -- con 41 consultas cada una pesa 0,024.

OJO: carga los dos indices a la vez, ~3 GB de RAM. El indice de e5 vive en
dev/intermedios/base_prueba/ y no hace falta reconstruirlo.

Uso:
    python dev/scripts/barrido_dos_encoders.py
    python dev/scripts/barrido_dos_encoders.py --sin-pooling
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DEV_DIR,
    ENCODER_PRIMARY_NAME,
    ENCODER_SECONDARY_NAME,
    N_DOCUMENTS_PER_QUERY,
)
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.rerank import rerank_por_segundo_encoder, verificar_alineacion  # noqa: E402
from src.retrieval.search import search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mini import f1, veredicto_signos  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
INDEX_SECUNDARIO = DEV_DIR / "intermedios" / "base_prueba" / f"encoder_{ENCODER_SECONDARY_NAME}"

# Fijados A PRIORI, no calibrados contra las 41 consultas.
PESOS = [0.25, 0.5, 1.0]

# Cuantos candidatos genera MiniLM ANTES de que e5 los re-puntue. Con 60 la
# cascada solo puede reordenar lo que ya entraba al pool, o sea que apenas
# cambia nada; con 200 e5 puede ascender un documento que MiniLM dejo en el
# puesto 150. Profundizar la generacion de candidatos es el diseno estandar de
# una cascada, no un parametro ajustado a estas 41 consultas.
PROFUNDIDADES = [200]


def docs_de(hits, k_pool: int) -> list[str]:
    return [
        d.doc_id
        for d in aggregate_documents(hits[:k_pool], top_n=N_DOCUMENTS_PER_QUERY, strategy="sum")
    ]


def rrf_de_documentos(listas_de_hits, k_pool: int, pesos=None) -> list[str]:
    """RRF sobre los rankings de DOCUMENTO de cada encoder.

    A diferencia de fusionar chunks, aqui basta que los dos encoders coincidan
    en el documento aunque cada uno haya llegado por un fragmento distinto, que
    es justo lo que pasa con este par (11,3% de acuerdo en documentos contra
    6,2% en fragmentos).

    `pesos` repite la lista de un encoder para darle mas autoridad; es la forma
    de ponderar RRF sin tocar la funcion, que solo suma 1/(k0+rank).
    """
    rankings = []
    for lista, veces in zip(listas_de_hits, pesos or [1] * len(listas_de_hits)):
        docs = aggregate_documents(lista[:k_pool], top_n=k_pool, strategy="sum")
        rankings.extend([docs] * veces)
    fusion = reciprocal_rank_fusion(rankings, key=lambda d: d.doc_id)
    return [d.doc_id for d, _ in fusion[:N_DOCUMENTS_PER_QUERY]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--k-pool", type=int, default=60, help="el de la entrega")
    parser.add_argument("--index-secundario", type=Path, default=INDEX_SECUNDARIO)
    parser.add_argument(
        "--sin-pooling",
        action="store_true",
        help="Solo las 10 consultas de anotacion independiente. OBLIGATORIO al "
        "comparar encoders entre si: una consulta anotada sobre los candidatos que "
        "propuso un encoder favorece a ese encoder.",
    )
    args = parser.parse_args()

    gt = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                fila = json.loads(linea)
                if args.sin_pooling and fila.get("pool"):
                    continue
                gt[fila["query_id"]] = set(fila["docs_relevantes"])

    consultas = {}
    with CONSULTAS_PATH.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                c = json.loads(linea)
                if c["query_id"] in gt:
                    consultas[c["query_id"]] = c["text"]

    enc_a = get_encoder(name=ENCODER_PRIMARY_NAME)
    index_a, meta_a = load_index(ENCODER_PRIMARY_NAME)
    enc_b = get_encoder(name=ENCODER_SECONDARY_NAME)
    index_b, meta_b = load_index(ENCODER_SECONDARY_NAME, index_dir=args.index_secundario)

    # Antes de nada: la cascada lee el vector del chunk por su FILA en el otro
    # indice. Si los dos indices no describen los mismos chunks en el mismo
    # orden, re-puntuaria el chunk equivocado y los resultados serian
    # plausibles pero mal fundados.
    verificar_alineacion(meta_a, meta_b)
    print(f"indices alineados: {index_a.ntotal} chunks, {len(gt)} consultas anotadas\n")

    kp = args.k_pool
    profundidades = [kp] + [n for n in PROFUNDIDADES if n != kp]
    nombres = (
        ["minilm"]
        + [f"casc:{w}@{n}" for n in profundidades for w in PESOS]
        + ["doc_rrf", "doc_rrf_pes"]
    )
    docs_por_variante: dict[str, dict[str, list[str]]] = defaultdict(dict)

    # El indice es exacto (IndexFlatIP), asi que los primeros kp hits de una
    # busqueda de 200 son exactamente los de una busqueda de kp: se recupera
    # una sola vez a la profundidad mayor y se recortan prefijos.
    k_max = max(profundidades)
    for qid, texto in consultas.items():
        hits_a = search(texto, enc_a, index_a, meta_a, k=k_max)
        hits_b = search(texto, enc_b, index_b, meta_b, k=kp)
        q_b = enc_b.encode_query(texto)

        docs_por_variante["minilm"][qid] = docs_de(hits_a, kp)
        for n in profundidades:
            for w in PESOS:
                recolocados = rerank_por_segundo_encoder(hits_a[:n], index_b, q_b, peso=w)
                docs_por_variante[f"casc:{w}@{n}"][qid] = docs_de(recolocados, kp)
        docs_por_variante["doc_rrf"][qid] = rrf_de_documentos([hits_a, hits_b], kp)
        docs_por_variante["doc_rrf_pes"][qid] = rrf_de_documentos([hits_a, hits_b], kp, [2, 1])

    puntajes = {
        n: {qid: f1(docs, gt[qid])[2] for qid, docs in docs_por_variante[n].items()}
        for n in nombres
    }

    ancho = max(len(n) for n in nombres) + 1
    print(f"{'consulta':10s} " + " ".join(f"{n:>{ancho}s}" for n in nombres))
    for qid in sorted(consultas):
        print(f"{qid:10s} " + " ".join(f"{puntajes[n][qid]:{ancho}.2f}" for n in nombres))
    print(
        f"\n{'F1@3 medio':10s} "
        + " ".join(f"{sum(puntajes[n].values()) / len(gt):{ancho}.3f}" for n in nombres)
    )

    for retador in nombres[1:]:
        gana_a = gana_b = empate = 0
        for qid in consultas:
            va, vb = puntajes["minilm"][qid], puntajes[retador][qid]
            if abs(va - vb) < 1e-9:
                empate += 1
            elif va > vb:
                gana_a += 1
            else:
                gana_b += 1
        print(f"\nminilm gana en {gana_a}, {retador} gana en {gana_b}, empatan {empate}")
        print(veredicto_signos("minilm", gana_a, retador, gana_b))


if __name__ == "__main__":
    main()
