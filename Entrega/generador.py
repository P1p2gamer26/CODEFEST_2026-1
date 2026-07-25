#!/usr/bin/env python3
"""generador.py -- script de recuperacion ONLINE (sec. 6, sec. 8, sec. 9).

Lee el indice FAISS + metadata ya construidos (ver
`scripts/build_corpus_index.py`, fase OFFLINE), lee el archivo de consultas,
y genera `resultados.jsonl` con exactamente el esquema exigido en la
seccion 9: por cada consulta, los 3 documentos y los 10 fragmentos (<=250
palabras cada uno) mas relevantes.

Ninguna etapa de este script usa un modelo generativo/decoder: solo el
encoder (mismo que en la indexacion), FAISS, y aritmetica sobre
puntuaciones/metadata (sec. 8.3).

Uso:
    python Entrega/generador.py --consultas consultas_prueba/consultas_prueba.jsonl
    python Entrega/generador.py --consultas <archivo> --out Entrega/resultados.jsonl
    python Entrega/generador.py --consultas <archivo> --use-graph
    python Entrega/generador.py --consultas <archivo> --use-fake-encoder   # sin red, solo pruebas

Formato esperado del archivo de consultas (PROVISIONAL -- ver
`load_consultas()`; el formato oficial de q001-q050 aun no lo entrega ADL):
JSON Lines, un objeto por linea, con los campos `query_id` (o `id`) y `text`
(o `query`/`consulta`).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CONSULTAS_PRUEBA_PATH,
    ENCODER_PRIMARY_NAME,
    GRAFO_PATH,
    MAX_FRAGMENT_WORDS,
    N_DOCUMENTS_PER_QUERY,
    N_FRAGMENTS_PER_QUERY,
    RESULTADOS_PATH,
)
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.search import Hit, search  # noqa: E402
from src.retrieval.truncate import enforce_word_limit  # noqa: E402

logger = logging.getLogger("generador")

DEFAULT_K_POOL = 30  # candidatos usados para agregar a nivel documento (sec. 8.6);
# mayor que los 10 fragmentos que se devuelven, para que la relevancia de un
# documento no dependa solo de si su mejor chunk entro en el top-10 mostrado.


def load_consultas(path: Path) -> list[dict]:
    consultas = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_id = obj.get("query_id") or obj.get("id")
            text = obj.get("text") or obj.get("query") or obj.get("consulta")
            if query_id is None or text is None:
                raise ValueError(
                    f"{path}:{line_number}: se esperaban los campos "
                    f"'query_id'/'id' y 'text'/'query'/'consulta'; linea: {obj}"
                )
            consultas.append({"query_id": str(query_id), "text": str(text)})
    return consultas


def build_result_object(
    query_id: str,
    hits: list[Hit],
    top_docs: int = N_DOCUMENTS_PER_QUERY,
    max_fragments: int = N_FRAGMENTS_PER_QUERY,
    max_words: int = MAX_FRAGMENT_WORDS,
    agg_strategy: str = "max",
) -> dict:
    doc_hits = aggregate_documents(hits, top_n=top_docs, strategy=agg_strategy)
    fragments = enforce_word_limit(hits, max_fragments=max_fragments, max_words=max_words)

    return {
        "query_id": query_id,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [
            {
                "rank": f["rank"],
                "chunk_id": f["chunk_id"],
                "doc_id": f["doc_id"],
                "text": f["text"],
            }
            for f in fragments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consultas", type=Path, default=CONSULTAS_PRUEBA_PATH)
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument(
        "--use-fake-encoder",
        action="store_true",
        help="Usa el encoder determinista sin red (solo pruebas de mecanica, no de calidad de recuperacion).",
    )
    parser.add_argument("--index-dir", type=Path, default=None, help="Por defecto Entrega/base_vectorial/encoder_<nombre>")
    parser.add_argument("--out", type=Path, default=RESULTADOS_PATH)
    parser.add_argument("--k-pool", type=int, default=DEFAULT_K_POOL)
    parser.add_argument("--fenomeno", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--formato", default=None)
    parser.add_argument("--idioma", default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--agg-strategy", default="max", choices=["max", "sum", "mean"])
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Fusiona la recuperacion vectorial con el grafo de conocimiento (bonus, sec. 8.5) via RRF.",
    )
    parser.add_argument("--graph-path", type=Path, default=GRAFO_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    encoder = get_encoder(name=args.encoder_name, use_fake=args.use_fake_encoder)
    logger.info(
        "encoder: %s%s", encoder.name, " [FAKE -- solo pruebas]" if args.use_fake_encoder else ""
    )

    index, metadata = load_index(encoder.name, index_dir=args.index_dir)
    logger.info("indice cargado: %d vectores", index.ntotal)
    metadata_by_chunk_id = {m["chunk_id"]: m for m in metadata}

    graph = None
    if args.use_graph:
        import networkx as nx

        graph = nx.read_graphml(args.graph_path)
        logger.info("grafo cargado: %d nodos, %d aristas", graph.number_of_nodes(), graph.number_of_edges())

    consultas = load_consultas(args.consultas)
    logger.info("consultas cargadas: %d", len(consultas))

    resultados = []
    for consulta in consultas:
        vector_hits = search(
            consulta["text"],
            encoder,
            index,
            metadata,
            k=args.k_pool,
            fenomeno=args.fenomeno,
            formato=args.formato,
            idioma=args.idioma,
            min_score=args.min_score,
        )

        hits = vector_hits
        if graph is not None:
            from src.graph.graph_retrieval import graph_search

            query_lang = vector_hits[0].idioma if vector_hits else None
            graph_hits = graph_search(consulta["text"], graph, lang=query_lang, k=args.k_pool)
            if graph_hits:
                fused = reciprocal_rank_fusion([vector_hits, graph_hits], key=lambda h: h.chunk_id)
                rebuilt: list[Hit] = []
                for item, score in fused[: args.k_pool]:
                    meta = metadata_by_chunk_id.get(item.chunk_id)
                    if meta is None:
                        continue
                    rebuilt.append(
                        Hit(
                            rank=0,
                            score=score,
                            chunk_id=meta["chunk_id"],
                            doc_id=meta["doc_id"],
                            fuente=meta["fuente"],
                            texto=meta["texto"],
                            formato=meta["formato"],
                            fenomeno=meta.get("fenomeno"),
                            idioma=meta.get("idioma"),
                        )
                    )
                for i, h in enumerate(rebuilt, start=1):
                    h.rank = i
                hits = rebuilt

        resultados.append(
            build_result_object(
                consulta["query_id"], hits, agg_strategy=args.agg_strategy
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for resultado in resultados:
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")

    logger.info("resultados escritos en: %s (%d lineas)", args.out, len(resultados))


if __name__ == "__main__":
    main()
