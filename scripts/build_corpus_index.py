#!/usr/bin/env python3
"""Construccion OFFLINE de la base de conocimiento vectorial (sec. 6): corpus
crudo -> extraccion -> limpieza -> chunking -> encoder -> indice FAISS +
metadata.jsonl, y opcionalmente el grafo de conocimiento (sec. 7, bonus).

Separado de `Entrega/generador.py`, que solo hace la parte ONLINE
(recuperacion sobre un indice ya construido) -- mismo patron OFFLINE/ONLINE
que el notebook 02_rag.ipynb de material de apoyo.

Uso:
    python scripts/build_corpus_index.py
    python scripts/build_corpus_index.py --with-graph
    python scripts/build_corpus_index.py --use-fake-encoder   # sin red, solo para probar la mecanica del pipeline
    python scripts/build_corpus_index.py --corpus-dir otra_carpeta/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CORPUS_DIR, ENCODER_PRIMARY_NAME, GRAFO_PATH  # noqa: E402
from src.embedding.build_index import build_and_persist  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.ingestion.doc_id import load_doc_id_manifest  # noqa: E402
from src.ingestion.pipeline import build_corpus_chunks, write_chunks_jsonl  # noqa: E402

logger = logging.getLogger("build_corpus_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument(
        "--use-fake-encoder",
        action="store_true",
        help="Usa el encoder determinista sin red (src/embedding/encoders.py:HashingFakeEncoder). "
        "SOLO para probar la mecanica del pipeline; no produce embeddings semanticamente validos.",
    )
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="Ademas del indice vectorial, construye y exporta el grafo de conocimiento (bonus, sec. 7).",
    )
    parser.add_argument(
        "--doc-id-manifest",
        type=Path,
        default=None,
        help="Archivo (JSON/JSONL/CSV) con el mapeo {archivo: doc_id} que entregue ADL. "
        "ADL aclaro que el ground truth se empareja por SU doc_id, no por uno propio: "
        "pasar este archivo con el corpus real. Sin el, se usa un hash del contenido "
        "(suficiente para el corpus de ejemplo, pero NO empareja con el ground truth).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--graph-out-path",
        type=Path,
        default=GRAFO_PATH,
        help="Ruta del grafo.graphml de salida (solo con --with-graph). "
        "OJO: el valor por defecto es la ruta oficial de entrega -- al probar con "
        "--use-fake-encoder, redirigir a una carpeta de prueba para no sobrescribirla.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    encoder = get_encoder(name=args.encoder_name, use_fake=args.use_fake_encoder)
    logger.info(
        "encoder: %s (dim=%d)%s",
        encoder.name,
        encoder.dim,
        " [FAKE -- solo pruebas, no usar para la entrega final]" if args.use_fake_encoder else "",
    )

    doc_id_manifest = load_doc_id_manifest(args.doc_id_manifest) if args.doc_id_manifest else None

    records = build_corpus_chunks(
        corpus_dir=args.corpus_dir,
        count_tokens=encoder.count_tokens,
        doc_id_manifest=doc_id_manifest,
    )
    if not records:
        logger.error("no se genero ningun chunk a partir de %s", args.corpus_dir)
        sys.exit(1)
    logger.info("total de chunks generados: %d", len(records))

    write_chunks_jsonl(records)

    out_dir = build_and_persist(records, encoder, out_dir=args.out_dir)
    logger.info("indice FAISS y metadata escritos en: %s", out_dir)

    if args.with_graph:
        from src.graph.build_graph import build_knowledge_graph, export_graphml

        logger.info("construyendo grafo de conocimiento (NER + relaciones heuristicas)...")
        graph = build_knowledge_graph(records)
        export_graphml(graph, args.graph_out_path)
        logger.info(
            "grafo exportado (%d nodos, %d aristas) en: %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            args.graph_out_path,
        )


if __name__ == "__main__":
    main()
