#!/usr/bin/env python3
"""Construccion OFFLINE de la base de conocimiento vectorial (sec. 6): corpus
crudo -> extraccion -> limpieza -> chunking -> encoder -> indice FAISS +
metadata.jsonl, y opcionalmente el grafo de conocimiento (sec. 7, bonus).

Separado de `Entrega/generador.py`, que solo hace la parte ONLINE
(recuperacion sobre un indice ya construido) -- mismo patron OFFLINE/ONLINE
que el notebook 02_rag.ipynb de material de apoyo.

La extraccion es REANUDABLE: cada documento se anexa a --chunks-path apenas se
procesa, asi que una corrida interrumpida se retoma donde iba en vez de repetir
casi una hora de extraccion y OCR. Con --desde-chunks se reindexa sin extraer.

Uso:
    python scripts/build_corpus_index.py
    python scripts/build_corpus_index.py --with-graph
    python scripts/build_corpus_index.py --solo-chunks        # solo extraer y fragmentar
    python scripts/build_corpus_index.py --desde-chunks intermedios/chunks_intermedios.jsonl
    python scripts/build_corpus_index.py --use-fake-encoder   # sin red, solo para probar la mecanica del pipeline
    python scripts/build_corpus_index.py --corpus-dir otra_carpeta/
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CHUNKS_INTERMEDIOS_PATH,
    CORPUS_DIR,
    ENCODER_PRIMARY_NAME,
    GRAFO_PATH,
    INTERMEDIOS_DIR,
)
from src.embedding.build_index import build_and_persist  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.ingestion.doc_id import load_doc_id_manifest  # noqa: E402
from src.ingestion.pipeline import (  # noqa: E402
    build_corpus_chunks,
    read_chunks_jsonl,
    write_chunks_jsonl,
)

logger = logging.getLogger("build_corpus_index")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument(
        "--encoder-name",
        nargs="+",
        default=[ENCODER_PRIMARY_NAME],
        help="Uno o varios encoders (sec. 4.4). Con varios se construye un indice "
        "independiente por encoder, cada uno en su carpeta encoder_<nombre>/, y "
        "Entrega/generador.py puede fusionar sus rankings con RRF. El chunking se "
        "hace una sola vez (con el tokenizer del primero) para que los chunk_id "
        "sean identicos entre indices.",
    )
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
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=CHUNKS_INTERMEDIOS_PATH,
        help="Archivo jsonl de chunks. Sirve de CHECKPOINT: la extraccion anexa "
        "cada documento apenas lo procesa y, si se interrumpe, la siguiente "
        "corrida reanuda desde ahi en vez de repetir casi una hora de extraccion "
        "y OCR. Borrarlo fuerza una extraccion limpia.",
    )
    parser.add_argument(
        "--sin-checkpoint",
        action="store_true",
        help="Desactiva el checkpoint incremental (extraccion siempre desde cero).",
    )
    parser.add_argument(
        "--desde-chunks",
        type=Path,
        default=None,
        help="Salta la extraccion y reindexa los chunks de este jsonl. Util para "
        "probar otro encoder sin volver a extraer el corpus.",
    )
    parser.add_argument(
        "--solo-chunks",
        action="store_true",
        help="Extrae y fragmenta, pero no construye indices.",
    )
    parser.add_argument(
        "--solo-grafo",
        action="store_true",
        help="Construye solo el grafo (implica --with-graph) sin re-indexar. El "
        "NER sobre el corpus completo tarda horas, asi que conviene correrlo "
        "aparte de la construccion de los indices.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--out-base",
        type=Path,
        default=None,
        help="Carpeta base donde crear las subcarpetas encoder_<nombre>/. Por defecto "
        "Entrega/base_vectorial (la oficial). Usar SIEMPRE al probar con varios "
        "encoders y --use-fake-encoder, para no escribir indices de prueba dentro "
        "de la entrega.",
    )
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

    if args.out_dir is not None and len(args.encoder_name) > 1:
        logger.error(
            "--out-dir solo vale con un unico --encoder-name (con varios, cada indice "
            "va a su propia carpeta encoder_<nombre>/ como exige la sec. 1.4)"
        )
        sys.exit(2)

    doc_id_manifest = load_doc_id_manifest(args.doc_id_manifest) if args.doc_id_manifest else None

    # Se fragmenta UNA sola vez, con el tokenizer del PRIMER encoder, y esos
    # mismos records se indexan con todos. Si cada encoder fragmentara por su
    # cuenta, los chunk_id colisionarian apuntando a textos distintos y la
    # fusion RRF por chunk_id (sec. 8.4) mezclaria fragmentos que no son el
    # mismo. Ver el docstring de src/retrieval/fusion.py.
    chunks_path = args.desde_chunks
    records: list = []

    if chunks_path is None:
        primero = get_encoder(name=args.encoder_name[0], use_fake=args.use_fake_encoder)
        logger.info("fragmentando con el tokenizer de: %s", primero.name)
        checkpoint = None if args.sin_checkpoint else args.chunks_path
        records = build_corpus_chunks(
            corpus_dir=args.corpus_dir,
            count_tokens=primero.count_tokens,
            doc_id_manifest=doc_id_manifest,
            checkpoint_path=checkpoint,
        )
        del primero  # se recarga en el bucle; la maquina de desarrollo va justa de RAM
        gc.collect()
        if checkpoint is None:
            write_chunks_jsonl(records, args.chunks_path)
        chunks_path = args.chunks_path

    if args.solo_chunks:
        logger.info("--solo-chunks: se omite la construccion de indices")
        return

    # Con checkpoint la extraccion no acumula en RAM: los registros se releen
    # del jsonl aqui, ya sin el tokenizer ni el extractor vivos.
    if not records:
        records = read_chunks_jsonl(chunks_path)
        logger.info("cargados %d chunks desde %s", len(records), chunks_path)

    # Lo que ADL no listo en el manifest no es un documento del reto: son los
    # catalogos y registros del scraping (ceeep_catalogo.json, sipri_full_
    # registro.json, ...). Su doc_id es un hash, o sea que jamas puede
    # aparecer en el ground truth; si se cuelan, solo pueden robarle uno de
    # los 3 cupos de documento a un candidato que si suma F1@3.
    if doc_id_manifest:
        validos = doc_id_manifest.doc_ids
        antes = len(records)
        records = [r for r in records if r.doc_id in validos]
        if antes != len(records):
            logger.info(
                "descartados %d chunks de documentos ajenos al manifest de ADL",
                antes - len(records),
            )

    if not records:
        logger.error("no se genero ningun chunk a partir de %s", args.corpus_dir)
        sys.exit(1)
    logger.info("total de chunks: %d", len(records))

    # Los encoders se cargan de a uno y se liberan: tener dos modelos de
    # transformers vivos a la vez es ~1-2 GB extra y esta maquina va justa.
    for nombre in [] if args.solo_grafo else args.encoder_name:
        enc = get_encoder(name=nombre, use_fake=args.use_fake_encoder)
        logger.info(
            "encoder: %s (dim=%d)%s",
            enc.name,
            enc.dim,
            " [FAKE -- solo pruebas, no usar para la entrega final]" if args.use_fake_encoder else "",
        )
        if args.out_dir is not None:
            target = args.out_dir
        elif args.out_base is not None:
            target = args.out_base / f"encoder_{enc.name}"
        else:
            target = None  # build_and_persist resuelve la ruta oficial de entrega
        # Cache de embeddings por encoder: codificar el corpus son horas, y sin
        # esto una interrupcion obliga a repetirlas enteras.
        cache_path = INTERMEDIOS_DIR / f"emb_{enc.name}.npy"
        out_dir = build_and_persist(records, enc, out_dir=target, cache_path=cache_path)
        logger.info("indice FAISS y metadata escritos en: %s", out_dir)
        del enc
        gc.collect()

    if args.with_graph or args.solo_grafo:
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
