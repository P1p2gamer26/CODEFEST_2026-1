#!/usr/bin/env python3
"""Diagnostico: ¿el segundo encoder aporta algo distinto al primero?

Sin el ground truth de ADL no se puede medir NDCG@10 ni F1@3, asi que no
podemos saber cual encoder es "mejor". Pero si podemos responder algo que
igual decide si vale la pena fusionarlos: ¿sus rankings difieren?

Fusionar dos rankings con RRF (sec. 8.4) solo aporta cuando los encoders
cometen errores distintos. Si ambos devuelven casi los mismos fragmentos en
casi el mismo orden, el segundo indice duplica el costo de indexacion y el
tamano de la entrega sin ganar nada, y conviene entregar uno solo.

Lectura de la salida:
  - solapamiento ALTO (>0.8): los encoders coinciden, fusionar aporta poco.
  - solapamiento MEDIO (0.4-0.8): se complementan, es el caso donde RRF suele ayudar.
  - solapamiento BAJO (<0.4): muy distintos; conviene revisar que ambos sean razonables.

Uso:
    python scripts/compare_encoders.py
    python scripts/compare_encoders.py --consultas consultas_prueba/consultas_50.jsonl
    python scripts/compare_encoders.py --use-fake-encoder   # solo mecanica, sin red
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CONSULTAS_PRUEBA_PATH,
    ROOT_DIR,
    ENCODER_PRIMARY_NAME,
    ENCODER_SECONDARY_NAME,
    N_FRAGMENTS_PER_QUERY,
)
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.search import search  # noqa: E402

sys.path.insert(0, str(ROOT_DIR / "Entrega"))
from generador import load_consultas  # noqa: E402

logger = logging.getLogger("compare_encoders")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--consultas", type=Path, default=CONSULTAS_PRUEBA_PATH)
    parser.add_argument(
        "--encoder-name", nargs=2, default=[ENCODER_PRIMARY_NAME, ENCODER_SECONDARY_NAME]
    )
    parser.add_argument("--use-fake-encoder", action="store_true")
    parser.add_argument("--k", type=int, default=N_FRAGMENTS_PER_QUERY)
    parser.add_argument(
        "--index-base",
        type=Path,
        default=None,
        help="Carpeta que contiene las subcarpetas encoder_<nombre>/. Sin esto se "
        "buscan las dos en Entrega/base_vectorial/, y el segundo encoder no esta "
        "ahi: los indices de prueba van a dev/intermedios/ para no mezclarse con "
        "la entrega (ver la nota sobre --out-base en build_corpus_index.py).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    loaded = []
    for name in args.encoder_name:
        enc = get_encoder(name=name, use_fake=args.use_fake_encoder)
        index_dir = args.index_base / f"encoder_{enc.name}" if args.index_base else None
        if index_dir is not None and not index_dir.is_dir():
            parser.error(f"no existe {index_dir}")
        idx, meta = load_index(enc.name, index_dir=index_dir)
        loaded.append((enc, idx, meta))

    consultas = load_consultas(args.consultas)
    name_a, name_b = args.encoder_name

    solapamientos = []
    solapamientos_doc = []
    coincidencias_top1 = 0

    for consulta in consultas:
        hits = [
            search(consulta["text"], enc, idx, meta, k=args.k) for enc, idx, meta in loaded
        ]
        top_a, top_b = ([h.chunk_id for h in hs] for hs in hits)
        docs_a, docs_b = ({h.doc_id for h in hs} for hs in hits)
        if not top_a or not top_b:
            continue

        # Jaccard sobre los conjuntos del top-k: que tanto coinciden en QUE
        # fragmentos traen, independientemente del orden.
        interseccion = len(set(top_a) & set(top_b))
        union = len(set(top_a) | set(top_b))
        solapamientos.append(interseccion / union if union else 0.0)
        # El de DOCUMENTO es el que importa para el F1@3, y es el que hay que
        # mirar a esta escala: con 128.526 fragmentos y k=10, dos encoders
        # buenos pueden elegir chunks distintos DEL MISMO documento, asi que el
        # Jaccard de chunks sale minusculo aunque coincidan en lo esencial.
        union_doc = len(docs_a | docs_b)
        solapamientos_doc.append(len(docs_a & docs_b) / union_doc if union_doc else 0.0)
        if top_a[0] == top_b[0]:
            coincidencias_top1 += 1

    if not solapamientos:
        logger.error("ninguna consulta produjo resultados en ambos indices")
        sys.exit(1)

    promedio = sum(solapamientos) / len(solapamientos)
    logger.info("")
    logger.info("Comparacion de %d consultas (top-%d)", len(solapamientos), args.k)
    logger.info("  A: %s", name_a)
    logger.info("  B: %s", name_b)
    logger.info("")
    promedio_doc = sum(solapamientos_doc) / len(solapamientos_doc)
    logger.info("  solapamiento de FRAGMENTOS (Jaccard): %.3f", promedio)
    logger.info("  minimo / maximo:                      %.3f / %.3f", min(solapamientos), max(solapamientos))
    logger.info("  consultas con el mismo top-1:         %d de %d", coincidencias_top1, len(solapamientos))
    logger.info("  solapamiento de DOCUMENTOS (Jaccard): %.3f", promedio_doc)
    logger.info("")

    # Se interpreta el solapamiento de DOCUMENTOS, no el de fragmentos: con
    # 128.526 fragmentos y k=10 el de fragmentos sale casi siempre minusculo,
    # incluso entre dos encoders buenos que eligen chunks distintos del mismo
    # documento. Interpretarlo llevaba a concluir "revisar que los encoders
    # sean razonables" cuando ambos funcionaban bien.
    if promedio_doc > 0.8:
        logger.info("=> Traen casi los mismos documentos: fusionar aportaria poco.")
        logger.info("   Conviene entregar un solo encoder y ahorrar tiempo y tamano.")
    elif promedio_doc >= 0.3:
        logger.info("=> Se complementan: es el escenario donde RRF puede ayudar.")
        logger.info("   Medirlo con eval_mini.py --comparar-con antes de decidir.")
    else:
        logger.info("=> Traen documentos muy distintos. Antes de fusionar, comprobar que")
        logger.info("   ambos encoders devuelvan resultados sensatos (inspect_results.py)")
        logger.info("   y que los prefijos de E5 se esten aplicando.")
    logger.info("")
    logger.info("Este script NO dice cual es mejor, solo cuanto se parecen.")
    logger.info("Para decidir: eval_mini.py --comparar-con <otro resultados.jsonl> --sin-pooling")


if __name__ == "__main__":
    main()
