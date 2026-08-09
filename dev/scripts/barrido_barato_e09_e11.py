#!/usr/bin/env python
"""Barrido barato de E09 (prior de recencia) y E11 (cupo_alineado).

Una sola pasada de FAISS: por consulta se recuperan 200 candidatos con el
primario, se re-puntuan con gte y e5 (peso 0.60) y se recorta a k_pool=100.
Las celdas salen variando SOLO los parametros de build_result_object
(prior_recencia / cupo_alineado), que actuan despues de la recuperacion, asi
que el coste de N celdas es el de 1.

Reproduce el pipeline de Entrega/generador.py sin flags, a proposito: la
celda base tiene que coincidir digito a digito con resultados.jsonl.

Pre-registro (dev/experimentos/cola.jsonl): E09 y E11, hipotesis escritas
antes de medir. Criterio: adoptar solo si el IC 90% del delta pareado excluye
-0.02 EN LAS DOS muestras (50 y 10 independientes).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

import generador  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")

RERANK_WEIGHT = generador.DEFAULT_RERANK_WEIGHT
RERANK_DEPTH = generador.DEFAULT_RERANK_DEPTH
K_POOL = generador.DEFAULT_K_POOL

CELDAS: list[tuple[str, float, int]] = [
    ("e00_baseline", 0.0, 10),
    ("e09_rec_0.05", 0.05, 10),
    ("e09_rec_0.10", 0.10, 10),
    ("e09_rec_0.20", 0.20, 10),
    ("e11_cupo_6", 0.0, 6),
    ("e11_cupo_7", 0.0, 7),
    ("e11_cupo_8", 0.0, 8),
    ("e11_cupo_9", 0.0, 9),
]


def main() -> None:
    consultas_path = (
        Path(__file__).resolve().parents[1] / "consultas_prueba" / "consultas_50_oficiales.jsonl"
    )
    out_dir = Path(__file__).resolve().parents[1] / "intermedios"
    out_dir.mkdir(parents=True, exist_ok=True)

    consultas = generador.load_consultas(consultas_path)
    logging.info("consultas cargadas: %d", len(consultas))

    enc_primario = generador.get_encoder(name=generador.ENCODER_PRIMARY_NAME)
    idx_primario, meta_primario = generador.load_index(enc_primario.name)
    metadata_by_chunk_id = {m["chunk_id"]: m for m in meta_primario}
    logging.info("primario: %s -> %d vectores", enc_primario.name, idx_primario.ntotal)

    reranks = []
    for nombre in generador.DEFAULT_RERANK_ENCODERS:
        enc_r = generador.get_encoder(name=nombre)
        idx_r, meta_r = generador.load_index(enc_r.name)
        generador.verificar_alineacion(meta_primario, meta_r)
        reranks.append((enc_r, idx_r))
        logging.info("re-puntuador: %s", enc_r.name)

    pool_por_consulta: dict[str, list[generador.Hit]] = {}

    for consulta in consultas:
        qid = consulta["query_id"]
        texto_busqueda = generador.expandir_consulta(consulta["text"])
        ranked_lists = [
            generador.search(
                texto_busqueda,
                enc_primario,
                idx_primario,
                meta_primario,
                k=max(K_POOL, RERANK_DEPTH),
            )
        ]
        ranked_lists = [lista[:RERANK_DEPTH] for lista in ranked_lists]
        for enc_r, idx_r in reranks:
            qv = enc_r.encode_query(texto_busqueda)
            ranked_lists = [
                generador.rerank_por_segundo_encoder(lista, idx_r, qv, peso=RERANK_WEIGHT)
                for lista in ranked_lists
            ]
        hits = ranked_lists[0][:K_POOL]
        pool_por_consulta[qid] = hits

    logging.info("pooles listos: %d consultas, %d celdas por escribir", len(pool_por_consulta), len(CELDAS))

    for nombre_celda, rec, cupo in CELDAS:
        resultados = []
        for consulta in consultas:
            qid = consulta["query_id"]
            prior_recencia = rec if generador.tiene_marcador_temporal(consulta["text"]) else 0.0
            resultados.append(
                generador.build_result_object(
                    qid,
                    pool_por_consulta[qid],
                    agg_strategy="top5",
                    alinear_fragmentos=True,
                    priorizar_idioma=True,
                    cupo_alineado=cupo,
                    prior_recencia=prior_recencia,
                )
            )
        out = out_dir / f"{nombre_celda}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for resultado in resultados:
                f.write(
                    generador.json.dumps(resultado, ensure_ascii=False) + "\n"
                )
        logging.info("celda %-14s -> %s (%d lineas)", nombre_celda, out, len(resultados))


if __name__ == "__main__":
    main()
