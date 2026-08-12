#!/usr/bin/env python3
"""Barrido reproducible de mejoras de recuperacion para la rama Thomas.

El experimento separa tres efectos que antes estaban mezclados:

1. calibracion de cosenos entre MiniLM, GTE y E5;
2. contexto local de chunks contiguos, sin alterar el corpus ni el ground truth;
3. profundidad del pool y agregacion documental.

FAISS usa IndexFlatIP, por lo que la busqueda base ya es exacta. El objetivo
no es cambiarla por un indice aproximado, sino aprovechar mejor sus candidatos.
La celda CONTROL debe reproducir la entrega antes de aceptar cualquier fila.

Uso:
    python dev/scripts/barrido_thomas.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import faiss
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "dev"
sys.path.insert(0, str(DEV))
sys.path.insert(0, str(DEV / "scripts"))
sys.path.insert(0, str(REPO / "Entrega"))

import generador  # noqa: E402
from eval_mini import cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402
from src.config import encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402


MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
GTE = "gte-multilingual-base"
E5 = "multilingual-e5-base"
ENCODERS = (MINILM, GTE, E5)

CONSULTAS = DEV / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH = DEV / "eval" / "ground_truth_mini.jsonl"

# Igual al rerank_depth de produccion; el control debe ser byte-equivalente
# en candidatos antes de probar cualquier otra transformacion.
PROFUNDIDAD = 200
NORMALIZACIONES = ("raw", "minmax", "robust", "rank")
VECINDADES = ("none", "mean", "max")
PESOS = (
    (0.60, 0.60),
    (1.00, 1.00),
    (1.00, 0.50),
    (0.75, 0.25),
    (0.50, 1.00),
)
K_POOLS = (100, 150)
AGREGACIONES = ("top3", "top5")

PESOS_REFINADOS = tuple(
    (gte, e5)
    for gte in (0.35, 0.45, 0.50, 0.55, 0.65)
    for e5 in (0.80, 0.90, 1.00, 1.10, 1.20)
)


@dataclass(frozen=True)
class Configuracion:
    normalizacion: str
    vecindad: str
    peso_gte: float
    peso_e5: float
    k_pool: int
    agregacion: str

    @property
    def nombre(self) -> str:
        return (
            f"{self.normalizacion}:{self.vecindad}:"
            f"gte{self.peso_gte:.2f}:e5{self.peso_e5:.2f}:"
            f"k{self.k_pool}:{self.agregacion}"
        )


CONTROL = Configuracion("raw", "none", 0.60, 0.60, 100, "top5")


def normalizar(valores: np.ndarray, modo: str) -> np.ndarray:
    valores = np.asarray(valores, dtype="float64")
    if modo == "raw":
        return valores
    if modo == "minmax":
        lo, hi = float(valores.min()), float(valores.max())
        return (valores - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(valores)
    if modo == "robust":
        lo, hi = np.quantile(valores, (0.05, 0.95))
        if hi - lo <= 1e-12:
            return np.zeros_like(valores)
        return np.clip((valores - lo) / (hi - lo), 0.0, 1.0)
    if modo == "rank":
        orden = np.argsort(-valores, kind="stable")
        rangos = np.empty(len(valores), dtype="float64")
        rangos[orden] = np.arange(len(valores), dtype="float64")
        return 1.0 - rangos / max(1, len(valores) - 1)
    raise ValueError(f"normalizacion desconocida: {modo}")


def score_vecino(
    fila: int,
    doc_id: str,
    consulta: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    centro: float,
    modo: str,
) -> float:
    """Combina el chunk central con sus vecinos inmediatos del mismo documento.

    No inventa texto ni consulta etiquetas. Es una lectura tardia de la ventana
    que el chunking de una oracion de solape ya deja contigua en metadata.
    """
    if modo == "none":
        return centro

    valores = [centro]
    for otra in (fila - 1, fila + 1):
        if 0 <= otra < len(metadata) and metadata[otra].get("doc_id") == doc_id:
            vector = index.reconstruct(otra)
            valores.append(float(np.dot(consulta, vector)))
    if modo == "mean":
        return float(np.mean(valores))
    if modo == "max":
        return max(valores)
    raise ValueError(f"vecindad desconocida: {modo}")


def construir_pools(consultas: list[dict], indices: dict, metadata: list[dict]) -> dict:
    encoders = {nombre: get_encoder(nombre) for nombre in ENCODERS}
    pools = {}
    for numero, consulta in enumerate(consultas, start=1):
        texto = expandir_consulta(consulta["text"])
        hits = search(
            texto,
            encoders[MINILM],
            indices[MINILM],
            metadata,
            k=PROFUNDIDAD,
        )[:PROFUNDIDAD]

        vectores_consulta = {
            nombre: np.asarray(encoders[nombre].encode_query(texto), dtype="float32").reshape(-1)
            for nombre in ENCODERS
        }
        centrales = {}
        for nombre in ENCODERS:
            if nombre == MINILM:
                centrales[nombre] = np.array([h.score for h in hits], dtype="float64")
            else:
                centrales[nombre] = np.array(
                    [
                        float(np.dot(vectores_consulta[nombre], indices[nombre].reconstruct(h.fila)))
                        for h in hits
                    ],
                    dtype="float64",
                )

        scores = {"none": centrales}
        for modo in ("mean", "max"):
            scores[modo] = {
                nombre: np.array(
                    [
                        score_vecino(
                            h.fila,
                            h.doc_id,
                            vectores_consulta[nombre],
                            indices[nombre],
                            metadata,
                            centrales[nombre][i],
                            modo,
                        )
                        for i, h in enumerate(hits)
                    ],
                    dtype="float64",
                )
                for nombre in ENCODERS
            }

        pools[consulta["query_id"]] = {
            "texto": texto,
            "hits": hits,
            "scores": scores,
        }
        print(f"  pools {numero:02d}/{len(consultas)}", end="\r", flush=True)
    print()
    return pools


def resultado_config(pools: dict, config: Configuracion) -> dict:
    resultados = {}
    for qid, pool in pools.items():
        s = pool["scores"][config.vecindad]
        total = normalizar(s[MINILM], config.normalizacion)
        total = total + config.peso_gte * normalizar(s[GTE], config.normalizacion)
        total = total + config.peso_e5 * normalizar(s[E5], config.normalizacion)
        orden = np.argsort(-total, kind="stable")[: config.k_pool]
        hits = [
            replace(pool["hits"][j], rank=i, score=float(total[j]))
            for i, j in enumerate(orden, start=1)
        ]
        resultados[qid] = generador.build_result_object(
            qid,
            hits,
            agg_strategy=config.agregacion,
            texto_consulta=pool["texto"],
        )
    return resultados


def hits_config(pool: dict, config: Configuracion) -> list:
    """Materializa una celda sin construir aun sus fragmentos de salida."""
    s = pool["scores"][config.vecindad]
    total = normalizar(s[MINILM], config.normalizacion)
    total = total + config.peso_gte * normalizar(s[GTE], config.normalizacion)
    total = total + config.peso_e5 * normalizar(s[E5], config.normalizacion)
    orden = np.argsort(-total, kind="stable")[: config.k_pool]
    return [
        replace(pool["hits"][j], rank=i, score=float(total[j]))
        for i, j in enumerate(orden, start=1)
    ]


def f1_documental(pools: dict, config: Configuracion, gt_por_qid: dict) -> dict:
    """Screening barato: reproduce exactamente filtro y top-3, sin NDCG."""
    salida = {}
    for qid, pool in pools.items():
        hits = hits_config(pool, config)
        hits = generador.filtrar_por_fenomeno_dominante(
            hits, umbral=generador.UMBRAL_FENOMENO
        )
        docs = generador.aggregate_documents(
            hits, top_n=3, strategy=config.agregacion
        )
        relevantes = set(gt_por_qid[qid]["docs_relevantes"])
        salida[qid] = f1([d.doc_id for d in docs], relevantes)[2]
    return salida


def metricas_por_consulta(resultados: dict, ground_truth: list[dict]) -> dict:
    salida = {}
    for gt in ground_truth:
        qid = gt["query_id"]
        res = resultados[qid]
        relevantes = set(gt["docs_relevantes"])
        salida[qid] = {
            "f1": f1([d["doc_id"] for d in res["documents"][:3]], relevantes)[2],
            "ndcg": ndcg(res["fragments"], relevantes),
            "ndcgp": ndcg_penalizado(res["fragments"], relevantes),
        }
    return salida


def resumen(valores: dict, qids: list[str]) -> dict:
    n = max(1, len(qids))
    return {
        "f1": sum(valores[q]["f1"] for q in qids) / n,
        "ndcg": sum(valores[q]["ndcg"] for q in qids) / n,
        "ndcgp": sum(valores[q]["ndcgp"] for q in qids) / n,
        "ceros_f1": sum(valores[q]["f1"] == 0 for q in qids),
    }


def objetivo(r: dict) -> float:
    return 0.50 * r["f1"] + 0.30 * r["ndcg"] + 0.20 * r["ndcgp"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEV / "intermedios" / "thomas")
    ap.add_argument(
        "--refinar",
        action="store_true",
        help="Explora una rejilla local alrededor de la mejor familia del barrido base.",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consultas = cargar_jsonl(CONSULTAS)
    gt = [fila for fila in cargar_jsonl(GROUND_TRUTH) if fila.get("docs_relevantes")]
    qids = [fila["query_id"] for fila in gt]
    qids_humanos = [fila["query_id"] for fila in gt if not fila.get("anotador")]
    qids_indep = [
        fila["query_id"]
        for fila in gt
        if not fila.get("pool") and not fila.get("anotador")
    ]
    gt_por_qid = {fila["query_id"]: fila for fila in gt}

    indices = {}
    metadata = None
    for nombre in ENCODERS:
        carpeta = encoder_dir(nombre)
        if not (carpeta / "index.faiss").is_file():
            raise SystemExit(f"falta {carpeta / 'index.faiss'}")
        if metadata is None:
            indices[nombre], metadata = load_index(nombre)
        else:
            indices[nombre] = faiss.read_index(str(carpeta / "index.faiss"))
        if indices[nombre].ntotal != len(metadata):
            raise SystemExit(f"indice desalineado: {nombre}")
        print(f"  {nombre}: {indices[nombre].ntotal:,} vectores")

    cache_pools = args.out_dir / "pools.pkl"
    if cache_pools.is_file():
        print(f"\nCargando pools cacheados: {cache_pools}")
        with cache_pools.open("rb") as fh:
            pools = pickle.load(fh)
    else:
        print("\nConstruyendo una sola pasada de pools y scores...")
        pools = construir_pools(consultas, indices, metadata)
        with cache_pools.open("wb") as fh:
            pickle.dump(pools, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Pools guardados: {cache_pools}")

    normas = ("raw", "minmax") if args.refinar else NORMALIZACIONES
    vecindades = ("none",) if args.refinar else VECINDADES
    pesos = PESOS_REFINADOS if args.refinar else PESOS
    k_pools = (100, 125, 150, 175, 200) if args.refinar else K_POOLS
    agregaciones = ("top4", "top5", "top6") if args.refinar else AGREGACIONES
    configuraciones = [
        Configuracion(norm, vec, pg, pe, k, agg)
        for norm in normas
        for vec in vecindades
        for pg, pe in pesos
        for k in k_pools
        for agg in agregaciones
    ]
    if args.refinar and CONTROL not in configuraciones:
        configuraciones.append(CONTROL)
    print(f"Screening F1 de {len(configuraciones)} configuraciones...")
    screening = []
    for numero, config in enumerate(configuraciones, start=1):
        por_qid = f1_documental(pools, config, gt_por_qid)
        screening.append(
            {
                "config_obj": config,
                "config": config.nombre,
                "f1": sum(por_qid[q] for q in qids) / len(qids),
                "f1_hum": sum(por_qid[q] for q in qids_humanos) / len(qids_humanos),
                "f1_ind": sum(por_qid[q] for q in qids_indep) / len(qids_indep),
                "ceros": sum(por_qid[q] == 0 for q in qids),
            }
        )
        if numero % 40 == 0:
            print(f"  screening {numero:03d}/{len(configuraciones)}", flush=True)

    control_screen = next(f for f in screening if f["config"] == CONTROL.nombre)
    if abs(control_screen["f1"] - 0.455) > 0.002:
        raise SystemExit(
            f"CONTROL F1 INVALIDO: {control_screen['f1']:.3f}, esperado 0.455"
        )

    preseleccion = [
        fila
        for fila in screening
        if fila["f1_hum"] >= control_screen["f1_hum"] - 0.01
        and fila["f1_ind"] >= control_screen["f1_ind"] - 0.01
        and fila["ceros"] <= control_screen["ceros"]
    ]
    preseleccion.sort(key=lambda f: (f["f1"], f["f1_hum"], f["f1_ind"]), reverse=True)
    finalistas = preseleccion[:30]
    if not any(f["config"] == CONTROL.nombre for f in finalistas):
        finalistas.append(control_screen)

    print(f"Evaluando NDCG completo para {len(finalistas)} finalistas...")
    filas = []
    for numero, candidata in enumerate(finalistas, start=1):
        config = candidata["config_obj"]
        res = resultado_config(pools, config)
        valores = metricas_por_consulta(res, gt)
        todos = resumen(valores, qids)
        humanos = resumen(valores, qids_humanos)
        independientes = resumen(valores, qids_indep)
        filas.append({
            "config": config.nombre,
            "todos": todos,
            "humanos": humanos,
            "independientes": independientes,
            "objetivo": objetivo(todos),
        })
        print(f"  NDCG {numero:02d}/{len(finalistas)}", flush=True)

    control = next(f for f in filas if f["config"] == CONTROL.nombre)
    esperado = {"f1": 0.455, "ndcg": 0.516, "ndcgp": 0.499}
    for metrica, valor in esperado.items():
        if abs(control["todos"][metrica] - valor) > 0.002:
            raise SystemExit(
                f"CONTROL INVALIDO: {metrica}={control['todos'][metrica]:.3f}, "
                f"esperado {valor:.3f}"
            )

    base_h = control["humanos"]
    base_i = control["independientes"]
    adoptables = [
        fila
        for fila in filas
        if fila["humanos"]["f1"] >= base_h["f1"] - 0.01
        and fila["humanos"]["ndcg"] >= base_h["ndcg"] - 0.01
        and fila["independientes"]["f1"] >= base_i["f1"] - 0.01
        and fila["independientes"]["ndcg"] >= base_i["ndcg"] - 0.01
        and fila["todos"]["ceros_f1"] <= control["todos"]["ceros_f1"]
    ]
    adoptables.sort(key=lambda f: f["objetivo"], reverse=True)
    filas.sort(key=lambda f: f["objetivo"], reverse=True)

    salida = {
        "control": control,
        "mejores_adoptables": adoptables[:20],
        "mejores_globales": filas[:20],
        "n_configuraciones": len(configuraciones),
        "n_adoptables": len(adoptables),
    }
    (args.out_dir / "barrido_thomas.json").write_text(
        json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if adoptables:
        ganadora = adoptables[0]
        (args.out_dir / "ganadora.json").write_text(
            json.dumps(ganadora, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\nCONTROL")
    print(json.dumps(control, ensure_ascii=False, indent=2))
    print("\nTOP 10 ADOPTABLES")
    for fila in adoptables[:10]:
        t, h, i = fila["todos"], fila["humanos"], fila["independientes"]
        print(
            f"  {fila['config']:55s} "
            f"F1/ND/NDp={t['f1']:.3f}/{t['ndcg']:.3f}/{t['ndcgp']:.3f}  "
            f"hum={h['f1']:.3f}/{h['ndcg']:.3f}  "
            f"ind={i['f1']:.3f}/{i['ndcg']:.3f}  ceros={t['ceros_f1']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
