#!/usr/bin/env python3
"""E04: multilingual-e5-large como re-puntuador, bajo el regimen vigente.

HIPOTESIS, escrita antes de medir: `multilingual-e5-large` (560M, ventana
512, dim 1024) re-puntua mejor que el par gte+e5-base que se entrega hoy.

JUSTIFICACION MECANICA. Es el encoder mas fuerte que cabe en esta CPU y
comparte familia con e5-base, que ya funciona como re-puntuador. El re-rank
solo lee vectores con `reconstruct(fila)`: una vez existe el indice, probar
todas las combinaciones cuesta una vectorizacion de consulta por encoder y
cero codificacion de pasajes. El indice comparte el orden de filas con los
otros tres (invariante del chunking unico, punto 8 de las notas del proyecto), asi que
`fila` significa el mismo chunk en los cuatro.

QUE SE PRUEBA. e5-large entra como REEMPLAZO y como AGREGADO, porque son dos
hipotesis distintas: reemplazar dice "es mejor que lo que hay", agregar dice
"aporta una senal que los otros no tienen". La entregada va primera y es la
base de todos los deltas.

RIESGO DECLARADO, con las mitigaciones fijadas ANTES de medir:
  - Son 5 celdas contra 50 consultas: la maquina de sobreajuste de la
    leccion 2. Se adopta solo si el IC al 90% del delta pareado excluye una
    perdida de 0,02 en las DOS muestras (50 y 10 independientes), y **ante
    empate se conserva la entregada**, que ya esta construida y validada.
  - El corolario de E08: cualquier ganancia que caiga sobre las 9 consultas
    con etiqueta asistida NO cuenta como evidencia. Por eso se reportan las
    41 humanas y las 10 independientes por separado, y un resultado que gane
    en las 50 pero no en las humanas se lee como sesgo de pooling, igual que
    `doc_rrf` y gte-primario.
  - Sumar un tercer re-puntuador con el mismo peso 0.60 le da al conjunto de
    secundarios 1.8 de autoridad contra 1.0 del primario. Si la celda de tres
    gana, hay que verificar que no sea solo un efecto de peso total antes de
    adoptarla: la comparacion limpia es contra las celdas de dos.

COSTE: una pasada de FAISS mas una vectorizacion de consulta por encoder.
Cero codificacion de pasajes. El indice de e5-large si cuesta ~55 h de CPU y
lo construye `build_corpus_index.py` aparte.

Uso:
    python dev/scripts/barrido_repuntuador_e04.py
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

import generador  # noqa: E402
from eval_mini import bootstrap_delta, cargar_jsonl, f1, ndcg, ndcg_penalizado  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
E5L = "multilingual-e5-large"

# e5-large no vive en Entrega/: su indice es de prueba (punto 13 de las notas del proyecto)
DIR_EXTRA = {E5L: DEV_DIR / "intermedios" / "e5large" / f"encoder_{E5L}"}

PESO = 0.60      # E01
AGG = "top5"     # E07
PROF = 200       # E08: la profundidad quedo inerte, se conserva la entregada
K_POOL = 100

CELDAS = [
    ("entregada: gte+e5", (GTE, E5)),
    ("e5large solo", (E5L,)),
    ("gte+e5large", (GTE, E5L)),
    ("e5large+e5", (E5L, E5)),
    ("gte+e5+e5large", (GTE, E5, E5L)),
]
BASE = CELDAS[0][0]

COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")


def dir_indice(nombre: str) -> Path:
    return DIR_EXTRA.get(nombre) or encoder_dir(nombre)


def evaluar(secundarios, consultas, cache_idx, qvecs):
    """Un pool por consulta, re-puntuado por el conjunto `secundarios`."""
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    res = {}
    for c in consultas:
        texto = expandir_consulta(c["text"])
        hits = search(texto, enc_p, idx_p, metadata, k=PROF)[:PROF]
        for sec in secundarios:
            idx_s = cache_idx[sec][0]
            qv = qvecs[sec][c["query_id"]]
            for h in hits:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        hits = sorted(hits, key=lambda h: -h.score)[:K_POOL]
        for i, h in enumerate(hits, 1):
            h.rank = i
        res[c["query_id"]] = generador.build_result_object(c["query_id"], hits, agg_strategy=AGG)
    return res


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEV_DIR / "intermedios" / "repunt_e04")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_todo = [g for g in cargar_jsonl(GT) if g["docs_relevantes"]]
    gt_indep = [g for g in gt_todo if not g.get("pool") and not g.get("anotador")]
    gt_hum = [g for g in gt_todo if not g.get("anotador")]
    consultas = cargar_jsonl(CONSULTAS)
    print(f"{len(gt_todo)} evaluables, {len(gt_indep)} indep, {len(gt_hum)} humanas\n")

    # Los cuatro indices comparten metadata byte-identica: se carga una vez.
    cache_idx, metadata = {}, None
    for nombre in (MINILM, GTE, E5, E5L):
        d = dir_indice(nombre)
        if not (d / "index.faiss").exists():
            sys.exit(f"falta el indice de {nombre}: {d}")
        if metadata is None:
            idx, metadata = load_index(nombre)
            cache_idx[nombre] = (idx, metadata)
        else:
            cache_idx[nombre] = (faiss.read_index(str(d / "index.faiss")), metadata)
        n = cache_idx[nombre][0].ntotal
        if n != len(metadata):
            sys.exit(f"{nombre}: {n:,} vectores contra {len(metadata):,} de metadata; las filas no se corresponden")
        print(f"  {nombre}: {n:,} vectores", flush=True)

    # Las consultas se vectorizan UNA vez por encoder y se reusan en las 5 celdas.
    qvecs = {}
    for sec in (GTE, E5, E5L):
        enc = get_encoder(name=sec)
        qvecs[sec] = {c["query_id"]: enc.encode_query(expandir_consulta(c["text"])) for c in consultas}
        print(f"  consultas vectorizadas con {sec}", flush=True)

    guardadas = {}
    for nombre, secs in CELDAS:
        res = evaluar(secs, consultas, cache_idx, qvecs)
        salida = args.out_dir / (nombre.split(":")[-1].strip().replace(" ", "_") + ".jsonl")
        with salida.open("w", encoding="utf-8") as f:
            for q in sorted(res):
                f.write(json.dumps(res[q], ensure_ascii=False) + "\n")
        guardadas[nombre] = metricas(res, gt_todo) + metricas(res, gt_indep) + metricas(res, gt_hum)
        print(f"  {nombre}: listo -> {salida.name}", flush=True)

    print(f"\n{'celda':22s}" + "".join(f"{c:>9s}" for c in COLS))
    for nombre in guardadas:
        et = nombre + (" *" if nombre == BASE else "")
        print(f"{et:22s}" + "".join(f"{sum(v)/len(v):>9.3f}" for v in guardadas[nombre]))
    print("\n  * = la configuracion entregada\n")

    print("deltas pareados contra la entregada, IC al 90% (criterio: bajo > -0.02):")
    base = guardadas[BASE]
    for nombre in guardadas:
        if nombre == BASE:
            continue
        print(f"\n  {nombre}")
        for i, col in enumerate(COLS):
            pares = list(zip(guardadas[nombre][i], base[i]))
            d, lo, hi = bootstrap_delta([a - b for a, b in pares])
            gana = sum(1 for a, b in pares if a > b)
            pierde = sum(1 for a, b in pares if a < b)
            marca = "pasa" if lo > -0.02 else "NO"
            print(f"    {col:>8s}  {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {gana}g/{pierde}p  {marca}")


if __name__ == "__main__":
    main()
