#!/usr/bin/env python3
"""E25: `BAAI/bge-m3` como re-puntuador, bajo el regimen vigente.

HIPOTESIS, escrita antes de medir: bge-m3 re-puntua mejor que el par gte+e5
que se entrega hoy.

JUSTIFICACION MECANICA. Es el unico candidato de `docs/plan_encoders.md` que
nunca se midio, y el unico que sobrevive a las tres restricciones duras del
proyecto:
  - backbone **encoder-only** (XLM-RoBERTa), o sea sin riesgo bajo la sec. 8.3
    -- que es lo que descarta a Qwen3, Harrier, KaLM/EmbeddingGemma y Jina v5;
  - licencia MIT (la de Jina v5 es CC BY-NC, no comercial);
  - sin prefijos de consulta/pasaje, una fuente de error menos.
Y sobre todo: su backbone es **distinto** al de los tres encoders actuales.
`plan_encoders.md` dice que para la cascada el par debe ser diverso porque dos
modelos de la misma familia se equivocan igual. E04 refuto e5-large, pero
e5-large es la MISMA familia que un re-puntuador ya presente: no refuta un
backbone nuevo.

PUNTO DE CORTE, que es lo que hace barato el experimento. El indice completo
de bge-m3 son ~14 h de GPU y 527 MB. Aca NO se construye: se codifican en
caliente los ~6.000 chunks que son candidatos de alguna de las 50 consultas
(~1 h de GPU) y se re-puntua con eso. **El resultado es fiel a lo que se
entregaria**, porque el re-puntuador solo mira los vectores de los candidatos;
el indice completo serviria para tenerlos por `reconstruct(fila)` en consulta,
no para cambiar el numero. Es el mismo esquema con el que se midio gte antes
de gastar sus 7,3 h. El indice solo se construye si esta medicion gana.

DOS FASES, y no es un capricho: el modelo de bge-m3 mas los tres indices no
entran en 8 GB de RAM (el arnes de E17 muere con `os error 1455`). La fase 1
carga MiniLM + bge y vuelca los vectores a un `.npy`; la fase 2 los lee y
tiene el perfil de memoria del arnes de E04.

QUE NO SE PRUEBA. La representacion **sparse** de bge-m3 no se usa: es BM25
con otro nombre, y el hibrido lexico ya se midio dos veces (RRF y union) y
perdio 15-4 y 19-2. Solo densa.

RIESGO DECLARADO, fijado antes de medir:
  - 5 celdas contra 50 consultas es la maquina de sobreajuste de la leccion 2.
    Se adopta solo si el IC al 90% del delta pareado excluye una perdida de
    0,02 en las DOS muestras, y **ante empate se conserva la entregada**.
  - Corolario de E08: una ganancia que caiga sobre las 9 consultas con
    etiqueta de agente NO cuenta como evidencia. Por eso van aparte las 41
    humanas y las 10 independientes.
  - La celda de tres le da a los secundarios 1,8 de autoridad contra 1,0 del
    primario. Si gana, hay que descartar que sea solo efecto del peso total
    antes de adoptarla: la comparacion limpia es contra las celdas de dos.
  - Adoptarlo REEMPLAZARIA a un re-puntuador, no se sumaria, y anadiria 527 MB
    a una entrega que ya pesa 1,5 GB.

Uso:
    .venv-cuda/Scripts/python.exe dev/scripts/barrido_repuntuador_e25.py --fase 1
    .venv/Scripts/python.exe      dev/scripts/barrido_repuntuador_e25.py --fase 2
"""

import argparse
import json
import sys
from pathlib import Path

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
SALIDA = DEV_DIR / "intermedios" / "repunt_e25"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
E5 = "multilingual-e5-base"
GTE = "gte-multilingual-base"
BGE = "bge-m3"

PESO = 0.60      # E01
AGG = "top5"     # E07
PROF = 200       # E08
K_POOL = 100

# El tercer elemento es el peso; None = PESO. Las dos ultimas celdas son el
# CONTROL exigido por el pre-registro: la de tres a 0.60 le da a los
# secundarios 1.8 de autoridad total contra 1.0 del primario, asi que hay que
# separar "el tercer encoder aporta senal" de "mas peso total al re-rank".
#   - gte+e5 @0.90  -> mismo 1.8 total, SIN bge. Si iguala a la de tres, el
#     efecto es el peso y bge no aporta nada.
#   - gte+e5+bge @0.40 -> 1.2 total, el mismo que la entregada tiene con dos
#     a 0.60. Si gana ahi, el aporte es del encoder.
CELDAS = [
    ("entregada: gte+e5", (GTE, E5), None),
    ("bge solo", (BGE,), None),
    ("gte+bge", (GTE, BGE), None),
    ("bge+e5", (BGE, E5), None),
    ("gte+e5+bge", (GTE, E5, BGE), None),
    ("CONTROL gte+e5 @0.90", (GTE, E5), 0.90),
    ("CONTROL gte+e5+bge @0.40", (GTE, E5, BGE), 0.40),
]
BASE = CELDAS[0][0]
COLS = ("F1(50)", "ND(50)", "NDp(50)", "F1(ind)", "ND(ind)", "NDp(ind)", "F1(hum)", "ND(hum)", "NDp(hum)")


def pools_del_primario(consultas):
    """PROF candidatos por consulta. Solo carga el indice del primario."""
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = load_index(MINILM)
    pools = {}
    for c in consultas:
        hits = search(expandir_consulta(c["text"]), enc_p, idx_p, metadata, k=PROF)[:PROF]
        pools[c["query_id"]] = hits
    return pools


def fase1(consultas):
    """Codifica en caliente los candidatos con bge-m3 y los vuelca a disco."""
    pools = pools_del_primario(consultas)

    # Un chunk candidato de varias consultas se codifica UNA vez.
    textos = {}
    for hits in pools.values():
        for h in hits:
            textos.setdefault(h.fila, h.texto)
    filas = sorted(textos)
    print(f"candidatos unicos a codificar: {len(filas)}", flush=True)

    enc = get_encoder(name=BGE)
    vecs = enc.encode_passages([textos[f] for f in filas])
    qvecs = np.stack([enc.encode_query(expandir_consulta(c["text"])) for c in consultas])

    SALIDA.mkdir(parents=True, exist_ok=True)
    np.save(SALIDA / "bge_pasajes.npy", vecs.astype(np.float32))
    np.save(SALIDA / "bge_consultas.npy", qvecs.astype(np.float32))
    (SALIDA / "filas.json").write_text(json.dumps(filas), encoding="utf-8")
    print(f"fase 1 lista: {vecs.shape} pasajes, {qvecs.shape} consultas -> {SALIDA}", flush=True)


def evaluar(secundarios, consultas, cache_idx, qvecs, bge, peso=PESO):
    """Un pool por consulta, re-puntuado por el conjunto `secundarios`.

    `bge` es (fila -> indice en la matriz, matriz de pasajes, matriz de
    consultas): bge-m3 no tiene indice, asi que sus vectores vienen de la
    fase 1 en vez de `reconstruct(fila)`. Los demas se leen del indice.
    """
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = cache_idx[MINILM]
    pos, pas, qs = bge
    res = {}
    for qi, c in enumerate(consultas):
        hits = search(expandir_consulta(c["text"]), enc_p, idx_p, metadata, k=PROF)[:PROF]
        for sec in secundarios:
            for h in hits:
                if sec == BGE:
                    sim = float(np.dot(qs[qi], pas[pos[h.fila]]))
                else:
                    idx_s = cache_idx[sec][0]
                    sim = float(np.dot(qvecs[sec][c["query_id"]], idx_s.reconstruct(h.fila)))
                h.score += peso * sim
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


def fase2(consultas):
    filas = json.loads((SALIDA / "filas.json").read_text(encoding="utf-8"))
    pas = np.load(SALIDA / "bge_pasajes.npy")
    qs = np.load(SALIDA / "bge_consultas.npy")
    bge = ({f: i for i, f in enumerate(filas)}, pas, qs)

    gt_todo = cargar_jsonl(GT)
    gt = [g for g in gt_todo if g.get("docs_relevantes")]
    # "Independiente" = anotada sin haber visto un pool, o sea SIN campo
    # `pool`. Es la unica muestra sin sesgo de pooling y la que manda cuando
    # discrepa de las 50 (por eso cayeron doc_rrf y gte-primario).
    indep = [g for g in gt if not g.get("pool")]
    humanas = [g for g in gt if g.get("anotador") != "panel-agentes"]
    print(f"ground truth: {len(gt)} con relevantes, {len(indep)} independientes, {len(humanas)} humanas")

    cache_idx = {MINILM: load_index(MINILM)}
    qvecs = {}
    for nombre in (GTE, E5):
        cache_idx[nombre] = load_index(nombre)
        enc = get_encoder(name=nombre)
        qvecs[nombre] = {c["query_id"]: enc.encode_query(expandir_consulta(c["text"])) for c in consultas}

    filas_tabla, crudo = [], {}
    for etiqueta, secundarios, peso in CELDAS:
        res = evaluar(secundarios, consultas, cache_idx, qvecs, bge, peso or PESO)
        vals = []
        for sub in (gt, indep, humanas):
            vals.extend(np.mean(m) if m else float("nan") for m in metricas(res, sub))
        crudo[etiqueta] = {"gt": metricas(res, gt), "ind": metricas(res, indep), "hum": metricas(res, humanas)}
        filas_tabla.append((etiqueta, vals))
        print(f"  {etiqueta:22} " + " ".join(f"{v:.3f}" for v in vals), flush=True)

    print("\n| celda | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 1))
    for etiqueta, vals in filas_tabla:
        print(f"| {etiqueta} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

    print(f"\nDeltas pareados contra '{BASE}' (IC al 90% por bootstrap):")
    for etiqueta, _, _ in CELDAS[1:]:
        for muestra in ("gt", "ind", "hum"):
            for j, nombre in enumerate(("F1", "ND", "NDp")):
                a, b = crudo[BASE][muestra][j], crudo[etiqueta][muestra][j]
                if not a:
                    continue
                d, lo, hi = bootstrap_delta([x - y for x, y in zip(b, a)])
                marca = "  <-- pasa" if lo > -0.02 and d > 0 else ""
                print(f"  {etiqueta:18} {muestra:4} {nombre:4} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{marca}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fase", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()

    consultas = cargar_jsonl(CONSULTAS)
    (fase1 if args.fase == 1 else fase2)(consultas)


if __name__ == "__main__":
    main()
