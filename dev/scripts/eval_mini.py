#!/usr/bin/env python3
"""F1@3 aproximado contra un mini ground truth anotado a mano.

ADL no publica el ground truth, asi que sin esto las decisiones de diseno
(un encoder o dos, fusion RRF si o no, grafo si o no) se toman a ojo. El
ground truth propio cubre hoy 41 de las 50 consultas; el valor absoluto NO
es la nota oficial (la anotacion es parcial: los documentos no anotados
cuentan como irrelevantes aunque quiza no lo sean, asi que el F1 real sera
mayor o igual que este).

**El promedio no decide.** Con ~40 consultas cada una pesa 0.025, asi que
dos que cambien de lado por azar mueven la media mas que un efecto real.
Para elegir entre dos configuraciones usar --comparar-con, que cuenta en
cuantas consultas gana cada una y aplica una prueba de signos.

NDCG@10 de fragmentos queda fuera a proposito: exigiria anotar relevancia
graduada fragmento por fragmento, y para elegir entre configuraciones el
F1@3 a nivel documento ya ordena igual.

Formato del ground truth (una linea por consulta):
    {"query_id": "q020", "docs_relevantes": ["F2-SWF-012", "F2-SWF-031"]}

Uso:
    python scripts/eval_mini.py
    python scripts/eval_mini.py --resultados intermedios/resultados_e5.jsonl
"""

import argparse
import math
import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, N_DOCUMENTS_PER_QUERY, RESULTADOS_PATH  # noqa: E402

GROUND_TRUTH_MINI_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def f1(recuperados: list[str], relevantes: set[str]) -> tuple[float, float, float]:
    """Precision, recall y F1 de una consulta segun la sec. 10.2.1 del PDF.

    El denominador del recall es min(|D*|, 3), NO |D*|: la especificacion lo
    limita asi para no penalizar los casos con mas documentos relevantes que
    cupos disponibles. Dividir por |D*| a secas subestima el recall y puede
    ordenar mal dos configuraciones que se comparan entre si.
    """
    aciertos = len(set(recuperados) & relevantes)
    if aciertos == 0:
        return 0.0, 0.0, 0.0
    p = aciertos / len(recuperados)
    r = aciertos / min(len(relevantes), len(recuperados))
    return p, r, 2 * p * r / (p + r)


def ndcg(fragmentos: list[dict], relevantes: set[str], k: int = 10) -> float:
    """NDCG@k sobre la lista de fragmentos (sec. 10.2.1 del PDF).

    APROXIMACION DELIBERADA: la relevancia de un fragmento se hereda de su
    documento -- vale 1 si su doc_id esta en el ground truth, 0 si no. No
    tenemos anotacion a nivel fragmento, y construirla son 500 juicios a mano
    contra los 41 que costo la de documento.

    Lo que esta aproximacion SI mide: si el sistema pone arriba fragmentos de
    documentos relevantes. Lo que NO mide: si el fragmento concreto responde
    la consulta, que es lo que el evaluador de ADL va a juzgar de verdad. Un
    fragmento de bibliografia de un documento relevante puntua 1 aca y
    probablemente 0 en la evaluacion real. **Sirve para comparar dos
    configuraciones entre si, no para estimar la nota.**
    """
    ganancias = [1.0 if f["doc_id"] in relevantes else 0.0 for f in fragmentos[:k]]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(ganancias, start=1))
    # El ideal se toma como los k cupos llenos de fragmentos relevantes. No
    # sabemos cuantos fragmentos relevantes existen de verdad, y normalizar
    # por los que uno mismo recupero seria tramposo: un sistema que devuelve
    # un solo fragmento relevante en el puesto 1 sacaria 1.0. Con este ideal
    # fijo, la metrica castiga tanto ordenar mal como traer pocos.
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / idcg


def bootstrap_delta(deltas: list[float], reps: int = 10000, semilla: int = 0) -> tuple[float, float, float]:
    """IC al 90% de la media del delta pareado, por bootstrap.

    Reemplaza a la prueba de signos como criterio principal. La prueba de
    signos tira la MAGNITUD de cada delta y tiene un piso duro: con menos de
    6 consultas discordantes no puede bajar de p=0.05 aunque se ganen todas
    (la cascada dio 5-0, p=0.062, y estaba fuera de alcance antes de
    correrla). El IC no tiene ese piso y se puede leer aunque contenga el
    cero: "el efecto esta entre -0.01 y +0.09" es accionable, "no
    concluyente" no lo es.
    """
    import random

    if not deltas:
        return 0.0, 0.0, 0.0
    rnd = random.Random(semilla)
    n = len(deltas)
    medias = sorted(
        sum(rnd.choice(deltas) for _ in range(n)) / n for _ in range(reps)
    )
    return (
        sum(deltas) / n,
        medias[int(0.05 * reps)],
        medias[int(0.95 * reps) - 1],
    )


def veredicto_bootstrap(media: float, bajo: float, alto: float, dano: float = -0.02) -> str:
    """Criterio de adopcion para un torneo, no para una publicacion.

    El coste de adoptar una mejora que en realidad es nula es ~0; el de
    rechazar una real es perder posiciones. Por eso el umbral no es "probar
    que mejora" sino **descartar solo lo que probablemente dana**: se adopta
    si el IC al 90% excluye una perdida de `dano`.
    """
    if bajo > dano:
        return f"  -> ADOPTABLE: el IC al 90% [{bajo:+.3f}, {alto:+.3f}] excluye una perdida de {dano:+.2f}."
    if alto < 0:
        return f"  -> DESCARTAR: el IC al 90% [{bajo:+.3f}, {alto:+.3f}] esta enteramente por debajo de cero."
    return (
        f"  -> NO ADOPTABLE todavia: el IC al 90% [{bajo:+.3f}, {alto:+.3f}] "
        f"admite una perdida mayor que {dano:+.2f}."
    )


def veredicto_signos(nombre_a: str, gana_a: int, nombre_b: str, gana_b: int) -> str:
    """Prueba de signos sobre las consultas en que dos configuraciones difieren.

    Bajo la hipotesis de que son equivalentes, cada consulta que difiere es una
    moneda al aire, asi que p es la probabilidad de ver un reparto al menos tan
    desigual por azar. Existe para que todas las herramientas den el MISMO
    veredicto sobre los mismos datos: antes eval_mini usaba la binomial exacta
    y barrido_retrieval un umbral ad-hoc que ademas callaba cuando la
    diferencia si era significativa.
    """
    difieren = gana_a + gana_b
    if difieren == 0:
        return "  -> las dos configuraciones dan exactamente lo mismo"
    ganador, mayor = (nombre_a, gana_a) if gana_a > gana_b else (nombre_b, gana_b)
    p = min(1.0, 2 * sum(math.comb(difieren, i) for i in range(mayor, difieren + 1)) / 2**difieren)
    linea = f"  prueba de signos sobre las {difieren} que difieren: p = {p:.3f}\n"
    if p > 0.05:
        return linea + (
            "  -> NO concluyente. Entregar la configuracion mas simple, y no "
            "cambiarla apoyandose en la diferencia de promedios."
        )
    return linea + f"  -> {ganador} gana de forma consistente, no solo en promedio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultados", type=Path, default=RESULTADOS_PATH)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_MINI_PATH)
    parser.add_argument("--k", type=int, default=N_DOCUMENTS_PER_QUERY)
    parser.add_argument(
        "--sin-pooling",
        action="store_true",
        help="Usa solo las consultas anotadas de forma independiente (sin campo 'pool'). "
        "OBLIGATORIO al comparar dos encoders entre si: una consulta anotada sobre los "
        "candidatos que propuso el encoder X favorece a X, porque los documentos que X "
        "nunca recupero jamas pudieron marcarse como relevantes.",
    )
    parser.add_argument(
        "--comparar-con",
        type=Path,
        default=None,
        help="Otro resultados.jsonl. En vez de solo promediar, cuenta en cuantas "
        "consultas gana cada uno. Es la forma correcta de decidir entre dos "
        "configuraciones: con ~30 consultas cada una pesa 0.03 en la media, asi que "
        "dos que cambien de lado por azar mueven el promedio mas que un efecto real.",
    )
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth} -- hay que anotarlo a mano primero")
        sys.exit(2)

    resultados = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    gt = cargar_jsonl(args.ground_truth)

    if args.sin_pooling:
        antes = len(gt)
        gt = [f for f in gt if not f.get("pool")]
        print(f"solo anotacion independiente: {len(gt)} de {antes} consultas\n")

    suma = 0.0
    suma_ndcg = 0.0
    print(f"{'consulta':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'NDCG':>6s}  aciertos")
    for fila in gt:
        qid = fila["query_id"]
        relevantes = set(fila["docs_relevantes"])
        res = resultados.get(qid)
        if res is None:
            print(f"{qid:10s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s}  (sin resultado)")
            continue
        docs = [d["doc_id"] for d in res["documents"][: args.k]]
        p, r, valor = f1(docs, relevantes)
        n = ndcg(res.get("fragments", []), relevantes)
        suma += valor
        suma_ndcg += n
        print(
            f"{qid:10s} {p:6.2f} {r:6.2f} {valor:6.2f} {n:6.2f}  "
            f"{sorted(set(docs) & relevantes)}"
        )

    print(f"\nF1@{args.k} promedio sobre {len(gt)} consultas: {suma / len(gt):.3f}")
    print(f"NDCG@10 aproximado (relevancia heredada del documento): {suma_ndcg / len(gt):.3f}")

    if args.comparar_con:
        otros = {r["query_id"]: r for r in cargar_jsonl(args.comparar_con)}
        a = args.resultados.name
        b = args.comparar_con.name
        gana_a = gana_b = empate = 0
        suma_b = 0.0
        deltas_f1: list[float] = []
        deltas_ndcg: list[float] = []
        for fila in gt:
            qid = fila["query_id"]
            relevantes = set(fila["docs_relevantes"])
            if qid not in resultados or qid not in otros:
                continue
            va = f1([d["doc_id"] for d in resultados[qid]["documents"][: args.k]], relevantes)[2]
            vb = f1([d["doc_id"] for d in otros[qid]["documents"][: args.k]], relevantes)[2]
            suma_b += vb
            deltas_f1.append(va - vb)
            deltas_ndcg.append(
                ndcg(resultados[qid].get("fragments", []), relevantes)
                - ndcg(otros[qid].get("fragments", []), relevantes)
            )
            if abs(va - vb) < 1e-9:
                empate += 1
            elif va > vb:
                gana_a += 1
            else:
                gana_b += 1
        print(f"\n{b}: F1@{args.k} promedio {suma_b / len(gt):.3f}")
        print(f"\n{a} gana en {gana_a}, {b} gana en {gana_b}, empatan {empate}")
        print(veredicto_signos(a, gana_a, b, gana_b))

        # Criterio principal: el delta pareado con su intervalo. El F1@3 se
        # mueve en escalones de ~0.32 y con 41 consultas no detecta nada por
        # debajo de 0.06; el NDCG@10 por fragmento es casi continuo y vale
        # como 6-9 veces mas consultas para la misma decision.
        for etiqueta, deltas in (("F1@3", deltas_f1), ("NDCG@10", deltas_ndcg)):
            media, bajo, alto = bootstrap_delta(deltas)
            print(f"\ndelta pareado en {etiqueta} ({a} menos {b}): {media:+.3f}")
            print(veredicto_bootstrap(media, bajo, alto))


if __name__ == "__main__":
    main()
