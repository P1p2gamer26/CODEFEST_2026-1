#!/usr/bin/env python3
"""F1@3 y NDCG@10 aproximados contra un mini ground truth anotado a mano.

ADL no publica el ground truth, asi que sin esto las decisiones de diseno
(un encoder o dos, fusion RRF si o no, grafo si o no) se toman a ojo. El
ground truth propio cubre **las 50 consultas**; el valor absoluto NO es la
nota oficial (la anotacion es parcial: los documentos no anotados cuentan
como irrelevantes aunque quiza no lo sean, asi que el F1 real sera mayor o
igual que este).

LA METRICA DE DECISION ES LA MEDIA SOBRE LAS 50. Es la forma exacta de las
ecs. (10) y (14) del PDF. Los subconjuntos (--sin-pooling, --solo-humanas)
son diagnostico, no titulares paralelos: sirven para detectar que un cambio
esta explotando el sesgo de pooling, no para reportar el resultado.

**El promedio no decide.** Con 50 consultas cada una pesa 0.02 y el F1@3 se
mueve en escalones de ~0.32, asi que dos que cambien de lado por azar mueven
la media mas que un efecto real. Para elegir entre dos configuraciones usar
--comparar-con, que reporta el delta pareado con su intervalo de confianza.

Techo del F1@3 sobre este ground truth: **0.906**. No es 1: la sec. 10.2.2
fija P@3 = |aciertos|/3, asi que una consulta con un solo documento relevante
no puede pasar de 0.50 y una con dos, de 0.80.

El NDCG@10 se reporta en dos versiones. Ver `ndcg` y `ndcg_penalizado`.

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
from src.retrieval.calidad_chunk import calidad  # noqa: E402

GROUND_TRUTH_MINI_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
# Relevancia graduada por fragmento, anotada a mano. Opcional: si no existe,
# el NDCG@10 sigue siendo el proxy heredado del documento.
GT_FRAGMENTOS_PATH = DEV_DIR / "eval" / "ground_truth_fragmentos.jsonl"


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
    contra los 50 que costo la de documento.

    Lo que esta aproximacion SI mide: si el sistema pone arriba fragmentos de
    documentos relevantes. Lo que NO mide: si el fragmento concreto responde
    la consulta, que es lo que el evaluador de ADL va a juzgar de verdad. Un
    fragmento de bibliografia de un documento relevante puntua 1 aca y
    probablemente 0 en la evaluacion real. **Sirve para comparar dos
    configuraciones entre si, no para estimar la nota.**

    `ndcg_penalizado` acota esa segunda parte: ver su docstring.
    """
    return _ndcg_desde_ganancias(
        [1.0 if f["doc_id"] in relevantes else 0.0 for f in fragmentos[:k]], k
    )


def ndcg_penalizado(fragmentos: list[dict], relevantes: set[str], k: int = 10) -> float:
    """Igual que `ndcg`, pero descuenta la parte del texto que es bibliografia.

    Existe para acotar CUANTO MIENTE el proxy binario. La sec. 10.2.1 dice que
    el evaluador juzga el campo `text`, asi que un fragmento que es una lista
    de notas al pie de un documento relevante vale 1 para `ndcg` y ~0 para
    ADL. Aca la ganancia es `relevancia * calidad(texto)`, con `calidad`
    contando patrones tipograficos (ver src/retrieval/calidad_chunk.py).

    Medido sobre la entrega actual: 0.3933 binario contra 0.3740 penalizado.
    O sea que la bibliografia explica ~0.019 (4.9%) de la brecha.

    ES UNA COTA INFERIOR, no una estimacion del error total del proxy. Solo
    detecta aparato bibliografico, que es la parte mecanicamente reconocible.
    El otro modo de fallo -- documento relevante, pasaje que no responde la
    consulta -- no lo ve nadie sin anotacion humana a nivel fragmento.
    """
    ganancias = [
        (1.0 if f["doc_id"] in relevantes else 0.0) * calidad(f.get("text", ""))
        for f in fragmentos[:k]
    ]
    return _ndcg_desde_ganancias(ganancias, k)


def ndcg_anotado(fragmentos: list[dict], notas: dict[str, int], k: int = 10) -> float:
    """NDCG@10 REAL, con relevancia graduada anotada fragmento por fragmento.

    Es el unico de los tres que no es un proxy: la nota la puso un humano
    mirando el texto, que es lo que hace el evaluador de ADL (sec. 10.2.1).
    Las notas salen de `dev/eval/ground_truth_fragmentos.jsonl`
    (ver scripts/anotar_fragmentos.py): 2 = responde, 1 = del tema pero no
    responde, 0 = no aporta.

    La ganancia se escala a [0, 1] dividiendo por 2, para que el numero sea
    directamente comparable con `ndcg` y `ndcg_penalizado` sobre las mismas
    consultas -- que es justo la comparacion que dice cuanto miente el proxy.

    Un fragmento sin nota cuenta 0: el .md se genera con los 10 fragmentos de
    la consulta, asi que si falta una nota es que el anotador la dejo vacia, y
    la consigna dice que vacio equivale a 0.
    """
    ganancias = [notas.get(f.get("chunk_id", ""), 0) / 2.0 for f in fragmentos[:k]]
    return _ndcg_desde_ganancias(ganancias, k)


def _ndcg_desde_ganancias(ganancias: list[float], k: int) -> float:
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(ganancias, start=1))
    # El ideal se toma como los k cupos llenos de fragmentos relevantes. No
    # sabemos cuantos fragmentos relevantes existen de verdad, y normalizar
    # por los que uno mismo recupero seria tramposo: un sistema que devuelve
    # un solo fragmento relevante en el puesto 1 sacaria 1.0. Con este ideal
    # fijo, la metrica castiga tanto ordenar mal como traer pocos.
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / idcg


def techo_f1(gt: list[dict], k: int = 3) -> float:
    """F1@k maximo alcanzable sobre este ground truth, no 1.0.

    La sec. 10.2.2 fija P@k = |aciertos|/k con k cupos SIEMPRE llenos, mientras
    que el denominador del recall es min(|D*|, k). Una consulta con un solo
    documento relevante topa en 0.50 y una con dos, en 0.80. Sin este numero a
    la vista es facil leer un 0.36 como "vamos por el 36%" cuando en realidad
    es el 40% de lo alcanzable.
    """
    total = 0.0
    for fila in gt:
        aciertos = min(len(fila["docs_relevantes"]), k)
        if aciertos == 0:
            continue
        p, r = aciertos / k, 1.0
        total += 2 * p * r / (p + r)
    return total / len(gt) if gt else 0.0


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
        "--solo-humanas",
        action="store_true",
        help="Excluye las consultas etiquetadas por el las rondas de anotacion asistida. "
        "Necesario para comparar contra cualquier medicion anterior al 2 ago 2026: "
        "la anotacion asistida reproduce al humano con F1 0.23, asi que sus etiquetas miden "
        "parecido entre modelos, no acierto.",
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

    # Una consulta con docs_relevantes vacio fue MIRADA y no tenia nada
    # relevante en su pool. Promediarla como F1=0 castigaria al recuperador
    # por no encontrar algo que el anotador tampoco encontro; excluirla es lo
    # correcto y ademas mantiene el promedio comparable con las corridas
    # anteriores, que simplemente no tenian esas consultas en el archivo.
    vacias = [f["query_id"] for f in gt if not f["docs_relevantes"]]
    if vacias:
        gt = [f for f in gt if f["docs_relevantes"]]
        print(f"excluidas {len(vacias)} consultas sin documento relevante en su pool: {' '.join(vacias)}\n")

    if args.solo_humanas:
        antes = len(gt)
        gt = [f for f in gt if not f.get("anotador")]
        print(f"solo etiquetas humanas: {len(gt)} de {antes} consultas\n")

    suma = 0.0
    suma_ndcg = 0.0
    suma_ndcg_pen = 0.0
    # Las etiquetas del la anotacion asistida reproducen al humano con F1 0.23
    # (dev/eval/anotacion_asistida/README.md). El promedio de decision es sobre
    # TODAS las consultas -- es la forma de las ecs. (10) y (14) del PDF --
    # pero se sigue mostrando el desglose para no perder de vista que esa
    # fraccion del numero descansa en etiquetas debiles.
    por_origen: dict[str, list[float]] = {}
    print(
        f"{'consulta':10s} {'P':>6s} {'R':>6s} {'F1':>6s} {'NDCG':>6s} {'NDCGp':>6s} "
        f"{'quien':>6s}  aciertos"
    )
    for fila in gt:
        qid = fila["query_id"]
        relevantes = set(fila["docs_relevantes"])
        res = resultados.get(qid)
        if res is None:
            print(f"{qid:10s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s} {'--':>6s}  (sin resultado)")
            continue
        docs = [d["doc_id"] for d in res["documents"][: args.k]]
        p, r, valor = f1(docs, relevantes)
        fragmentos = res.get("fragments", [])
        n = ndcg(fragmentos, relevantes)
        n_pen = ndcg_penalizado(fragmentos, relevantes)
        suma += valor
        suma_ndcg += n
        suma_ndcg_pen += n_pen
        origen = "asistido" if fila.get("anotador") else "humano"
        por_origen.setdefault(origen, []).append(valor)
        print(
            f"{qid:10s} {p:6.2f} {r:6.2f} {valor:6.2f} {n:6.2f} {n_pen:6.2f} {origen:>6s}  "
            f"{sorted(set(docs) & relevantes)}"
        )

    media_ndcg = suma_ndcg / len(gt)
    media_ndcg_pen = suma_ndcg_pen / len(gt)
    print(f"\nF1@{args.k} promedio sobre {len(gt)} consultas: {suma / len(gt):.3f}")
    print(f"  techo alcanzable con este ground truth: {techo_f1(gt, args.k):.3f}")
    print(f"NDCG@10 aproximado (relevancia heredada del documento): {media_ndcg:.3f}")
    print(f"NDCG@10 penalizado por calidad de texto:                {media_ndcg_pen:.3f}")
    print(
        f"  el proxy sobreestima al menos {media_ndcg - media_ndcg_pen:+.3f} "
        f"({100 * (media_ndcg - media_ndcg_pen) / max(media_ndcg, 1e-9):.1f}%) "
        f"solo por bibliografia; ver ndcg_penalizado()."
    )
    # Si hay anotacion a nivel fragmento, el NDCG deja de ser proxy en esas
    # consultas. Se reportan las tres versiones sobre EL MISMO subconjunto,
    # que es la unica comparacion que dice cuanto miente el proxy de verdad.
    notas_frag = (
        {f["query_id"]: f["fragmentos"] for f in cargar_jsonl(GT_FRAGMENTOS_PATH)}
        if GT_FRAGMENTOS_PATH.exists()
        else {}
    )
    anotadas = [
        fila
        for fila in gt
        if fila["query_id"] in notas_frag and fila["query_id"] in resultados
    ]
    if anotadas:
        def media(fn):
            return sum(fn(fila) for fila in anotadas) / len(anotadas)

        frags = lambda fila: resultados[fila["query_id"]].get("fragments", [])  # noqa: E731
        rel = lambda fila: set(fila["docs_relevantes"])  # noqa: E731
        real = media(lambda f: ndcg_anotado(frags(f), notas_frag[f["query_id"]]))
        proxy = media(lambda f: ndcg(frags(f), rel(f)))
        pen = media(lambda f: ndcg_penalizado(frags(f), rel(f)))
        print(f"\nsobre las {len(anotadas)} consultas con anotacion POR FRAGMENTO:")
        print(f"  NDCG@10 real (relevancia graduada a mano): {real:.3f}")
        print(f"  NDCG@10 proxy (heredado del documento):    {proxy:.3f}")
        print(f"  NDCG@10 penalizado por bibliografia:       {pen:.3f}")
        print(
            f"  el proxy se equivoca en {proxy - real:+.3f}; el penalizado, en "
            f"{pen - real:+.3f}. Es la unica medicion no-proxy del proyecto y "
            "vale para las decisiones sobre fragmentos."
        )

    if len(por_origen) > 1:
        print("\ndesglose por procedencia de la etiqueta:")
        for origen in sorted(por_origen):
            vs = por_origen[origen]
            print(f"  {origen:7s} {len(vs):2d} consultas  F1@{args.k} {sum(vs) / len(vs):.3f}")
        print(
            "  el desglose es diagnostico, no un segundo titular: la metrica de\n"
            "  decision es la media sobre todas. Pero la anotacion asistida reproduce al humano\n"
            "  con F1 0.23 (dev/eval/anotacion_asistida/README.md), asi que esas etiquetas\n"
            "  son el eslabon debil del promedio -- re-anotarlas a mano es la tarea de\n"
            "  anotacion de mayor prioridad. Para comparar contra mediciones anteriores\n"
            "  al 2 ago 2026 sigue estando --solo-humanas."
        )

    if args.comparar_con:
        otros = {r["query_id"]: r for r in cargar_jsonl(args.comparar_con)}
        a = args.resultados.name
        b = args.comparar_con.name
        gana_a = gana_b = empate = 0
        suma_b = 0.0
        deltas_f1: list[float] = []
        deltas_ndcg: list[float] = []
        deltas_ndcg_pen: list[float] = []
        for fila in gt:
            qid = fila["query_id"]
            relevantes = set(fila["docs_relevantes"])
            if qid not in resultados or qid not in otros:
                continue
            va = f1([d["doc_id"] for d in resultados[qid]["documents"][: args.k]], relevantes)[2]
            vb = f1([d["doc_id"] for d in otros[qid]["documents"][: args.k]], relevantes)[2]
            suma_b += vb
            deltas_f1.append(va - vb)
            fa = resultados[qid].get("fragments", [])
            fb = otros[qid].get("fragments", [])
            deltas_ndcg.append(ndcg(fa, relevantes) - ndcg(fb, relevantes))
            deltas_ndcg_pen.append(
                ndcg_penalizado(fa, relevantes) - ndcg_penalizado(fb, relevantes)
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
        for etiqueta, deltas in (
            ("F1@3", deltas_f1),
            ("NDCG@10", deltas_ndcg),
            ("NDCG@10 penalizado", deltas_ndcg_pen),
        ):
            media, bajo, alto = bootstrap_delta(deltas)
            print(f"\ndelta pareado en {etiqueta} ({a} menos {b}): {media:+.3f}")
            print(veredicto_bootstrap(media, bajo, alto))


if __name__ == "__main__":
    main()
