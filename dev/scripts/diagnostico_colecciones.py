#!/usr/bin/env python3
"""Separa los fallos de F1@3 en "tema equivocado" y "hermano equivocado".

Por que hace falta otro diagnostico
-----------------------------------
`diagnostico_ceros.py` responde DONDE se pierde el documento en la tuberia
(ausente del indice / fuera del pool / pierde la agregacion) y para eso
necesita el indice FAISS cargado. Este responde otra cosa, mas barata: QUE TAN
LEJOS quedo la respuesta. Solo necesita `resultados.jsonl` y el ground truth,
asi que corre sin indice, sin corpus y sin GPU.

La distincion importa porque manda a herramientas distintas:

  TEMA EQUIVOCADO    ni siquiera acertamos la coleccion. Es un fallo de
                     representacion de la consulta -- vocabulario, idioma --
                     y lo ataca la expansion por glosario.
  HERMANO EQUIVOCADO acertamos la coleccion y erramos el documento dentro de
                     ella. El encoder entendio el tema; lo que no sabe es cual
                     de N informes casi identicos de la misma serie responde.
                     Eso lo ataca el grafo (sec. 8.5), no el glosario.

Medido sobre la entrega actual: 22% de las consultas son hermano equivocado y
solo 12% tema equivocado. O sea que el bucket grande y direccionable NO es el
que ataca el glosario.

Ademas reporta densidad de aparato bibliografico por coleccion, porque las dos
cosas resultaron estar relacionadas: F1-CSET es a la vez la coleccion que mas
aparece como "esperada pero no devuelta" en los fallos anotados a mano y la que
mas bibliografia tiene (2.6x la siguiente).

Uso:
    python dev/scripts/diagnostico_colecciones.py
    python dev/scripts/diagnostico_colecciones.py --resultados otro.jsonl
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, N_DOCUMENTS_PER_QUERY, RESULTADOS_PATH  # noqa: E402
from src.retrieval.calidad_chunk import fraccion_aparato  # noqa: E402
from src.retrieval.truncate import UMBRAL_APARATO  # noqa: E402

GROUND_TRUTH_MINI_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"

# Los doc_id son "F<fase>-<COLECCION>-<numero>": F1-CSET-098, F3-MAPPOEA-030.
_COLECCION = re.compile(r"^(F\d-[A-Z]+)-")


def coleccion(doc_id: str) -> str:
    m = _COLECCION.match(doc_id)
    return m.group(1) if m else doc_id


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def texto_consulta(fila: dict) -> str:
    """El campo con el enunciado cambia de nombre entre versiones del archivo;
    se toma la cadena larga que no sea el id."""
    for clave, valor in fila.items():
        if clave != "query_id" and isinstance(valor, str) and len(valor) > 25:
            return valor
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=RESULTADOS_PATH)
    ap.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_MINI_PATH)
    ap.add_argument("--k", type=int, default=N_DOCUMENTS_PER_QUERY)
    args = ap.parse_args()

    gt = {f["query_id"]: f for f in cargar_jsonl(args.ground_truth)}
    res = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    consultas = (
        {c["query_id"]: c for c in cargar_jsonl(CONSULTAS_PATH)}
        if CONSULTAS_PATH.exists()
        else {}
    )

    acierta_doc: list[str] = []
    hermano: list[tuple[str, set[str], list[str]]] = []
    tema: list[tuple[str, set[str], list[str]]] = []

    for qid, fila in gt.items():
        if qid not in res:
            continue
        relevantes = set(fila["docs_relevantes"])
        if not relevantes:
            continue
        cols_rel = {coleccion(d) for d in relevantes}
        devueltos = [d["doc_id"] for d in res[qid]["documents"][: args.k]]
        cols_dev = {coleccion(d) for d in devueltos}

        if set(devueltos) & relevantes:
            acierta_doc.append(qid)
        elif cols_dev & cols_rel:
            hermano.append((qid, cols_rel, devueltos))
        else:
            tema.append((qid, cols_rel, devueltos))

    n = len(acierta_doc) + len(hermano) + len(tema)
    if not n:
        print("no hay consultas evaluables")
        return

    print(f"sobre {n} consultas con ground truth:\n")
    for etiqueta, grupo in (
        ("acierta al menos 1 DOCUMENTO", acierta_doc),
        ("COLECCION correcta, documento no  <- hermano equivocado", hermano),
        ("falla hasta la COLECCION          <- tema equivocado", tema),
    ):
        print(f"  {etiqueta:56s} {len(grupo):3d} ({100 * len(grupo) / n:.0f}%)")

    for titulo, grupo in (
        ("HERMANO EQUIVOCADO -- candidatas del grafo (sec. 8.5)", hermano),
        ("TEMA EQUIVOCADO -- candidatas del glosario bilingue", tema),
    ):
        print(f"\n=== {titulo} ===")
        for qid, cols_rel, devueltos in grupo:
            origen = "agente" if gt[qid].get("anotador") else "humano"
            print(f"\n  {qid} [{origen}]")
            if qid in consultas:
                print(f"    consulta : {texto_consulta(consultas[qid])[:140]}")
            print(f"    esperaba : {','.join(sorted(cols_rel))}")
            print(f"    devolvio : {devueltos}")

    # Densidad de aparato por coleccion. Se mide sobre los fragmentos EMITIDOS,
    # que son una muestra sesgada (son los que puntuaron alto), asi que sirve
    # para comparar colecciones entre si, no para estimar el corpus.
    por_coleccion: dict[str, list[float]] = defaultdict(list)
    for fila in res.values():
        for frag in fila.get("fragments", []):
            por_coleccion[coleccion(frag["doc_id"])].append(fraccion_aparato(frag["text"]))

    print("\n=== densidad de aparato bibliografico por coleccion ===")
    print("(sobre los fragmentos entregados: muestra sesgada, sirve para comparar)")
    print(f"\n  {'coleccion':16s} {'n':>5s} {'media':>8s} {'>=' + str(UMBRAL_APARATO):>8s}")
    for col, vals in sorted(por_coleccion.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        if len(vals) < 5:
            continue
        alto = 100 * sum(1 for v in vals if v >= UMBRAL_APARATO) / len(vals)
        print(f"  {col:16s} {len(vals):5d} {sum(vals) / len(vals):8.3f} {alto:7.1f}%")


if __name__ == "__main__":
    main()
