#!/usr/bin/env python3
"""Anotar relevancia a nivel FRAGMENTO, para que el NDCG@10 deje de ser un proxy.

Por que hace falta
------------------
Hoy `eval_mini.py` calcula el NDCG@10 heredando la relevancia del documento:
un fragmento vale 1 si su `doc_id` esta en el ground truth. Eso no puede ver
lo unico que el evaluador de ADL si ve -- si el pasaje concreto responde la
consulta (sec. 10.2.1: la relevancia se juzga sobre el campo `text`). El
NDCG@10 es la mitad del puntaje, y todo lo que se hizo sobre fragmentos
(prioridad de idioma, gate de bibliografia) se adopto con argumento mecanico
porque el instrumento no podia medirlo.

Anotar 10 consultas son 100 juicios y convierte esa mitad en algo medible.

Escala GRADUADA, que es lo que un NDCG necesita
-----------------------------------------------
    2  responde la consulta: el evaluador encontraria aca la respuesta
    1  es del tema pero no responde (contexto, definicion, una mencion)
    0  no aporta: bibliografia, indice, portada, otro tema, otro idioma

La distincion 2/1 es justamente la que el proxy binario no tiene: un chunk
de un documento relevante que solo menciona el tema de pasada hoy puntua 1 y
para ADL vale poco.

SESGO QUE HAY QUE DECLARAR AL USARLO: se anota sobre los fragmentos que el
sistema entrego, o sea pooling a nivel fragmento. Sirve para comparar dos
configuraciones sobre el mismo pool y para saber cuanto miente el proxy, NO
para estimar la nota de ADL.

Por defecto se anotan las 10 consultas de anotacion INDEPENDIENTE, que son la
muestra que ya se usa como guardia contra el sesgo de pooling a nivel
documento.

Uso:
    python dev/scripts/anotar_fragmentos.py --generar     # -> dev/eval/fragmentos.md
    python dev/scripts/anotar_fragmentos.py --recolectar  # -> ground_truth_fragmentos.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, RESULTADOS_PATH  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT_DOCS_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
FRAGMENTOS_MD = DEV_DIR / "eval" / "fragmentos.md"
GT_FRAGMENTOS_PATH = DEV_DIR / "eval" / "ground_truth_fragmentos.jsonl"

# "- [2] q020 f03 :: F2-SWF-124-c0031" -- la nota va dentro de los corchetes.
PATRON_NOTA = re.compile(r"^-\s*\[([012 ])\]\s*(\S+)\s+f(\d+)\s*::\s*(\S+)")
PATRON_CONSULTA = re.compile(r"^##\s+(q\d+)")


def cargar_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def consultas_independientes(gt_docs: list[dict]) -> list[str]:
    """Las que se anotaron sin pasar por el recuperador. Son la muestra limpia."""
    return [
        g["query_id"]
        for g in gt_docs
        if not g.get("pool") and not g.get("anotador") and g["docs_relevantes"]
    ]


def generar(args: argparse.Namespace) -> None:
    resultados = {r["query_id"]: r for r in cargar_jsonl(args.resultados)}
    consultas = {c["query_id"]: c for c in cargar_jsonl(CONSULTAS_PATH)}
    gt_docs = {g["query_id"]: g for g in cargar_jsonl(GT_DOCS_PATH)}

    qids = args.solo or consultas_independientes(list(gt_docs.values()))
    qids = [q for q in qids if q in resultados]
    if not qids:
        sys.exit("no hay consultas para anotar")

    lineas = [
        "# Anotacion de fragmentos",
        "",
        f"Escribi la nota DENTRO de los corchetes ({len(qids)} consultas, "
        f"{sum(len(resultados[q]['fragments']) for q in qids)} fragmentos):",
        "",
        "- `[2]` responde la consulta",
        "- `[1]` es del tema pero no responde (contexto, definicion, mencion)",
        "- `[0]` no aporta (bibliografia, indice, portada, otro tema, otro idioma)",
        "",
        "Dejar `[ ]` sin tocar equivale a 0. Despues:",
        "`python dev/scripts/anotar_fragmentos.py --recolectar`",
        "",
        "El `doc?` de cada fragmento dice si su documento estaba marcado como "
        "relevante en el ground truth de DOCUMENTOS: sirve para ver de un vistazo "
        "los casos que el proxy no distingue (documento relevante, pasaje que no "
        "responde).",
        "",
    ]

    for qid in qids:
        relevantes = set(gt_docs.get(qid, {}).get("docs_relevantes", []))
        lineas += [f"## {qid}", "", f"**{consultas.get(qid, {}).get('text', '')}**", ""]
        for i, frag in enumerate(resultados[qid]["fragments"], start=1):
            marca = "SI" if frag["doc_id"] in relevantes else "no"
            texto = " ".join(frag["text"].split())
            lineas += [
                f"- [ ] {qid} f{i:02d} :: {frag['chunk_id']}   (doc {frag['doc_id']}, doc? {marca})",
                "",
                f"  > {texto[: args.max_chars]}"
                + ("..." if len(texto) > args.max_chars else ""),
                "",
            ]

    FRAGMENTOS_MD.write_text("\n".join(lineas), encoding="utf-8")
    print(f"escrito: {FRAGMENTOS_MD}  ({len(qids)} consultas)")


def recolectar(args: argparse.Namespace) -> None:
    if not FRAGMENTOS_MD.exists():
        sys.exit(f"no existe {FRAGMENTOS_MD} -- correr primero --generar")

    notas: dict[str, dict[str, int]] = {}
    vistas: list[str] = []
    for linea in FRAGMENTOS_MD.read_text(encoding="utf-8").splitlines():
        m_q = PATRON_CONSULTA.match(linea)
        if m_q:
            vistas.append(m_q.group(1))
            notas.setdefault(m_q.group(1), {})
            continue
        m = PATRON_NOTA.match(linea)
        if m:
            valor = m.group(1).strip()
            notas.setdefault(m.group(2), {})[m.group(4)] = int(valor) if valor else 0

    if not vistas:
        sys.exit(f"ninguna consulta en {FRAGMENTOS_MD}")

    # Re-anotar una consulta la reemplaza, no la duplica.
    por_qid = {f["query_id"]: f for f in cargar_jsonl(GT_FRAGMENTOS_PATH)}
    for qid in vistas:
        por_qid[qid] = {
            "query_id": qid,
            "nota": "relevancia graduada por fragmento, anotada a mano",
            "fragmentos": notas.get(qid, {}),
        }

    with GT_FRAGMENTOS_PATH.open("w", encoding="utf-8") as f:
        for qid in sorted(por_qid):
            f.write(json.dumps(por_qid[qid], ensure_ascii=False) + "\n")

    total = sum(len(por_qid[q]["fragmentos"]) for q in vistas)
    con_nota = sum(
        1 for q in vistas for v in por_qid[q]["fragmentos"].values() if v > 0
    )
    print(f"escrito: {GT_FRAGMENTOS_PATH}")
    print(f"  {len(vistas)} consultas, {total} fragmentos, {con_nota} con nota > 0")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--generar", action="store_true")
    ap.add_argument("--recolectar", action="store_true")
    ap.add_argument("--resultados", type=Path, default=RESULTADOS_PATH)
    ap.add_argument("--solo", nargs="+", default=None, help="Limitar a estos query_id.")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Recorte del texto mostrado. Solo afecta la lectura, no la anotacion.",
    )
    args = ap.parse_args()

    if args.generar:
        generar(args)
    elif args.recolectar:
        recolectar(args)
    else:
        ap.error("hay que pasar --generar o --recolectar")


if __name__ == "__main__":
    main()
