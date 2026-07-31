#!/usr/bin/env python3
"""Prepara y recoge la anotacion manual del ground truth.

Ampliar dev/eval/ground_truth_mini.jsonl es la mejora pendiente de mayor
impacto: con 10 consultas, cualquier ajuste del recuperador tiene mas chance
de ser ruido de muestreo que senal (ver seccion 7 del informe tecnico). Pero
anotar desde cero exige leer 1818 documentos, asi que este script propone
candidatos y deja que la persona solo marque.

Dos modos:

    --generar     escribe dev/eval/candidatos.md, una lista de casillas por
                  consulta con un extracto de cada documento candidato.
    --recolectar  lee ese mismo .md ya marcado y lo convierte al formato de
                  ground_truth_mini.jsonl.

El sesgo obvio: los candidatos salen del propio recuperador, asi que un
documento relevante que el sistema nunca recupera no aparece en la lista y
no se puede marcar. Es una anotacion por pooling, la misma tecnica de TREC:
sirve para comparar configuraciones entre si, no para medir recall absoluto.

Uso:
    python dev/scripts/anotar_candidatos.py --generar
    python dev/scripts/anotar_candidatos.py --recolectar
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, ENCODER_PRIMARY_NAME  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.search import search  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
CANDIDATOS_PATH = DEV_DIR / "eval" / "candidatos.md"

N_CANDIDATOS = 10  # por consulta
K_POOL = 200  # pool amplio: la anotacion debe ver mas alla de lo que entrega


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def generar(args: argparse.Namespace) -> None:
    consultas = cargar_jsonl(CONSULTAS_PATH)
    ya_anotadas = {f["query_id"] for f in cargar_jsonl(GROUND_TRUTH_PATH)} if GROUND_TRUTH_PATH.exists() else set()
    pendientes = [c for c in consultas if c["query_id"] not in ya_anotadas]
    if args.solo:
        pendientes = [c for c in pendientes if c["query_id"] in set(args.solo)]
    if not pendientes:
        print("no hay consultas pendientes de anotar")
        return

    encoder = get_encoder(name=args.encoder_name)
    index, metadata = load_index(args.encoder_name, index_dir=args.index_dir)
    print(f"encoder: {encoder.name} -> {index.ntotal} vectores")

    # doc_id -> (fuente, mejor fragmento) para dar contexto al anotador
    lineas = [
        "# Candidatos para anotar el ground truth",
        "",
        f"Marca con `[x]` los documentos RELEVANTES para cada consulta ({len(pendientes)} pendientes).",
        "Deja en `[ ]` los que no lo sean y no borres ninguna linea. Al terminar, corre:",
        "",
        "```",
        "python dev/scripts/anotar_candidatos.py --recolectar",
        "```",
        "",
        "Una consulta sin ningun documento marcado se omite del ground truth.",
        "",
    ]

    for consulta in pendientes:
        qid = consulta["query_id"]
        hits = search(consulta["text"], encoder, index, metadata, k=K_POOL)
        docs = aggregate_documents(hits, top_n=N_CANDIDATOS, strategy="sum")
        mejor_frag = {}
        for h in hits:
            mejor_frag.setdefault(h.doc_id, h)

        lineas += [f"## {qid}", "", f"> {consulta['text']}", ""]
        for d in docs:
            h = mejor_frag[d.doc_id]
            extracto = " ".join(h.texto.split())[:300]
            lineas += [
                f"- [ ] `{d.doc_id}` &mdash; {h.fuente}",
                f"      {extracto}...",
                "",
            ]
        lineas.append("")
        print(f"  {qid}: {len(docs)} candidatos")

    CANDIDATOS_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nescrito: {CANDIDATOS_PATH}")


PATRON_QID = re.compile(r"^##\s+(q\d{3})\s*$")
PATRON_MARCA = re.compile(r"^-\s+\[([ xX])\]\s+`([^`]+)`")


def recolectar(args: argparse.Namespace) -> None:
    if not CANDIDATOS_PATH.exists():
        print(f"no existe {CANDIDATOS_PATH} -- correr primero --generar")
        sys.exit(2)

    marcados: dict[str, list[str]] = {}
    qid = None
    for linea in CANDIDATOS_PATH.read_text(encoding="utf-8").splitlines():
        m = PATRON_QID.match(linea)
        if m:
            qid = m.group(1)
            continue
        m = PATRON_MARCA.match(linea)
        if m and qid and m.group(1).lower() == "x":
            marcados.setdefault(qid, []).append(m.group(2))

    if not marcados:
        print("ningun documento marcado con [x] -- nada que agregar")
        return

    existentes = cargar_jsonl(GROUND_TRUTH_PATH) if GROUND_TRUTH_PATH.exists() else []
    por_qid = {f["query_id"]: f for f in existentes}
    for qid, docs in marcados.items():
        # Re-anotar una consulta ya anotada la reemplaza, no la duplica.
        # `pool` deja registrado de que encoder salieron los candidatos: una
        # consulta anotada sobre el pool de X favorece a X, asi que NO sirve
        # para comparar X contra otro encoder. eval_mini puede filtrarlas.
        por_qid[qid] = {
            "query_id": qid,
            "nota": "anotado por pooling con --recolectar",
            "pool": args.encoder_name if args else None,
            "docs_relevantes": docs,
        }

    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as f:
        for qid in sorted(por_qid):
            f.write(json.dumps(por_qid[qid], ensure_ascii=False) + "\n")

    print(f"consultas anotadas en esta pasada: {len(marcados)}")
    print(f"total en el ground truth: {len(por_qid)} -> {GROUND_TRUTH_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generar", action="store_true")
    parser.add_argument("--recolectar", action="store_true")
    parser.add_argument("--solo", nargs="+", default=None, help="Limitar a estos query_id.")
    parser.add_argument("--encoder-name", default=ENCODER_PRIMARY_NAME)
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.recolectar:
        recolectar(args)
    else:
        generar(args)


if __name__ == "__main__":
    main()
