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

Ese sesgo dejo de ser teorico: las 9 consultas que faltan del ground truth
(q001 q007 q008 q011 q012 q015 q028 q038 q048) son exactamente las 9 que
sacaron CERO marcas. No es que nadie las mirara, es que ninguno de los 10
candidatos densos era relevante -- y el material si esta en el corpus (CBRN
66 chunks, "on-orbit servicing" 114, "Clan del Golfo" 172). Para eso esta
--rescate: agrega al pool los documentos que BM25 encuentra por el termino
discriminante y el encoder denso se pierde. Es una herramienta de ANOTACION,
no un cambio del sistema entregado.

Uso:
    python dev/scripts/anotar_candidatos.py --generar
    python dev/scripts/anotar_candidatos.py --generar --rescate --solo q001 q038
    python dev/scripts/anotar_candidatos.py --recolectar
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, ENCODER_PRIMARY_NAME  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.aggregate import aggregate_documents  # noqa: E402
from src.retrieval.bm25 import BM25  # noqa: E402
from src.retrieval.search import search  # noqa: E402

CONSULTAS_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"
CANDIDATOS_PATH = DEV_DIR / "eval" / "candidatos.md"

N_CANDIDATOS = 10  # por consulta
K_POOL = 200  # pool amplio: la anotacion debe ver mas alla de lo que entrega
N_RESCATE = 5  # documentos lexicos extra por consulta, ademas de los densos

# La procedencia del pool viaja en el .md para que --recolectar no dependa de
# que el anotador vuelva a pasar los mismos flags. eval_mini.py usa el campo
# `pool` para separar las consultas anotadas de forma independiente.
PATRON_POOL = re.compile(r"^<!--\s*pool:\s*(\S+)\s*-->$")


def cargar_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def etiqueta_pool(args: argparse.Namespace) -> str:
    return f"{args.encoder_name}+bm25" if args.rescate else args.encoder_name


def generar(args: argparse.Namespace) -> None:
    consultas = cargar_jsonl(CONSULTAS_PATH)
    ya_anotadas = {f["query_id"] for f in cargar_jsonl(GROUND_TRUTH_PATH)} if GROUND_TRUTH_PATH.exists() else set()
    pendientes = [c for c in consultas if c["query_id"] not in ya_anotadas]
    if args.solo:
        # Nombrar una consulta explicitamente con --solo la genera AUNQUE ya
        # este anotada: es la unica forma de rehacer a mano las 9 que quedaron
        # con etiquetas del la anotacion asistida (F1 0.23 contra el humano). Al
        # recolectar, el registro se reemplaza entero, asi que el campo
        # `anotador` desaparece solo y eval_mini deja de contarlas como
        # debiles.
        solo = set(args.solo)
        pendientes = [c for c in consultas if c["query_id"] in solo]
    if not pendientes:
        print("no hay consultas pendientes de anotar")
        return

    encoder = get_encoder(name=args.encoder_name)
    index, metadata = load_index(args.encoder_name, index_dir=args.index_dir)
    print(f"encoder: {encoder.name} -> {index.ntotal} vectores")

    terminos_extra = dict(par.split("=", 1) for par in args.terminos or [])
    desconocidos = set(terminos_extra) - {c["query_id"] for c in pendientes}
    if desconocidos:
        print(f"--terminos menciona consultas que no estan pendientes: {sorted(desconocidos)}")
        sys.exit(2)

    bm25 = None
    if args.rescate:
        t0 = time.perf_counter()
        bm25 = BM25(metadata)
        print(f"BM25: {len(bm25.postings)} terminos en {time.perf_counter() - t0:.1f} s")

    # doc_id -> (fuente, mejor fragmento) para dar contexto al anotador
    lineas = [
        "# Candidatos para anotar el ground truth",
        "",
        f"<!-- pool: {etiqueta_pool(args)} -->",
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
        candidatos = [(d.doc_id, "denso") for d in docs]

        # BM25 sobre la consulta cruda solo rescata cuando el termino raro de
        # la consulta aparece LITERAL en el corpus. Medido sobre las 9: sirve
        # para q038 (MAPP/OEA, "Clan del Golfo") y a medias para q015
        # ("semiconductor" es cognado), y NO sirve para q001 ni q028, donde la
        # consulta esta en espanol y el corpus en ingles (NBQR/CBRN,
        # "reabastecimiento en orbita"/"on-orbit servicing"). No hay puente
        # lexico en el dato y traducirlo pediria un recurso que la sec. 8.3
        # prohibe, asi que el termino lo pone quien anota con --terminos.
        #
        # El termino REEMPLAZA a la consulta, no se le suma. Medido en q001:
        # "CBRN" solo trae los CSET/SIPRI correctos, mientras que la consulta
        # completa mas "CBRN" no trae ninguno. Una consulta de 20 palabras
        # comunes acumula mas score que un termino de idf alto que aparece en
        # 66 chunks, asi que concatenar equivale a no haber puesto el termino.
        texto_lexico = terminos_extra.get(qid) or consulta["text"]
        if bm25 is not None:
            # Union por DOCUMENTO, sin fusionar scores: aca no hay que rankear,
            # hay que no perder candidatos. Mezclar un coseno (~0.4) con un
            # BM25 (~20) es el error documentado en dev/docs/PLAN_MAESTRO.md -- la magnitud
            # de BM25 decidiria toda la lista.
            ya = {d.doc_id for d in docs}
            lexicos = bm25.search(texto_lexico, k=K_POOL)
            for h in lexicos:
                mejor_frag.setdefault(h.doc_id, h)
            for d in aggregate_documents(lexicos, top_n=N_CANDIDATOS + N_RESCATE, strategy="sum"):
                if d.doc_id not in ya:
                    candidatos.append((d.doc_id, "lexico"))
                    ya.add(d.doc_id)
                    if sum(1 for _, o in candidatos if o == "lexico") >= N_RESCATE:
                        break

        lineas += [f"## {qid}", "", f"> {consulta['text']}", ""]
        for doc_id, origen in candidatos:
            h = mejor_frag[doc_id]
            extracto = " ".join(h.texto.split())[:300]
            marca = "" if origen == "denso" else " _(rescate lexico)_"
            lineas += [
                f"- [ ] `{doc_id}` &mdash; {h.fuente}{marca}",
                f"      {extracto}...",
                "",
            ]
        lineas.append("")
        n_lex = sum(1 for _, o in candidatos if o == "lexico")
        print(f"  {qid}: {len(candidatos)} candidatos ({n_lex} por rescate lexico)")

    CANDIDATOS_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\nescrito: {CANDIDATOS_PATH}")


PATRON_QID = re.compile(r"^##\s+(q\d{3})\s*$")
PATRON_MARCA = re.compile(r"^-\s+\[([ xX])\]\s+`([^`]+)`")


def recolectar(args: argparse.Namespace) -> None:
    if not CANDIDATOS_PATH.exists():
        print(f"no existe {CANDIDATOS_PATH} -- correr primero --generar")
        sys.exit(2)

    # Una consulta VISTA y dejada sin marcar no es lo mismo que una sin
    # anotar: es el dato de que el pool no contenia nada relevante. Antes las
    # dos se veian igual (simplemente no entraban al jsonl) y por eso las 9
    # consultas con cero marcas figuraban como "pendientes de anotar" durante
    # semanas. Ahora se escriben con docs_relevantes vacio.
    vistas: list[str] = []
    marcados: dict[str, list[str]] = {}
    pool = None
    qid = None
    for linea in CANDIDATOS_PATH.read_text(encoding="utf-8").splitlines():
        m = PATRON_POOL.match(linea.strip())
        if m:
            pool = m.group(1)
            continue
        m = PATRON_QID.match(linea)
        if m:
            qid = m.group(1)
            vistas.append(qid)
            continue
        m = PATRON_MARCA.match(linea)
        if m and qid and m.group(1).lower() == "x":
            marcados.setdefault(qid, []).append(m.group(2))

    if not vistas:
        print(f"ninguna consulta en {CANDIDATOS_PATH} -- correr primero --generar")
        return
    if pool is None:  # .md generado antes de que existiera la marca de pool
        pool = args.encoder_name if args else None

    existentes = cargar_jsonl(GROUND_TRUTH_PATH) if GROUND_TRUTH_PATH.exists() else []
    por_qid = {f["query_id"]: f for f in existentes}
    for qid in vistas:
        docs = marcados.get(qid, [])
        # Re-anotar una consulta ya anotada la reemplaza, no la duplica.
        # `pool` deja registrado de que encoder salieron los candidatos: una
        # consulta anotada sobre el pool de X favorece a X, asi que NO sirve
        # para comparar X contra otro encoder. eval_mini puede filtrarlas.
        por_qid[qid] = {
            "query_id": qid,
            "nota": "anotado por pooling con --recolectar",
            "pool": pool,
            "docs_relevantes": docs,
        }

    with GROUND_TRUTH_PATH.open("w", encoding="utf-8") as f:
        for qid in sorted(por_qid):
            f.write(json.dumps(por_qid[qid], ensure_ascii=False) + "\n")

    vacias = [q for q in vistas if not marcados.get(q)]
    print(f"consultas anotadas en esta pasada: {len(vistas)} ({len(vacias)} sin ningun documento relevante)")
    if vacias:
        print(f"  vacias: {' '.join(vacias)}")
    print(f"total en el ground truth: {len(por_qid)} -> {GROUND_TRUTH_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generar", action="store_true")
    parser.add_argument("--recolectar", action="store_true")
    parser.add_argument(
        "--rescate",
        action="store_true",
        help="Agregar al pool los documentos que BM25 encuentra y el denso se pierde. "
        "Para las consultas cuyo pool denso no contiene nada relevante (el termino "
        "discriminante existe en el corpus pero el encoder no lo alcanza).",
    )
    parser.add_argument(
        "--terminos",
        nargs="+",
        default=None,
        metavar="qNNN=TERMINO",
        help="Terminos extra para la busqueda lexica de una consulta, p. ej. "
        "'q001=CBRN chemical biological'. Necesario cuando la consulta esta en "
        "espanol y el corpus nombra la misma cosa en ingles. Solo afecta a los "
        "candidatos que se proponen para anotar, no al sistema entregado.",
    )
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
