#!/usr/bin/env python3
"""Puertas de entrada de E22 (idioma sobre top-3) y E23 (cobertura lexica).

NO carga ningun indice FAISS: solo lee Entrega/resultados.jsonl y STREAMEA
metadata.jsonl linea a linea para sacar el idioma de los 500 chunk_id
entregados. Sirve para saber si cada experimento tiene casos sobre los que
actuar antes de gastar el arnes completo, que si necesita FAISS.

E22 -> cuantos de los 500 fragmentos son ilegibles y en que consultas.
E23 -> cuantos de los 500 no cubren ni una palabra de contenido de la
       consulta (cruda y expandida por el glosario), y en que consultas.

Uso:
    .venv/Scripts/python.exe dev/scripts/puertas_e22_e23.py
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "dev"))

from src.config import DEV_DIR  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402

RESULTADOS = RAIZ / "Entrega" / "resultados.jsonl"
METADATA = RAIZ / "Entrega" / "base_vectorial" / "encoder_multilingual-e5-base" / "metadata.jsonl"
CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GT = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

IDIOMAS_LEGIBLES = frozenset({"es", "en", "pt"})

# Definicion FIJADA ANTES DE MEDIR (riesgo nº 2 de E23): tokens alfanumericos
# de >= 4 caracteres, minusculas sin acentos, menos estas stopwords. No se
# retoca segun el resultado.
STOPWORDS = frozenset("""
para como cual cuales cuanto cuantos donde desde entre sobre segun hacia hasta
ante bajo cabe contra durante mediante salvo tras esta este estos estas esos
esas aquel aquella ellos ellas nosotros vosotros pero sino aunque porque
cuando mientras siempre nunca tambien tanto tanta menos mismo misma otro otra
otros otras cada todo toda todos todas algun alguna algunos algunas ningun
ninguna mucho mucha muchos muchas poco poca pocos pocas
that this these those with from have been they their there where which what
when while about into over under between during through after before more
most such than then them your ours will would could should because being
qué cómo cuál dónde cuáles cuántos
""".split())


def normalizar(texto: str) -> set:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]{4,}", t) if w not in STOPWORDS}


def cargar_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def idioma_de_los_entregados(chunk_ids: set) -> dict:
    """Streamea metadata.jsonl; solo retiene los chunk_id que interesan."""
    out = {}
    with METADATA.open(encoding="utf-8") as f:
        for linea in f:
            # filtro barato antes de parsear: 128k lineas de ~1 KB
            cid = linea[13 : linea.index('"', 13)] if linea.startswith('{"chunk_id":') else None
            if cid is None:
                r = json.loads(linea)
                cid = r["chunk_id"]
                if cid in chunk_ids:
                    out[cid] = r["idioma"]
            elif cid in chunk_ids:
                out[cid] = json.loads(linea)["idioma"]
            if len(out) == len(chunk_ids):
                break
    return out


def main() -> None:
    res = cargar_jsonl(RESULTADOS)
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(CONSULTAS)}
    gt = {g["query_id"]: set(g["docs_relevantes"]) for g in cargar_jsonl(GT) if g["docs_relevantes"]}

    cids = {fr["chunk_id"] for r in res for fr in r["fragments"]}
    print(f"{len(res)} consultas, {sum(len(r['fragments']) for r in res)} fragmentos, "
          f"{len(cids)} chunk_id distintos\n")

    print("leyendo idioma desde metadata.jsonl (streaming, sin FAISS)...", flush=True)
    idioma = idioma_de_los_entregados(cids)

    # ---------------- puerta de E22 ----------------
    ileg_por_q = Counter()
    reparto = Counter()
    for r in res:
        for fr in r["fragments"]:
            lg = idioma.get(fr["chunk_id"], "?")
            reparto[lg] += 1
            if lg not in IDIOMAS_LEGIBLES:
                ileg_por_q[r["query_id"]] += 1
    total_ileg = sum(ileg_por_q.values())
    print("\n=== E22: puerta de entrada ===")
    print(f"fragmentos ilegibles: {total_ileg} de 500")
    print("  reparto de idiomas:", dict(reparto.most_common()))
    print(f"  concentrados en {len(ileg_por_q)} consultas: "
          + ", ".join(f"{q}({n})" for q, n in sorted(ileg_por_q.items())))
    # cuantos fragmentos vienen del top-3 hoy (la propiedad que E22 pone en juego)
    alineados = sum(
        1 for r in res for fr in r["fragments"]
        if fr["doc_id"] in {d["doc_id"] for d in r["documents"][:3]}
    )
    print(f"  fragmentos hoy alineados al top-3: {alineados} de 500")
    # el intercambio real: ilegibles del top-3 contra legibles disponibles fuera
    ileg_en_top3 = sum(
        1 for r in res for fr in r["fragments"]
        if idioma.get(fr["chunk_id"], "?") not in IDIOMAS_LEGIBLES
        and fr["doc_id"] in {d["doc_id"] for d in r["documents"][:3]}
    )
    print(f"  de esos ilegibles, {ileg_en_top3} estan DENTRO del top-3 "
          "(son los unicos que la permutacion puede mover)")

    # ---------------- puerta de E23 ----------------
    print("\n=== E23: puerta de entrada ===")
    cero_cruda = cero_exp = 0
    cero_exp_por_q = Counter()
    cobertura_exp = []
    for r in res:
        q = consultas[r["query_id"]]
        toks_c, toks_e = normalizar(q), normalizar(expandir_consulta(q))
        for fr in r["fragments"]:
            t = normalizar(fr["text"])
            if not (t & toks_c):
                cero_cruda += 1
            n = len(t & toks_e)
            cobertura_exp.append(n)
            if n == 0:
                cero_exp += 1
                cero_exp_por_q[r["query_id"]] += 1
    print(f"fragmentos con cobertura CERO (consulta cruda):     {cero_cruda} de 500")
    print(f"fragmentos con cobertura CERO (consulta expandida): {cero_exp} de 500")
    print(f"  en {len(cero_exp_por_q)} consultas; distribucion de cobertura: "
          + str(dict(sorted(Counter(cobertura_exp).items()))))

    # el punto que E11 obliga a comprobar: hay pares que desempatar?
    # dentro de una consulta, pares de fragmentos con cobertura distinta que
    # HOY estan ordenados al reves de lo que el criterio nuevo pediria.
    inversiones = 0
    consultas_con_inversion = set()
    for r in res:
        q = consultas[r["query_id"]]
        toks_e = normalizar(expandir_consulta(q))
        cob = [len(normalizar(fr["text"]) & toks_e) for fr in r["fragments"]]
        for i in range(len(cob)):
            for j in range(i + 1, len(cob)):
                if cob[i] == 0 and cob[j] > 0:
                    inversiones += 1
                    consultas_con_inversion.add(r["query_id"])
    print(f"  pares (i<j) con cobertura 0 delante de cobertura >0: {inversiones}, "
          f"en {len(consultas_con_inversion)} consultas")
    print("  (cota SUPERIOR optimista: el criterio nuevo va CUARTO, asi que solo")
    print("   actua sobre pares que ya empatan en top-3, idioma y aparato)")

    # ---------------- contexto para los vetos ----------------
    ceros = [r["query_id"] for r in res
             if r["query_id"] in gt
             and not ({d["doc_id"] for d in r["documents"][:3]} & gt[r["query_id"]])]
    print(f"\nveto compartido: consultas con F1@3 = 0 hoy: {len(ceros)} -> {ceros}")


if __name__ == "__main__":
    main()
