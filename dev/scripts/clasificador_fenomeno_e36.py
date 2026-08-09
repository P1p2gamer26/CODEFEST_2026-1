#!/usr/bin/env python3
"""E36, FASE 1: clasificar el fenomeno de cada consulta con el TEXTO, no con el pool.

Construido SOLO con frecuencias del CORPUS. No mira el ground truth en ningun
punto: la tabla que decide es `df[termino][fenomeno]`, contada por streaming
sobre `metadata.jsonl`, exactamente como hizo E35.

Regla (sin parametros que calibrar):

    tasa_f(t) = df_f,es(t)/N_f,es + df_f,en(t)/N_f,en    # una vez por idioma
    p_f(t)    = tasa_f(t) / sum_g tasa_g(t)              # de quien es el termino
    score_f(consulta) = media de p_f(t) sobre los terminos de contenido

Un termino ubicuo reparte 1/3 a cada fenomeno y no inclina nada; uno propio de
un fenomeno se lo lleva casi entero.

POR QUE LA TASA VA POR IDIOMA, y no df_f(t)/N_f a secas. Los fenomenos 1 y 2
son corpus en INGLES (27k y 26k chunks en) con una minoria en espanol (5,6k),
y el 3 es en espanol. Sin separar por idioma, cualquier palabra espanola
generica ("estados", "capacidad") es densisima en el 3 y casi ausente de los
otros dos: la version sin separar clasifica 39 de 50 consultas como fenomeno 3
y clava el bloque tematico del enunciado en 25/50. Es un efecto de IDIOMA, no
de tema. Normalizando dentro de cada idioma se compara lo comparable.
El ajuste se hizo contra el bloque tematico del enunciado (q001-q016 IA,
q017-q032 espacio, q033-q050 territorio), que sale de LEER las consultas; el
ground truth no se toco en ningun momento.

El tokenizador es `tokens_de` de truncate.py, el mismo que ya usa el gate de
cobertura: no se reimplementa.

    .venv/Scripts/python.exe dev/scripts/clasificador_fenomeno_e36.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, encoder_dir  # noqa: E402
from src.retrieval.truncate import tokens_de  # noqa: E402

META = encoder_dir("multilingual-e5-base") / "metadata.jsonl"
CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
TABLA = DEV_DIR / "intermedios" / "fenomeno_texto_e36.json"
FENOMENOS = (1, 2, 3)
IDIOMAS = ("es", "en")


def construir_tabla() -> dict:
    """df por (fenomeno, idioma) de los terminos de las 50 consultas. Solo corpus."""
    vocab = set()
    for linea in CONSULTAS.open(encoding="utf-8"):
        vocab |= tokens_de(json.loads(linea)["text"])

    df = {t: Counter() for t in vocab}
    n_celda = Counter()
    n = 0
    with META.open(encoding="utf-8") as fh:
        for linea in fh:
            reg = json.loads(linea)
            fen, idi = reg.get("fenomeno"), reg.get("idioma")
            if fen not in FENOMENOS or idi not in IDIOMAS:
                continue
            n_celda[f"{fen}-{idi}"] += 1
            for t in tokens_de(reg["texto"]) & vocab:
                df[t][f"{fen}-{idi}"] += 1
            n += 1
            if n % 20000 == 0:
                print(f"  {n:,} chunks...", file=sys.stderr, flush=True)
    celdas = [f"{f}-{i}" for f in FENOMENOS for i in IDIOMAS]
    print(f"{n:,} chunks usados; celdas {dict(n_celda)}", file=sys.stderr)
    return {"n_celda": {c: n_celda[c] for c in celdas},
            "df": {t: {c: df[t][c] for c in celdas if df[t][c]} for t in sorted(vocab)}}


def cargar_tabla() -> dict:
    if not TABLA.exists():
        tabla = construir_tabla()
        TABLA.write_text(json.dumps(tabla), encoding="utf-8")
    return json.loads(TABLA.read_text(encoding="utf-8"))


def clasificar(texto: str, tabla: dict) -> tuple[int, dict, int]:
    """Devuelve (fenomeno, scores, terminos usados). Sin ground truth de por medio."""
    n_celda = tabla["n_celda"]
    df = tabla["df"]
    acum = defaultdict(float)
    usados = 0
    for t in tokens_de(texto):
        fila = df.get(t)
        if not fila:
            continue
        tasa = {f: sum(fila.get(f"{f}-{i}", 0) / n_celda[f"{f}-{i}"]
                       for i in IDIOMAS if n_celda[f"{f}-{i}"])
                for f in FENOMENOS}
        total = sum(tasa.values())
        if total <= 0:
            continue
        for f in FENOMENOS:
            acum[f] += tasa[f] / total
        usados += 1
    if not usados:
        return 0, {}, 0
    scores = {f: acum[f] / usados for f in FENOMENOS}
    return max(scores, key=scores.get), scores, usados


def clasificar_las_50(tabla: dict) -> dict:
    out = {}
    for linea in CONSULTAS.open(encoding="utf-8"):
        c = json.loads(linea)
        fen, scores, usados = clasificar(c["text"], tabla)
        out[c["query_id"]] = {"fenomeno": fen, "scores": scores, "terminos": usados,
                              "dominancia": max(scores.values()) if scores else 0.0}
    return out


def _autochequeo(tabla: dict) -> None:
    """Un termino inventado no debe romper; uno ubicuo no debe inclinar."""
    assert clasificar("zzzzqqqq", tabla)[2] == 0
    fen, scores, _ = clasificar("desechos orbitales satelites", tabla)
    assert fen == 2, scores
    fen, _, _ = clasificar("restitucion de tierras despojo campesinos", tabla)
    assert fen == 3, fen


def main() -> None:
    tabla = cargar_tabla()
    _autochequeo(tabla)
    cls = clasificar_las_50(tabla)
    print(f"{'qid':6s}{'fen':>5s}{'dom':>7s}{'terms':>7s}   scores")
    for q in sorted(cls):
        c = cls[q]
        s = " ".join(f"{f}:{c['scores'].get(f, 0):.3f}" for f in FENOMENOS)
        print(f"{q:6s}{c['fenomeno']:>5d}{c['dominancia']:>7.3f}{c['terminos']:>7d}   {s}")
    esperado = {**{f"q{i:03d}": 1 for i in range(1, 17)},
                **{f"q{i:03d}": 2 for i in range(17, 33)},
                **{f"q{i:03d}": 3 for i in range(33, 51)}}
    ok = sum(1 for q in cls if cls[q]["fenomeno"] == esperado[q])
    print(f"\ncoincide con el bloque tematico de la consulta: {ok}/50")
    print("  (el bloque q001-q016 / q017-q032 / q033-q050 sale del ENUNCIADO,")
    print("   no del ground truth: es lectura de las consultas, no de las etiquetas)")
    for q in sorted(cls):
        if cls[q]["fenomeno"] != esperado[q]:
            print(f"  {q}: dice {cls[q]['fenomeno']}, el bloque es {esperado[q]}")


if __name__ == "__main__":
    main()
