#!/usr/bin/env python3
"""Pruebas de robustez de Entrega/generador.py, corriendolo como lo correra
el evaluador: por subprocess, desde un directorio ajeno al repo y con
PYTHONPATH vacio.

La suite de pytest cubre las funciones por separado y con encoder falso.
Esto cubre otra cosa: que el SCRIPT ENTERO aguante lo que le puedan tirar.
Cada fallo de aca seria un cero en la evaluacion, no un test rojo.

Uso:
    python dev/scripts/pruebas_robustez.py
    python dev/scripts/pruebas_robustez.py --rapido   # omite las que cargan modelos
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
GENERADOR = RAIZ / "Entrega" / "generador.py"
CONSULTAS = RAIZ / "dev" / "consultas_prueba" / "consultas_50_oficiales.jsonl"
OFICIAL = RAIZ / "Entrega" / "resultados.jsonl"

fallos = []


def ok(nombre, cond, detalle=""):
    # `detalle` solo se imprime cuando FALLA: es informacion de diagnostico y
    # mostrarla junto a un PASA hace pensar que algo anda mal.
    print(f"  {'PASA ' if cond else 'FALLA'}  {nombre}{('' if cond or not detalle else ' -- ' + detalle)}")
    if not cond:
        fallos.append(nombre)


def correr(args, cwd, espera_exito=True, timeout=3600):
    """Corre generador.py aislado: fuera del repo y sin PYTHONPATH, que es
    como lo va a correr quien reciba solo la carpeta Entrega/."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    r = subprocess.run(
        [sys.executable, str(GENERADOR), *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )
    if espera_exito and r.returncode != 0:
        print(f"      stderr: {r.stderr[-600:]}")
    return r


def validar_esquema(path):
    """Sec. 9.3: 50 lineas, 3 documentos, 10 fragmentos, <=250 palabras."""
    filas = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    problemas = []
    if len(filas) != 50:
        problemas.append(f"{len(filas)} lineas en vez de 50")
    for f in filas:
        if set(f) != {"query_id", "documents", "fragments"}:
            problemas.append(f"{f.get('query_id')}: claves {sorted(f)}")
        if len(f["documents"]) != 3:
            problemas.append(f"{f['query_id']}: {len(f['documents'])} documentos")
        if len(f["fragments"]) != 10:
            problemas.append(f"{f['query_id']}: {len(f['fragments'])} fragmentos")
        if [d["rank"] for d in f["documents"]] != [1, 2, 3]:
            problemas.append(f"{f['query_id']}: ranks de documento mal")
        if [x["rank"] for x in f["fragments"]] != list(range(1, 11)):
            problemas.append(f"{f['query_id']}: ranks de fragmento mal")
        for x in f["fragments"]:
            if len(x["text"].split()) > 250:
                problemas.append(f"{f['query_id']}#{x['rank']}: {len(x['text'].split())} palabras")
    return problemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="robustez_"))
    print(f"directorio de trabajo: {tmp}\n")

    consultas = [json.loads(l) for l in CONSULTAS.read_text(encoding="utf-8").splitlines() if l.strip()]

    print("== esquema de lo entregado (sec. 9.3) ==")
    probs = validar_esquema(OFICIAL)
    ok("resultados.jsonl cumple el esquema", not probs, "; ".join(probs[:3]))

    print("\n== el script arranca y explica ==")
    r = correr(["--help"], cwd=tmp)
    ok("--help sin importar nada del repo", r.returncode == 0)
    ok("--help documenta --rerank-encoder", "--rerank-encoder" in r.stdout)

    print("\n== entradas mal formadas: debe fallar CLARO, no con un traceback ==")
    vacio = tmp / "vacio.jsonl"
    vacio.write_text("", encoding="utf-8")
    r = correr(["--consultas", str(vacio), "--out", str(tmp / "o1.jsonl")], cwd=tmp, espera_exito=False)
    ok("archivo de consultas vacio no revienta con traceback",
       r.returncode != 0 and "Traceback" not in r.stderr,
       "sale con traceback" if "Traceback" in r.stderr else "")

    inexistente = correr(["--consultas", str(tmp / "no_existe.jsonl"), "--out", str(tmp / "o2.jsonl")],
                         cwd=tmp, espera_exito=False)
    ok("archivo inexistente no revienta con traceback",
       inexistente.returncode != 0 and "Traceback" not in inexistente.stderr)

    roto = tmp / "roto.jsonl"
    roto.write_text('{"query_id": "q001", "text": "hola"}\nesto no es json\n', encoding="utf-8")
    r = correr(["--consultas", str(roto), "--out", str(tmp / "o3.jsonl")], cwd=tmp, espera_exito=False)
    ok("JSON invalido a mitad de archivo no revienta con traceback", "Traceback" not in r.stderr)

    if args.rapido:
        print("\n(--rapido: se omiten las pruebas que cargan modelos)")
    else:
        print("\n== caminos alternativos que el evaluador podria tomar ==")
        # El fallback documentado en el mensaje de error de generador.py.
        r = correr(["--consultas", str(CONSULTAS), "--rerank-encoder", "multilingual-e5-base",
                    "--out", str(tmp / "e5.jsonl")], cwd=tmp)
        ok("--rerank-encoder multilingual-e5-base (fallback documentado)", r.returncode == 0)
        if r.returncode == 0:
            ok("  ...y su salida cumple el esquema", not validar_esquema(tmp / "e5.jsonl"))

        r = correr(["--consultas", str(CONSULTAS), "--rerank-encoder", "none",
                    "--out", str(tmp / "sin.jsonl")], cwd=tmp)
        ok("--rerank-encoder none (cascada apagada)", r.returncode == 0)
        if r.returncode == 0:
            ok("  ...y su salida cumple el esquema", not validar_esquema(tmp / "sin.jsonl"))

        print("\n== consultas distintas de las 50 oficiales ==")
        # Una sola consulta, y con caracteres que rompen tokenizadores ingenuos.
        raras = tmp / "raras.jsonl"
        raras.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
            {"query_id": "x001", "text": "¿Qué capacidades ASAT tiene China?"},
            {"query_id": "x002", "text": "AI"},
            {"query_id": "x003", "text": "a" * 3000},
            {"query_id": "x004", "text": "军事人工智能"},
            {"query_id": "x005", "text": "drones   con    espacios    raros\ty tabs"},
        ]) + "\n", encoding="utf-8")
        r = correr(["--consultas", str(raras), "--out", str(tmp / "raras_out.jsonl")], cwd=tmp)
        ok("consultas cortas, larguisimas, en chino y con espacios raros", r.returncode == 0)
        if r.returncode == 0:
            filas = [json.loads(l) for l in (tmp / "raras_out.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            por_id = {f["query_id"]: f for f in filas}
            ok("  ...devuelve 5 lineas", len(filas) == 5)
            ok("  ...10 fragmentos en todas", all(len(f["fragments"]) == 10 for f in filas))
            # x003 son 3.000 caracteres repetidos: su vector cae en un rincon
            # donde los 60 chunks del pool salen del mismo documento, asi que
            # no hay 3 documentos distintos que devolver. Limitacion conocida
            # y AVISADA por el script; no se da con consultas reales.
            reales = [f for qid, f in por_id.items() if qid != "x003"]
            ok("  ...3 documentos en las consultas realistas",
               all(len(f["documents"]) == 3 for f in reales))
            ok("  ...la degenerada avisa en vez de callarse",
               "solo 1 documento" in r.stderr or len(por_id["x003"]["documents"]) == 3,
               "no aviso")
            ok("  ...ningun fragmento supera 250 palabras",
               all(len(x["text"].split()) <= 250 for f in filas for x in f["fragments"]))

        print("\n== reproducibilidad ==")
        r = correr(["--consultas", str(CONSULTAS), "--out", str(tmp / "repro.jsonl")], cwd=tmp)
        ok("corre sin flags desde fuera del repo", r.returncode == 0)
        if r.returncode == 0:
            import hashlib
            h1 = hashlib.sha256(OFICIAL.read_bytes()).hexdigest()
            h2 = hashlib.sha256((tmp / "repro.jsonl").read_bytes()).hexdigest()
            ok("reproduce resultados.jsonl BYTE A BYTE", h1 == h2, f"{h1[:12]} vs {h2[:12]}")

    print(f"\n{'=' * 60}")
    if fallos:
        print(f"{len(fallos)} FALLO(S): {', '.join(fallos)}")
        sys.exit(1)
    print("todas las pruebas de robustez pasan")


if __name__ == "__main__":
    main()
