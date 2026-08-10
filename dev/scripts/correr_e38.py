#!/usr/bin/env python3
"""E38 -- driver de la rejilla completa, para dejarla corriendo sin vigilancia.

Corre las seis celdas de punta a punta y en SERIE: puerta de conservacion ->
generar chunks -> indice de MiniLM en GPU -> verificar alineacion -> evaluar
con k_pool crudo y escalado. Escribe una linea de progreso por paso y un JSON
con todas las lecturas al final.

En serie a proposito: la GTX 1650 tiene 4 GB y dos builds concurrentes se
quedan sin VRAM.

Lanzarlo SIEMPRE desacoplado de la sesion (ver CLAUDE.md, "Corridas largas"):

    $repo = "C:\\Users\\Julian\\Downloads\\CODEFEST_2026-1"
    Start-Process -FilePath "$repo\\.venv\\Scripts\\python.exe" `
      -ArgumentList "$repo\\dev\\scripts\\correr_e38.py" `
      -WorkingDirectory $repo -WindowStyle Hidden `
      -RedirectStandardOutput "$repo\\dev\\intermedios\\e38_driver.log" `
      -RedirectStandardError  "$repo\\dev\\intermedios\\e38_driver.err"

Es REANUDABLE: cada paso comprueba si su salida ya existe y la saltea. Si la
corrida muere a mitad, relanzarla retoma donde iba.

Spec: dev/docs/superpowers/specs/2026-08-09-rejilla-chunking-design.md
Plan: dev/docs/superpowers/plans/2026-08-09-e38-rejilla-chunking.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "dev"
INTER = DEV / "intermedios"
PY_CPU = REPO / ".venv" / "Scripts" / "python.exe"
PY_GPU = REPO / ".venv-cuda" / "Scripts" / "python.exe"
CHUNKS_BASE = INTER / "chunks_intermedios_limpio.jsonl"
CONSULTAS = DEV / "consultas_prueba" / "consultas_50_oficiales.jsonl"
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"

# (presupuesto, solape). La celda 280/1 es el CONTROL: reproduce el chunking
# entregado, y si su conteo se aleja de 128.526 la reconstruccion altera algo.
CELDAS = [(280, 1), (280, 0), (384, 1), (384, 0), (512, 1), (512, 0)]

# k_pool se mide en chunks, no en volumen de texto: al agrandar el chunk, un
# pool de 100 abarca mas texto. Se mide con el crudo y con el escalado.
K_POOL_CRUDO = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def correr(cmd, descripcion):
    log(f"INICIO {descripcion}")
    t0 = time.time()
    res = subprocess.run([str(c) for c in cmd], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    dur = time.time() - t0
    if res.returncode != 0:
        log(f"FALLO {descripcion} (exit {res.returncode}, {dur:.0f}s)")
        log((res.stdout or "")[-1500:])
        log((res.stderr or "")[-1500:])
        return None
    log(f"OK {descripcion} ({dur:.0f}s)")
    return res.stdout or ""


def puerta_de_conservacion():
    """Sin conservacion de texto no se gasta una hora de GPU."""
    salida = correr(
        [PY_CPU, DEV / "scripts" / "rechunkear_e38.py", "--verificar",
         "--chunks", CHUNKS_BASE],
        "puerta de conservacion sobre el corpus completo")
    if salida is None:
        return False
    for linea in salida.splitlines():
        log(f"  puerta | {linea}")
        if linea.startswith("docs_con_perdida:"):
            return linea.split(":")[1].strip() == "0"
    return False


def generar_celda(pres, sol):
    destino = INTER / f"chunks_e38_{pres}_{sol}.jsonl"
    if destino.exists():
        log(f"SALTEO generacion {pres}/{sol}: ya existe")
        return destino
    ok = correr(
        [PY_CPU, DEV / "scripts" / "rechunkear_e38.py", "--generar",
         "--chunks", CHUNKS_BASE, "--presupuesto", pres, "--solape", sol,
         "--encoder-name", ENCODER, "--salida", destino],
        f"generar chunks de la celda {pres}/{sol}")
    return destino if ok is not None else None


def construir_indice(pres, sol, chunks):
    base = INTER / f"idx_e38_{pres}_{sol}"
    faiss_path = base / f"encoder_{ENCODER}" / "index.faiss"
    if faiss_path.exists():
        log(f"SALTEO build {pres}/{sol}: el indice ya existe")
        return base
    py = PY_GPU if PY_GPU.exists() else PY_CPU
    ok = correr(
        [py, DEV / "scripts" / "build_corpus_index.py", "--desde-chunks",
         chunks, "--encoder-name", ENCODER, "--out-base", base],
        f"indice de MiniLM de la celda {pres}/{sol}")
    return base if ok is not None else None


def verificar_alineacion(base):
    """Un indice desalineado da resultados sin sentido EN SILENCIO."""
    import faiss

    carpeta = base / f"encoder_{ENCODER}"
    ix = faiss.read_index(str(carpeta / "index.faiss"))
    n = sum(1 for _ in open(carpeta / "metadata.jsonl", encoding="utf-8"))
    log(f"  alineacion | vectores={ix.ntotal} metadata={n}")
    return ix.ntotal == n


def evaluar(base, pres, sol, k_pool):
    etiqueta = f"{pres}_{sol}_kp{k_pool}"
    res_path = INTER / f"res_e38_{etiqueta}.jsonl"
    if not res_path.exists():
        ok = correr(
            [PY_CPU, REPO / "Entrega" / "generador.py", "--consultas", CONSULTAS,
             "--rerank-encoder", "none", "--index-base", base,
             "--k-pool", k_pool, "--out", res_path],
            f"recuperar con la celda {etiqueta}")
        if ok is None:
            return None
    salida = correr(
        [PY_CPU, DEV / "scripts" / "eval_mini.py", "--resultados", res_path],
        f"evaluar la celda {etiqueta}")
    if salida is None:
        return None
    for linea in salida.splitlines():
        log(f"  eval {etiqueta} | {linea}")
    return salida


def main():
    log("=" * 60)
    log("E38 -- rejilla de chunking, corrida completa")
    log(f"GPU disponible: {PY_GPU.exists()}")

    if not puerta_de_conservacion():
        log("ABORTA: la reconstruccion pierde texto. No se gasta GPU.")
        return 1
    log("PUERTA ABIERTA: la reconstruccion conserva el texto")

    lecturas = {}
    for pres, sol in CELDAS:
        log("-" * 60)
        log(f"CELDA {pres}/{sol}")

        chunks = generar_celda(pres, sol)
        if chunks is None:
            continue
        n_chunks = sum(1 for _ in open(chunks, encoding="utf-8"))
        log(f"  celda {pres}/{sol}: {n_chunks} chunks")
        if (pres, sol) == (280, 1):
            desvio = abs(n_chunks - 128526) / 128526
            log(f"  CONTROL 280/1: desvio de {desvio:.2%} contra los 128.526 "
                f"entregados ({'OK' if desvio <= 0.02 else 'REVISAR'})")

        base = construir_indice(pres, sol, chunks)
        if base is None or not verificar_alineacion(base):
            log(f"  celda {pres}/{sol} DESCARTADA: indice ausente o desalineado")
            continue

        k_escalado = round(K_POOL_CRUDO * 280 / pres)
        for k_pool in sorted({K_POOL_CRUDO, k_escalado}):
            salida = evaluar(base, pres, sol, k_pool)
            lecturas[f"{pres}_{sol}_kp{k_pool}"] = {
                "chunks": n_chunks, "eval": salida,
            }
            (INTER / "e38_lecturas.json").write_text(
                json.dumps(lecturas, ensure_ascii=False, indent=2),
                encoding="utf-8")

    log("=" * 60)
    log(f"TERMINADO: {len(lecturas)} lecturas en {INTER / 'e38_lecturas.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
