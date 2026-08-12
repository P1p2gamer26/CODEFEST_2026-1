"""Smoke test de BAAI/bge-reranker-v2-m3 como re-puntuador cross-encoder (E39).

Verifica, antes de gastar GPU en el arnes completo:
  1. Que el modelo carga y emite scores acotados (sigmoid -> [0,1]).
  2. Sanidad sobre q001 (consulta ES, corpus EN, tema NBQR/CBRN): el chunk
     relevante en ingles (informe DHS sobre IA + amenazas quimicas/biologicas/
     radiological/nucleares) debe puntuar sobre uno claramente fuera de tema
     (laser espacial / counterspace), y los probes en espanol deben quedar
     separados igual de claro.
  3. Sensibilidad posicional: revolver el orden de las palabras debe cambiar
     el score (la trampa de gte era codificar sin informacion posicional).
  4. Control de truncacion: el mismo par a max_length 512 vs 8192.
  5. Control de direccion: (consulta, documento) vs (documento, consulta).
  6. Costo online: ms/pair en CPU con lotes realistas; extrapolacion a
     50 consultas x 100 pares.

Uso: python scripts/smoke_cross_encoder.py [--device cpu|cuda] [--max-length 512|8192]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "dev" / "intermedios" / "pools_entregados.json"
MODEL_ID = "BAAI/bge-reranker-v2-m3"

Q001 = ("C\u00f3mo est\u00e1 transformando la inteligencia artificial la "
        "capacidad de los Estados para prevenir, detectar y contrarrestar "
        "amenazas NBQR?")


def cargar_probes() -> list[tuple[str, str]]:
    d = json.load(open(POOLS, encoding="utf-8"))
    pool = d["pools"]["q001"]
    relevante = next(e["texto"] for e in pool if e["rank"] == 2)  # informe DHS, IA + NBQR
    irrelevante = next(
        e["texto"] for e in d["pools"]["q024"] if e["idioma"] == "en" and "laser" in e["texto"]
    )
    relevante_es = (
        "La inteligencia artificial ayuda a los Estados a prevenir y detectar "
        "amenazas quimicas, biologicas, radiol\u00f3gicas y nucleares (NBQR). "
        "El Departamento de Seguridad Nacional publico un informe sobre los "
        "riesgos en la interseccion entre la IA y las amenazas CBRN."
    )
    irrelevante_es = (
        "El ejercito brasile\u00f1o publico una directiva sobre el empleo de "
        "nuevas tecnologias en las operaciones militares y la actualizacion "
        "del estado mayor."
    )
    return [
        ("relevante EN (DHS)", relevante),
        ("irrelevante EN (laser)", irrelevante),
        ("relevante ES (probe)", relevante_es),
        ("irrelevante ES (probe)", irrelevante_es),
    ]


def revuelto(texto: str, seed: int) -> str:
    palabras = texto.split()
    return " ".join(random.Random(seed).sample(palabras, len(palabras)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    probes = cargar_probes()
    print(f"modelo: {MODEL_ID}  device: {args.device}  max_length: {args.max_length}")
    t0 = time.time()
    model = CrossEncoder(MODEL_ID, max_length=args.max_length, device=args.device)
    print(f"carga: {time.time() - t0:.1f} s")

    pares = [(Q001, texto) for _, texto in probes]
    rev_en = revuelto(probes[0][1], 42)
    rev_es = revuelto(probes[2][1], 43)
    pares += [(Q001, rev_en), (Q001, rev_es), (probes[0][1], Q001)]
    nombres = [p[0] for p in probes] + ["revuelto EN", "revuelto ES", "orden invertido"]
    t0 = time.time()
    scores = model.predict(pares, apply_sigmoid=True, batch_size=8)
    print(f"puntaje de {len(pares)} pares: {time.time() - t0:.2f} s")
    for n, s in zip(nombres, scores):
        print(f"  {n:26s} score={float(s):.4f}")

    rel = [float(s) for n, s in zip(nombres, scores) if n.startswith("relevante")]
    irr = [float(s) for n, s in zip(nombres, scores) if n.startswith("irrelevante")]
    ok_sanidad = min(rel) > max(irr)
    ok_posicional = abs(float(scores[4]) - float(scores[0])) > 1e-3 and abs(float(scores[5]) - float(scores[2])) > 1e-3
    inv = float(scores[6])
    print(f"  sanidad: {ok_sanidad}  [min relevante {min(rel):.4f} vs max irrelevante {max(irr):.4f}]")
    print(f"  sensibilidad posicional: {ok_posicional}  [EN {float(scores[4]):.4f} vs {float(scores[0]):.4f}]")
    print(f"  orden invertido (doc,query): {inv:.4f} vs (query,doc) {float(scores[0]):.4f}")

    # costo por par, lotes realistas (~1100 chars como los chunks del pool)
    n_pair = 40
    lotes = [(Q001, p[1]) for p in probes[:2]] * (n_pair // 2)
    model.predict(lotes, apply_sigmoid=True, batch_size=8)
    t0 = time.time()
    model.predict(lotes, apply_sigmoid=True, batch_size=8)
    dt = time.time() - t0
    print(f"  costo: {dt / n_pair * 1000:.0f} ms/pair (batch 8, chunks de ~1100 chars)")
    for pares_por_q in (100, 200):
        seg_q = dt / n_pair * pares_por_q
        print(f"  extrapolacion: {pares_por_q} pares/consulta -> {seg_q:.0f} s por consulta; 50 consultas -> {seg_q * 50 / 60:.1f} min")

    if not ok_sanidad or not ok_posicional:
        print("SMOKE FALLIDO: el modelo no distingue relevancia o no es sensible al orden.")
        sys.exit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
