#!/usr/bin/env python3
"""E35, FASE 1: auditoria termino por termino de las 50 consultas oficiales.

Cuenta, por STREAMING sobre `metadata.jsonl`, en cuantos chunks aparece cada
termino de dominio en espanol y su forma inglesa. No mira el ground truth ni
metrica alguna: es una propiedad del corpus.

El desglose por fenomeno importa porque E03 ya midio que la asimetria ES/EN
existe SOLO en los fenomenos 1 y 2; en el 3 va al reves.

    .venv/Scripts/python.exe dev/scripts/auditoria_glosario_e35.py
"""

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import encoder_dir  # noqa: E402
from src.retrieval.glosario import GLOSARIO, _normalizar  # noqa: E402

META = encoder_dir("multilingual-e5-base") / "metadata.jsonl"

# (etiqueta, forma ES, forma EN, consultas que la usan). Sacados leyendo las
# 50 consultas una por una y quedandose con los sintagmas de DOMINIO (no las
# palabras funcionales). Los que ya estan en la tabla se marcan aparte.
PARES = [
    # --- fenomeno 1 (IA y capacidades estrategicas) ---
    ("nbqr", "nbqr", "cbrn", "q001"),
    ("inteligencia artificial", "inteligencia artificial", "artificial intelligence", "q001+"),
    ("sistemas no tripulados", "sistemas no tripulados", "unmanned systems", "q002"),
    ("operaciones militares", "operaciones militares", "military operations", "q003 q009"),
    ("talento especializado", "talento especializado", "ai talent", "q004"),
    ("escasez de talento", "escasez de talento", "talent shortage", "q004"),
    ("seguridad nacional", "seguridad nacional", "national security", "q005"),
    ("inteligencia militar", "inteligencia militar", "military intelligence", "q006"),
    ("sistemas autonomos", "sistemas autonomos", "autonomous systems", "q007"),
    ("armas autonomas", "armas autonomas", "autonomous weapons", "q007"),
    ("derecho internacional humanitario", "derecho internacional humanitario",
     "international humanitarian law", "q007"),
    ("enjambres de drones", "enjambres de drones", "drone swarm", "q008"),
    ("conflictos armados", "conflictos armados", "armed conflict", "q009"),
    ("contra drones", "neutralizacion de drones", "counter-drone", "q010"),
    ("fuerzas militares", "fuerzas militares", "armed forces", "q011"),
    ("amenazas ciberneticas", "amenazas ciberneticas", "cyber threats", "q013"),
    ("infraestructuras criticas", "infraestructuras criticas", "critical infrastructure", "q013"),
    ("infraestructura de computo", "infraestructura de computo", "computing infrastructure",
     "q014"),
    ("computo avanzado", "computo avanzado", "advanced computing", "q014"),
    ("semiconductores", "semiconductores", "semiconductor", "q015"),
    ("hardware especializado", "hardware especializado", "specialized hardware", "q015"),
    ("drones de bajo costo", "drones de bajo costo", "low-cost drones", "q016"),
    # --- fenomeno 2 (espacio) ---
    ("derecho internacional en el espacio", "derecho internacional en el espacio",
     "international space law", "q017"),
    ("contraespacial", "contraespacial", "counterspace", "q018"),
    ("sistemas satelitales", "sistemas satelitales", "satellite systems", "q018 q024"),
    ("guerra electronica", "guerra electronica", "electronic warfare", "q019"),
    ("interferencia", "interferir sistemas espaciales", "jamming", "q019"),
    ("spoofing", "suplantacion", "spoofing", "q020"),
    ("servicios satelitales", "servicios satelitales", "satellite services", "q020"),
    ("maniobras de proximidad", "maniobras de proximidad",
     "rendezvous and proximity operations", "q021"),
    ("energia dirigida", "energia dirigida", "directed energy", "q022"),
    ("infraestructura satelital", "infraestructura satelital", "satellite infrastructure",
     "q023"),
    ("vulnerabilidades ciberneticas", "vulnerabilidades ciberneticas",
     "cyber vulnerabilities", "q023"),
    ("capacidades laser", "capacidades laser", "laser weapons", "q024"),
    ("antisatelite", "antisatelite", "anti-satellite", "q025 q026"),
    ("desechos orbitales", "desechos orbitales", "space debris", "q026"),
    ("operaciones espaciales", "operaciones espaciales", "space operations", "q027 q030"),
    ("reabastecimiento en orbita", "reabastecimiento en orbita", "on-orbit servicing", "q028"),
    ("satelites rusos", "satelites rusos", "russian satellites", "q029"),
    ("operaciones espaciales dinamicas", "operaciones espaciales dinamicas",
     "dynamic space operations", "q030"),
    ("corea del norte", "corea del norte", "north korea", "q031"),
    ("capacidades espaciales", "capacidades espaciales", "space capabilities", "q031"),
    ("dominio espacial", "dominio espacial", "space domain", "q032"),
    ("rusia ucrania", "ucrania", "ukraine", "q032"),
    # --- fenomeno 3 (territorial): NO son candidatos, se cuentan para
    # confirmar que la asimetria va al reves y cerrar el eje por escrito.
    ("control territorial", "control territorial", "territorial control", "q033 q039"),
    ("grupos armados", "grupos armados", "armed groups", "q033+"),
    ("economias ilicitas", "economias ilicitas", "illicit economies", "q035 q041"),
    ("crimen organizado", "crimen organizado", "organized crime", "q037"),
    ("mineria ilegal de oro", "mineria ilegal", "illegal gold mining", "q042"),
    ("narcotrafico", "narcotrafico", "drug trafficking", "q043"),
    ("reclutamiento", "reclutamiento", "child recruitment", "q044"),
    ("homicidios selectivos", "homicidios selectivos", "targeted killings", "q045"),
    ("minerales estrategicos", "minerales estrategicos", "critical minerals", "q046"),
    ("restitucion de tierras", "restitucion de tierras", "land restitution", "q049"),
]


def main() -> None:
    formas = sorted({f for _, es, en, _ in PARES for f in (es, en)})
    cuenta = Counter()
    por_fen = {f: Counter() for f in formas}
    n = 0
    with META.open(encoding="utf-8") as fh:
        for linea in fh:
            reg = json.loads(linea)
            texto = _normalizar(reg["texto"])
            fen = reg.get("fenomeno")
            for f in formas:
                if f in texto:
                    cuenta[f] += 1
                    por_fen[f][fen] += 1
            n += 1
            if n % 20000 == 0:
                print(f"  {n:,} chunks...", file=sys.stderr, flush=True)
    print(f"{n:,} chunks leidos\n")

    en_tabla = {t for t, _ in GLOSARIO}
    print(f"{'termino ES':36s}{'ES':>7s}{'forma EN':32s}{'EN':>7s}{'x':>7s}"
          f"  {'tabla':5s} consultas")
    for etiqueta, es, en, qids in PARES:
        ces, cen = cuenta[es], cuenta[en]
        ratio = cen / ces if ces else float("inf")
        marca = "SI" if any(t in _normalizar(es) or _normalizar(es) in t
                            for t in en_tabla) else "-"
        print(f"{es[:35]:36s}{ces:>7d}{en[:31]:32s}{cen:>7d}"
              f"{(f'{ratio:.0f}' if ces else 'inf'):>7s}  {marca:5s} {qids}")

    print("\n--- reparto por fenomeno de las formas inglesas (1/2/3) ---")
    for etiqueta, es, en, qids in PARES:
        d = por_fen[en]
        e = por_fen[es]
        print(f"{es[:32]:33s} ES {e[1]:>5d}/{e[2]:>5d}/{e[3]:>5d}   "
              f"EN {d[1]:>5d}/{d[2]:>5d}/{d[3]:>5d}   {en}")


if __name__ == "__main__":
    main()
