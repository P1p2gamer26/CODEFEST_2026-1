#!/usr/bin/env python3
"""E24: normalizacion de `fuente` a texto codificable, y su auditoria de puerta.

El nombre del archivo es lo unico del corpus que nombra el capitulo, y nunca
entro a un vector (el indice codifica solo `texto`). Aca se convierte en una
frase: se quita la extension, el prefijo de coleccion, las fechas y los
identificadores, y se separan los tokens pegados por `_`/`-`.

Un nombre con menos de MIN_TOKENS tokens de contenido es OPACO (`informes01`,
los tiles de zoom): no se puede puntuar y recibe score neutro.

Uso:
    python dev/scripts/nombres_doc_e24.py     # auditoria de la puerta
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR  # noqa: E402

MIN_TOKENS = 3

# Palabras que no dicen de que trata el documento: extensiones, formatos,
# ruido de scraping. Se fija ANTES de medir y no se toca segun el resultado.
_RUIDO = {
    "pdf", "json", "csv", "xlsx", "txt", "jpg", "pbf", "html", "htm",
    "informe", "informes", "report", "reports", "doc", "docs", "file",
    "final", "web", "www", "com", "org", "http", "https", "download",
    "page", "pages", "vol", "no", "num", "part", "en", "es", "pt", "de",
    "la", "el", "los", "las", "del", "and", "the", "of", "for", "a", "an",
}


def normalizar(fuente: str) -> str:
    """`AIINDEX_ai-index-2025-ch7-policy-governance.pdf` -> frase codificable."""
    s = Path(fuente).stem
    # El prefijo de coleccion va en MAYUSCULAS antes del primer `_`: es el
    # nombre de la serie, comun a todos sus hermanos, o sea que no discrimina.
    s = re.sub(r"^[A-Z0-9]{2,}_", "", s)
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)   # camelCase
    toks = []
    for t in s.lower().split():
        t = re.sub(r"[^a-z0-9]", "", t)
        # Umbral en 2 y no en 3: "ai" es el termino de dominio del fenomeno 1
        # y "ai-index" es media coleccion. Corregido ANTES de puntuar nada.
        if not t or t.isdigit() or len(t) < 2:
            continue          # fechas, indices de tile, numeracion
        if re.fullmatch(r"[a-z]{1,3}\d+", t):
            continue          # ch7, v2, p012
        if t in _RUIDO:
            continue
        toks.append(t)
    return " ".join(toks)


def mapa_fuentes(metadata) -> dict:
    """doc_id -> (fuente, nombre normalizado). Un doc_id tiene una sola fuente."""
    out = {}
    for m in metadata:
        if m["doc_id"] not in out:
            out[m["doc_id"]] = (m["fuente"], normalizar(m["fuente"]))
    return out


def auditar(mapa) -> dict:
    """Cuantos documentos puede puntuar realmente el criterio."""
    info = {d: v for d, v in mapa.items() if len(v[1].split()) >= MIN_TOKENS}
    por_col = defaultdict(lambda: [0, 0])
    for d, (f, n) in mapa.items():
        col = f.split("_")[0] if "_" in f else "(sin prefijo)"
        por_col[col][0] += 1
        por_col[col][1] += int(d in info)
    print(f"{len(mapa):,} documentos con `fuente`, "
          f"{len({v[0] for v in mapa.values()}):,} nombres distintos")
    print(f"INFORMATIVOS (>= {MIN_TOKENS} tokens de contenido): "
          f"{len(info):,} ({len(info)/len(mapa):.1%})")
    print(f"OPACOS (score neutro):                {len(mapa)-len(info):,}\n")
    print(f"{'coleccion':22s} {'docs':>6s} {'informativos':>13s}")
    for col, (n, ok) in sorted(por_col.items(), key=lambda kv: -kv[1][0]):
        print(f"  {col:20s} {n:6d} {ok:13d}")
    print("\n  ejemplos informativos:")
    for d in list(info)[:5]:
        print(f"    {mapa[d][0][:70]:72s} -> {mapa[d][1][:60]}")
    print("  ejemplos opacos:")
    for d in [d for d in mapa if d not in info][:5]:
        print(f"    {mapa[d][0][:70]:72s} -> {mapa[d][1][:60]!r}")
    return info


def main() -> None:
    import json
    ruta = (DEV_DIR.parent / "Entrega" / "base_vectorial"
            / "encoder_multilingual-e5-base" / "metadata.jsonl")
    with ruta.open(encoding="utf-8") as f:
        meta = [json.loads(l) for l in f]
    auditar(mapa_fuentes(meta))


if __name__ == "__main__":
    main()
