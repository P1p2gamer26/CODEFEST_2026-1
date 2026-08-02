"""Repara los guiones de fin de linea de los chunks ya extraidos.

El corpus se extrajo antes de que `src/cleaning/clean.py` supiera reparar el
guion de fin de linea de los PDF (U+FFFE), asi que 38.397 de los 128.526
chunks llevan palabras partidas. Re-extraer el corpus entero costaria horas
de OCR para nada: el defecto es puramente textual y se arregla sobre el
checkpoint `intermedios/chunks_intermedios.jsonl`.

IMPORTANTE: no re-chunkea. El numero de chunks, su orden y sus `chunk_id`
quedan intactos -- es el invariante que permite que la cascada de dos
encoders lea el vector del secundario por `reconstruct(fila)`.

    python dev/scripts/reparar_guiones.py

Deja `intermedios/chunks_intermedios_limpio.jsonl`, que se le pasa a
`build_corpus_index.py --desde-chunks`.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning.clean import _CORTE_RE, clean_text  # noqa: E402
from src.config import INTERMEDIOS_DIR  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402

ENTRADA = INTERMEDIOS_DIR / "chunks_intermedios.jsonl"
SALIDA = INTERMEDIOS_DIR / "chunks_intermedios_limpio.jsonl"


def construir_mapa(ruta: Path) -> dict[tuple[str, str], str]:
    """Desempata cada corte contando en el propio corpus la forma unida
    ("satellites") contra la forma con guion ("open-source"). Gana la mas
    frecuente; sin ninguna de las dos, la entrada no se emite y `clean_text`
    cae en su heuristica."""
    # Una sola pasada: los cortes por un lado, y por otro el vocabulario del
    # corpus contando "satellites" y "open-source" como un token cada uno.
    # Las palabras partidas no ensucian ese conteo, porque el separador
    # U+FFFE no es \w y parte el token en dos.
    palabra = re.compile(r"\w+(?:-\w+)?")
    cortes: Counter[tuple[str, str]] = Counter()
    vocab: Counter[str] = Counter()
    for linea in ruta.open(encoding="utf-8"):
        texto = json.loads(linea)["texto"]
        for izq, der in _CORTE_RE.findall(texto):
            if izq and der:
                cortes[(izq.lower(), der.lower())] += 1
        vocab.update(palabra.findall(texto.lower()))

    mapa: dict[tuple[str, str], str] = {}
    for izq, der in cortes:
        unida, guionada = vocab[izq + der], vocab[f"{izq}-{der}"]
        if unida or guionada:
            mapa[(izq, der)] = "" if unida >= guionada else "-"
    return mapa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    ap.add_argument("--encoder-name", default=None, help="para recalcular num_tokens")
    args = ap.parse_args()

    print(f"Construyendo el mapa de desempate desde {args.entrada}...")
    mapa = construir_mapa(args.entrada)
    con_guion = sum(1 for v in mapa.values() if v == "-")
    print(f"  {len(mapa)} cortes decididos con evidencia del corpus ({con_guion} con guion)")

    encoder = get_encoder(args.encoder_name) if args.encoder_name else None
    n = tocados = 0
    with args.salida.open("w", encoding="utf-8") as salida:
        for linea in args.entrada.open(encoding="utf-8"):
            record = json.loads(linea)
            limpio = clean_text(record["texto"], mapa)
            if limpio != record["texto"]:
                tocados += 1
                record["texto"] = limpio
                if encoder is not None:
                    record["num_tokens"] = encoder.count_tokens(limpio)
            salida.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"{n} chunks escritos en {args.salida}; {tocados} reparados")


if __name__ == "__main__":
    main()
