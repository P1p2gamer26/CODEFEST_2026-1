#!/usr/bin/env python3
"""Construye el indice FAISS de un encoder nuevo a partir de una metadata YA
existente, sin volver a extraer ni re-chunkear.

POR QUE ASI Y NO CON build_corpus_index.py:

1. **Alineacion garantizada por construccion.** La cascada lee el vector del
   secundario con `reconstruct(fila)`, o sea que la fila N de este indice
   tiene que ser el MISMO chunk que la fila N del indice primario. Si el
   conjunto o el orden difieren, `reconstruct` devuelve el vector equivocado
   **en silencio** -- no hay excepcion, solo resultados peores. Partiendo de
   la metadata ya entregada, el conjunto y el orden son identicos por
   definicion, y la metadata de salida es copia literal de la de entrada.
   Es el invariante del chunking unico (punto 8 de las notas del proyecto).

2. **No arrastra el stack de extraccion.** `src/embedding/build_index.py`
   importa `ChunkRecord` desde `src/ingestion/pipeline.py`, que importa
   spacy, OCR y los lectores de PBF. Nada de eso hace falta para codificar
   texto que ya esta chunkeado, y en el entorno de GPU (.venv-cuda) obligaria
   a instalar ~500 MB de dependencias inertes.

Checkpoint reanudable con validacion de CONTENIDO (leccion 8): guarda filas
hechas, el chunk_id de la ultima y un hash del texto. Si cualquiera de las
tres no cuadra, el cache se descarta solo. Sin el hash, un cambio en el texto
que no cambie ni el numero de chunks ni sus ids daria el cache por bueno y
produciria un indice con los vectores del texto viejo, tras horas de computo.

Uso:
    python dev/scripts/indexar_desde_metadata.py \
        --metadata Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl \
        --encoder-name gte-multilingual-base \
        --out-dir dev/intermedios/encoder_gte-multilingual-base
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.encoders import get_encoder  # noqa: E402

LOTE = 512


def hash_textos(textos, n=2048):
    """Hash de una muestra del texto: barato y suficiente para detectar que
    el corpus cambio. Hashear 128k textos completos en cada arranque seria
    minutos de CPU por nada."""
    h = hashlib.sha256()
    paso = max(1, len(textos) // n)
    for t in textos[::paso]:
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--encoder-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="Por defecto <out-dir>/emb.npy")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache or (args.out_dir / "emb.npy")
    progreso_path = cache.with_suffix(".progreso")

    print(f"leyendo {args.metadata}", flush=True)
    filas = []
    with args.metadata.open(encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                filas.append(json.loads(linea))
    textos = [r["texto"] for r in filas]
    firma = hash_textos(textos)
    print(f"{len(filas):,} chunks, hash {firma[:12]}", flush=True)

    encoder = get_encoder(name=args.encoder_name)
    print(f"encoder {encoder.name}, dim {encoder.dim}", flush=True)

    hechas = 0
    if cache.exists() and progreso_path.exists():
        prog = json.loads(progreso_path.read_text(encoding="utf-8"))
        ok = (
            prog.get("total") == len(filas)
            and prog.get("hash") == firma
            and prog.get("dim") == encoder.dim
            and prog.get("ultimo") == filas[prog["hechas"] - 1]["chunk_id"]
        )
        if ok:
            hechas = prog["hechas"]
            print(f"reanudando desde {hechas:,}", flush=True)
        else:
            print("cache invalido (cambio el texto, el total o la dimension) -- se descarta", flush=True)

    if hechas:
        emb = np.lib.format.open_memmap(cache, mode="r+")
    else:
        emb = np.lib.format.open_memmap(cache, mode="w+", dtype="float32", shape=(len(filas), encoder.dim))

    t0 = time.perf_counter()
    inicio = hechas
    for i in range(hechas, len(filas), LOTE):
        lote = textos[i : i + LOTE]
        emb[i : i + len(lote)] = encoder.encode_passages(lote)
        emb.flush()
        hechas = i + len(lote)
        progreso_path.write_text(
            json.dumps({"hechas": hechas, "total": len(filas), "hash": firma,
                        "dim": encoder.dim, "ultimo": filas[hechas - 1]["chunk_id"]}),
            encoding="utf-8",
        )
        transcurrido = time.perf_counter() - t0
        ritmo = (hechas - inicio) / transcurrido if transcurrido > 0 else 0
        faltan = (len(filas) - hechas) / ritmo if ritmo > 0 else 0
        print(
            f"codificados {hechas:,}/{len(filas):,} "
            f"({transcurrido / 60:.1f} min, {ritmo:.1f} chunks/s, faltan ~{faltan / 3600:.2f} h)",
            flush=True,
        )

    print("construyendo el indice FAISS", flush=True)
    # IndexFlatIP con vectores normalizados = coseno exacto, igual que los
    # otros dos indices. Nada de indices aproximados: lo que se evalua es
    # justamente el recall (ver punto 2 de "Pendientes" en las notas del proyecto).
    vectores = np.asarray(emb, dtype="float32")
    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    if not np.allclose(normas, 1.0, atol=1e-3):
        print(f"  normalizando (norma media {float(normas.mean()):.4f})", flush=True)
        vectores = vectores / np.maximum(normas, 1e-12)
    index = faiss.IndexFlatIP(encoder.dim)
    index.add(vectores)

    faiss.write_index(index, str(args.out_dir / "index.faiss"))
    # Copia LITERAL: cualquier reescritura podria alterar el orden o el
    # formato y romper la alineacion que este script existe para garantizar.
    shutil.copyfile(args.metadata, args.out_dir / "metadata.jsonl")

    print(f"LISTO  {index.ntotal:,} vectores -> {args.out_dir}", flush=True)
    print(f"  verificar con: python dev/scripts/verificar_alineacion.py {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
