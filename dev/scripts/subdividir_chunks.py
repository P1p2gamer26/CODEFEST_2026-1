"""Re-chunkea el corpus ya extraido a un presupuesto de tokens mas chico y sin
solape (experimento E21).

MOTIVO. El primario `paraphrase-multilingual-MiniLM-L12-v2` trunca en 128
tokens y la mediana de los chunks entregados es de 256: no lee la segunda
mitad de la mitad del corpus. Y el solape de una oracion entre chunks
consecutivos es la causa por la que concatenar vecinos para llenar las 250
palabras del fragmento duplicaba texto y se revirtio.

NO re-extrae ni pasa OCR: parte de `chunks_intermedios_limpio.jsonl`, que ya
trae el texto limpio y los guiones reparados. Reconstruye el texto de cada
documento juntando sus chunks, le quita el solape que el chunker viejo metio,
y vuelve a empaquetar con `_pack_sentences` -- el mismo empaquetador del
pipeline, no una copia.

    python dev/scripts/subdividir_chunks.py
    python dev/scripts/subdividir_chunks.py --autoprueba   # chequeo sin corpus

Deja `intermedios/chunks_128.jsonl`, que se le pasa a
`build_corpus_index.py --desde-chunks`.

OJO: el corpus resultante es OTRO corpus, no una reparacion del viejo. Los
`chunk_id` cambian, asi que sus indices no se pueden mezclar con los de
`Entrega/` (invariante del chunking unico).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.chunker import _pack_sentences  # noqa: E402
from src.chunking.sentence_split import split_sentences  # noqa: E402
from src.config import INTERMEDIOS_DIR  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402

ENTRADA = INTERMEDIOS_DIR / "chunks_intermedios_limpio.jsonl"
SALIDA = INTERMEDIOS_DIR / "chunks_128.jsonl"

# Cada fila de un CSV/XLSX es una unidad atomica que el chunker viejo indexo
# entera (ver `_chunk_tabular_rows`): no lleva solape ni tiene sentido
# subdividirla, asi que estos formatos pasan tal cual.
FORMATOS_ATOMICOS = ("csv", "xlsx")


def _sin_solape(chunks, lang):
    """Oraciones del documento, quitando el solape que dejo el chunker viejo.

    `CHUNK_OVERLAP_SENTENCES = 1`, o sea que cada chunk repite la ultima
    oracion del anterior. Se descartan las oraciones iniciales que ya estan al
    final de lo acumulado; se miran las dos ultimas por si el solape se
    configuro mas ancho alguna vez.
    """
    oraciones: list[str] = []
    for texto in chunks:
        nuevas = split_sentences(texto, lang)
        i = 0
        while i < len(nuevas) and nuevas[i] in oraciones[-2:]:
            i += 1
        oraciones.extend(nuevas[i:])
    return oraciones


def subdividir_documento(registros, count_tokens, presupuesto, solape):
    """Registros de un documento -> registros nuevos, re-numerados."""
    base = registros[0]
    if base["formato"] in FORMATOS_ATOMICOS:
        return registros

    oraciones = _sin_solape([r["texto"] for r in registros], base.get("idioma"))
    if not oraciones:
        return []

    # ponytail: se empaqueta el documento entero de corrido, no seccion por
    # seccion. El texto de los chunks ya no trae los encabezados Markdown, asi
    # que las secciones no son recuperables aca; el precio es que un chunk
    # puede cruzar una frontera de seccion y que `titulo_seccion` se pierde
    # (no es metadata obligatoria de la Tabla 1). Si alguna vez importa, hay
    # que re-chunkear desde la extraccion, no desde este checkpoint.
    nuevos = _pack_sentences(oraciones, None, 0, count_tokens, presupuesto, solape)

    salida = []
    for chunk in nuevos:
        salida.append(
            {
                **{k: base[k] for k in ("doc_id", "fuente", "formato", "fenomeno", "idioma", "url")},
                "chunk_id": f"{base['doc_id']}-c{chunk.posicion:04d}",
                "posicion": chunk.posicion,
                "num_tokens": chunk.num_tokens,
                "texto": chunk.texto,
                "titulo_seccion": None,
            }
        )
    return salida


def _por_documento(ruta):
    """Agrupa el jsonl por doc_id sin cargarlo entero en memoria."""
    actual, grupo = None, []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            reg = json.loads(linea)
            if reg["doc_id"] != actual:
                if grupo:
                    yield grupo
                actual, grupo = reg["doc_id"], []
            grupo.append(reg)
    if grupo:
        yield grupo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=ENTRADA)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument("--presupuesto", type=int, default=128)
    parser.add_argument("--solape", type=int, default=0)
    parser.add_argument("--encoder-name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--autoprueba", action="store_true")
    args = parser.parse_args()

    if args.autoprueba:
        _autoprueba()
        return

    print(f"cargando tokenizer de {args.encoder_name}...", flush=True)
    count_tokens = get_encoder(args.encoder_name).count_tokens

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    docs = entrada_chunks = salida_chunks = 0
    with open(args.salida, "w", encoding="utf-8", newline="\n") as out:
        for grupo in _por_documento(args.entrada):
            docs += 1
            entrada_chunks += len(grupo)
            for reg in subdividir_documento(grupo, count_tokens, args.presupuesto, args.solape):
                out.write(json.dumps(reg, ensure_ascii=False) + "\n")
                salida_chunks += 1
            if docs % 100 == 0:
                print(f"  {docs} documentos, {salida_chunks} chunks", flush=True)

    print(f"\n{docs} documentos: {entrada_chunks} chunks -> {salida_chunks}")
    print(f"escrito en {args.salida}")


def _autoprueba():
    """Chequeo de las tres cosas que pueden salir mal, sin tocar el corpus."""
    contar = lambda t: len(t.split())  # noqa: E731

    # 1. El solape del chunker viejo se quita: la oracion repetida no se duplica.
    regs = [
        {"doc_id": "D1", "formato": "pdf", "fuente": "f", "fenomeno": 1, "idioma": "es",
         "url": None, "texto": "Uno uno uno. Dos dos dos. Tres tres tres."},
        {"doc_id": "D1", "formato": "pdf", "fuente": "f", "fenomeno": 1, "idioma": "es",
         "url": None, "texto": "Tres tres tres. Cuatro cuatro cuatro."},
    ]
    nuevos = subdividir_documento(regs, contar, presupuesto=4, solape=0)
    texto = " ".join(r["texto"] for r in nuevos)
    assert texto.count("Tres tres tres") == 1, texto

    # 2. Nada supera el presupuesto, salvo una oracion que sola ya lo excede.
    for r in nuevos:
        assert r["num_tokens"] <= 4 or len(split_sentences(r["texto"], "es")) == 1, r

    # 3. Ninguna oracion aparece en dos chunks consecutivos (solape 0).
    for a, b in zip(nuevos, nuevos[1:]):
        assert not (set(split_sentences(a["texto"], "es")) & set(split_sentences(b["texto"], "es")))

    # 4. Los formatos atomicos pasan intactos.
    filas = [{"doc_id": "D2", "formato": "csv", "fuente": "f", "fenomeno": 1,
              "idioma": "en", "url": None, "texto": "a,b,c", "chunk_id": "D2-c0000",
              "posicion": 0, "num_tokens": 3, "titulo_seccion": None}]
    assert subdividir_documento(filas, contar, 4, 0) == filas

    print("autoprueba OK")


if __name__ == "__main__":
    main()
