"""Parchea metadata.jsonl de los tres encoders de la entrega (una sola vez).

Dos correcciones, ambas alineadas con la Q&A de ADL:

1. **`formato` "img" -> extension real.** La Tabla 1/FAQ definen `formato`
   como la extension real del archivo en minusculas. Cinco chunks OCR de
   imagenes SWF (F2-SWF-076 x2, 077, 084, 089) salieron con el valor
   placeholder "img"; el archivo real es `.jpg`.

2. **Filas metadata-only para las 5 imagenes sin texto del manifest.**
   F2-SWF-065 (`.avif`) y 066/067/068/071 (`.jpg`) estan en el inventario de
   ADL con DOC_ID, pero al no tener texto no generaron chunks ni fila de
   metadata. La Q&A dice que sin texto se conserva la metadata SIN inventar
   contenido: se agregan filas con `texto: ""`, `num_tokens: 0` y el marcador
   `en_indice: false` para que el guard de alineacion indice/metadata
   (sec. 5.3) sepa que no tienen vector.

Garantias:

- Las filas metadata-only van AL FINAL: las filas 0..ntotal-1 siguen
  alineando con los vectores FAISS, que es el invariante que consume
  `reconstruct(fila)` en la cascada.
- Los tres metadata.jsonl se escriben byte-identicos (se genera el contenido
  una vez y se copia), asi que LFS los deduplica por OID y
  `verificar_alineacion()` sigue pasando.
- Reanudable: si ya tiene las filas, se aborta sin tocar nada.

Uso: .venv/Scripts/python.exe dev/scripts/parchear_metadata_imagenes.py
"""

import glob
import json
import sys
from pathlib import Path

ENTREGA = Path(__file__).resolve().parents[2] / "Entrega"

# (doc_id, nombre estandarizado en el inventario, extension real)
IMAGENES_SIN_TEXTO = [
    ("F2-SWF-065", "SWF_68239a54783e8c917bedf423-hs-2025-victoriasamson-web.avif", "avif"),
    ("F2-SWF-066", "SWF_68399caedbb7cbe28229c326-38236.jpg", "jpg"),
    ("F2-SWF-067", "SWF_68399cb24dc4a66ec0d2c535-s43-83-082orig.jpg", "jpg"),
    ("F2-SWF-068", "SWF_68399cb30e67dd1473cc427a-sts063-712-072medium.jpg", "jpg"),
    ("F2-SWF-071", "SWF_685948b82c82d26ce150c2a9-kbrett.jpg", "jpg"),
]


def _filas_metadata() -> list[dict]:
    rutas = sorted(glob.glob(str(ENTREGA / "base_vectorial" / "encoder_*" / "metadata.jsonl")))
    if not rutas:
        sys.exit("error: no hay metadata.jsonl en Entrega/base_vectorial/encoder_*/")
    filas = [json.loads(linea) for linea in Path(rutas[0]).read_text(encoding="utf-8").splitlines() if linea.strip()]
    return filas


def main() -> None:
    rutas = sorted(glob.glob(str(ENTREGA / "base_vectorial" / "encoder_*" / "metadata.jsonl")))
    contenidos = {Path(r).read_bytes() for r in rutas}
    if len(contenidos) != 1:
        sys.exit("error: los metadata.jsonl NO eran byte-identicos antes de parchear; revisar a mano")
    filas = _filas_metadata()
    # Algunas filas "img" que aun queden sin tocar: el formato real es jpg.
    n_img = sum(1 for f in filas if f.get("formato") == "img")
    ya_tiene = [f for f in filas if f.get("en_indice") is False]
    if ya_tiene:
        sys.exit(f"reanudando: ya hay {len(ya_tiene)} filas metadata-only; no se toca nada")
    if n_img == 0:
        sys.exit("nada que hacer: no hay filas con formato 'img'")

    # 1. formato img -> jpg (la extension real de los cinco archivos).
    for f in filas:
        if f.get("formato") == "img":
            f["formato"] = "jpg"

    # 2. Filas metadata-only al final, en orden de doc_id.
    for doc_id, fuente, formato in IMAGENES_SIN_TEXTO:
        filas.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}-c0000",
                "fuente": fuente,
                "formato": formato,
                "fenomeno": 2,
                "posicion": 0,
                "num_tokens": 0,
                "texto": "",
                "idioma": None,
                "titulo_seccion": None,
                "url": None,
                "en_indice": False,
            }
        )

    # Verificacion antes de escribir.
    ids = [f["chunk_id"] for f in filas]
    if len(set(ids)) != len(ids):
        sys.exit("error: chunk_id duplicados tras el parche")
    contenido = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas)

    # Los tres encoders comparten chunking y por tanto metadata byte-identica:
    # se escribe el MISMO contenido en los tres (o LFS no deduplicaria).
    rutas = sorted(glob.glob(str(ENTREGA / "base_vectorial" / "encoder_*" / "metadata.jsonl")))
    for ruta in rutas:
        p = Path(ruta)
        p.write_text(contenido, encoding="utf-8", newline="\n")
        print(f"  escrito {p} ({len(filas)} filas)")

    # Confirmar byte-identidad.
    hashes = {Path(r).read_bytes() for r in rutas}
    if len(hashes) != 1:
        sys.exit("error: los metadata.jsonl dejaron de ser byte-identicos")
    print(f"ok: {n_img} filas img -> jpg; {len(IMAGENES_SIN_TEXTO)} filas metadata-only agregadas")


if __name__ == "__main__":
    main()
