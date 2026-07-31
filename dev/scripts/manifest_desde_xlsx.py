#!/usr/bin/env python3
"""Convierte el `Indice_Datos_Codefest.xlsx` de ADL en el manifest de doc_id
que consume el pipeline (`--doc-id-manifest`).

La hoja `Inventario de Archivos` trae una fila por documento con el DOC_ID
oficial (columna calculada, pero Excel dejo los valores cacheados). Se emite
un CSV con `ruta` (clave primaria: `Carpeta` + `/` + `Nombre estandarizado`),
`nombre` y `doc_id`. La ruta es imprescindible porque 59 nombres de archivo
aparecen dos veces con DOC_ID distintos -- ver `src/ingestion/doc_id.py`.

Uso:
    python scripts/manifest_desde_xlsx.py
    python scripts/manifest_desde_xlsx.py --xlsx otra/ruta.xlsx --out otro.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openpyxl  # noqa: E402

from src.config import DOC_ID_MANIFEST_PATH, INDICE_DATOS_XLSX_PATH  # noqa: E402

HOJA = "Inventario de Archivos"
COL_DOC_ID = "DOC_ID"
COL_NOMBRE = "Nombre estandarizado"
COL_CARPETA = "Carpeta"


def leer_inventario(xlsx_path: Path) -> list[dict[str, str]]:
    libro = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    hoja = libro[HOJA]
    filas = hoja.iter_rows(values_only=True)
    cabecera = [str(c).strip() if c is not None else "" for c in next(filas)]
    idx = {nombre: cabecera.index(nombre) for nombre in (COL_DOC_ID, COL_NOMBRE, COL_CARPETA)}

    registros = []
    for fila in filas:
        nombre = fila[idx[COL_NOMBRE]]
        doc_id = fila[idx[COL_DOC_ID]]
        if not nombre or not doc_id:
            continue
        carpeta = (fila[idx[COL_CARPETA]] or "").strip().strip("/")
        ruta = f"{carpeta}/{nombre}" if carpeta else str(nombre)
        registros.append({"ruta": ruta, "nombre": str(nombre), "doc_id": str(doc_id)})
    return registros


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", type=Path, default=INDICE_DATOS_XLSX_PATH)
    parser.add_argument("--out", type=Path, default=DOC_ID_MANIFEST_PATH)
    args = parser.parse_args()

    registros = leer_inventario(args.xlsx)
    if not registros:
        print(f"ERROR: la hoja '{HOJA}' de {args.xlsx} no produjo ninguna fila", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["ruta", "nombre", "doc_id"])
        escritor.writeheader()
        escritor.writerows(registros)

    doc_ids = {r["doc_id"] for r in registros}
    nombres = {r["nombre"] for r in registros}
    print(f"{len(registros)} filas -> {args.out}")
    print(f"  doc_id unicos: {len(doc_ids)}")
    print(f"  nombres de archivo unicos: {len(nombres)} "
          f"({len(registros) - len(nombres)} colisiones, resolubles solo por ruta)")


if __name__ == "__main__":
    main()
