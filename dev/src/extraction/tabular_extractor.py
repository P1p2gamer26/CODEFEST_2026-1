"""Extraccion de texto desde CSV/XLSX (sec. 2.1 de la especificacion).

Se lee la fila de cabecera y luego cada registro se convierte en una
secuencia de pares "columna: valor" separados por " | ", de modo que cada
valor conserve el nombre de su columna como contexto. Las celdas vacias se
omiten. Cada fila queda como un bloque de texto independiente (separado por
"\n\n" en `texto_crudo`), ya que el propio reglamento permite tratar cada
fila como una unidad de fragmentacion independiente -- src/chunking la trata
como un chunk atomico (ver `src/chunking/chunker.py`).
"""

from pathlib import Path

import pandas as pd

from .base import RawDocument


def _row_to_text(row: pd.Series) -> str:
    pares = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
    return " | ".join(pares)


def _dataframe_to_raw_document(df: pd.DataFrame, path: Path, formato: str) -> RawDocument:
    bloques = [_row_to_text(row) for _, row in df.iterrows()]
    bloques = [b for b in bloques if b.strip()]
    texto_crudo = "\n\n".join(bloques)
    return RawDocument(
        source_path=path,
        formato=formato,
        texto_crudo=texto_crudo,
        extra_metadata={"n_filas": len(df), "columnas": list(df.columns)},
    )


def extract_csv(path: Path) -> RawDocument:
    df = pd.read_csv(path)
    return _dataframe_to_raw_document(df, path, "csv")


def extract_xlsx(path: Path) -> RawDocument:
    df = pd.read_excel(path)
    return _dataframe_to_raw_document(df, path, "xlsx")
