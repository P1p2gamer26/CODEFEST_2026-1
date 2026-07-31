"""Extraccion de texto desde CSV/XLSX (sec. 2.1 de la especificacion).

Se lee la fila de cabecera y luego cada registro se convierte en una
secuencia de pares "columna: valor" separados por " | ", de modo que cada
valor conserve el nombre de su columna como contexto. Las celdas vacias se
omiten. Cada fila queda como un bloque de texto independiente (separado por
"\n\n" en `texto_crudo`), ya que el propio reglamento permite tratar cada
fila como una unidad de fragmentacion independiente -- src/chunking la trata
como un chunk atomico (ver `src/chunking/chunker.py`).
"""

import logging
from pathlib import Path

import pandas as pd

from ..config import MAX_FILAS_TABULARES
from .base import RawDocument

logger = logging.getLogger(__name__)


def _row_to_text(row: pd.Series) -> str:
    pares = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
    return " | ".join(pares)


def _dataframe_to_raw_document(df: pd.DataFrame, path: Path, formato: str) -> RawDocument:
    truncado = len(df) > MAX_FILAS_TABULARES
    if truncado:
        logger.warning(
            "%s excede %d filas; se indexan solo las primeras (ver config.MAX_FILAS_TABULARES)",
            path.name, MAX_FILAS_TABULARES,
        )
        df = df.head(MAX_FILAS_TABULARES)

    bloques = [_row_to_text(row) for _, row in df.iterrows()]
    bloques = [b for b in bloques if b.strip()]
    texto_crudo = "\n\n".join(bloques)
    return RawDocument(
        source_path=path,
        formato=formato,
        texto_crudo=texto_crudo,
        extra_metadata={
            "n_filas": len(df),
            "truncado": truncado,
            "columnas": list(df.columns),
        },
    )


def extract_csv(path: Path) -> RawDocument:
    # +1 fila para poder detectar el truncamiento sin cargar el archivo entero
    # (los CSV de PubMed del AI Index pesan hasta 35 MB).
    # `on_bad_lines="skip"`: algun CSV del corpus trae filas con mas campos que
    # la cabecera (p. ej. AIINDEX_lit-covid-ai-covid-literature-csv.csv), y sin
    # esto pandas aborta y se pierde el documento ENTERO en vez de unas filas.
    # `sep=None` + engine python: algunos ".csv" del corpus son en realidad
    # TSV; sin sniffear el delimitador quedarian como una sola columna con el
    # nombre de las tres columnas repetido en cada fila.
    df = pd.read_csv(
        path,
        nrows=MAX_FILAS_TABULARES + 1,
        sep=None,
        engine="python",
        on_bad_lines="skip",
    )
    return _dataframe_to_raw_document(df, path, "csv")


def extract_xlsx(path: Path) -> RawDocument:
    df = pd.read_excel(path)
    return _dataframe_to_raw_document(df, path, "xlsx")
