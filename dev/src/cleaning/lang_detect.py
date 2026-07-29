"""Deteccion del idioma predominante del documento (sec. 2.2), campo extra
`idioma` en la metadata. No es un campo obligatorio de la Tabla 1, pero
habilita el post-filtro por idioma exigido en la seccion de recuperacion."""

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deteccion determinista


def detect_language(texto: str) -> str | None:
    texto = texto.strip()
    if len(texto) < 20:
        return None
    try:
        return detect(texto)
    except LangDetectException:
        return None
