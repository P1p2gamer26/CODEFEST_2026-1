"""Segmentacion de oraciones multilingue (ES/EN/PT), base de la garantia de
completitud linguistica exigida en la seccion 3.3 de la especificacion:
ningun chunk puede cortar una oracion, porque la unidad minima de
empaquetado del chunker (chunker.py) es siempre la oracion completa.

Estrategia: `pysbd` (reglas linguisticas dedicadas) para es/en, que es donde
tiene soporte nativo; `spaCy` (parser entrenado de pt_core_news_sm) para
portugues, que pysbd no cubre; y un sentencizer generico de spaCy (reglas,
sin modelo entrenado) como ultimo recurso para cualquier otro idioma o
cuando no se pudo detectar el idioma del documento.
"""

from functools import lru_cache

import pysbd
import spacy

from ..config import SPACY_MODEL_BY_LANG

PYSBD_SUPPORTED_LANGS = {"en", "es"}


@lru_cache(maxsize=None)
def _get_pysbd_segmenter(lang: str) -> pysbd.Segmenter:
    return pysbd.Segmenter(language=lang, clean=False)


@lru_cache(maxsize=None)
def _get_spacy_trained_sentencizer(model_name: str):
    return spacy.load(
        model_name,
        disable=["tagger", "ner", "lemmatizer", "attribute_ruler", "morphologizer"],
    )


@lru_cache(maxsize=None)
def _get_spacy_blank_sentencizer(lang: str):
    try:
        nlp = spacy.blank(lang)
    except Exception:
        nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str, lang: str | None) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if lang in PYSBD_SUPPORTED_LANGS:
        sentences = _get_pysbd_segmenter(lang).segment(text)
    elif lang and lang in SPACY_MODEL_BY_LANG:
        nlp = _get_spacy_trained_sentencizer(SPACY_MODEL_BY_LANG[lang])
        sentences = [s.text for s in nlp(text).sents]
    else:
        nlp = _get_spacy_blank_sentencizer(lang or "xx")
        sentences = [s.text for s in nlp(text).sents]

    return [s.strip() for s in sentences if s and s.strip()]
