"""Reconocimiento de entidades nombradas (NER), sec. 7.2 paso 1.

Se usa el componente NER ya entrenado en los modelos spaCy
`es/en/pt_core_news_sm` (arquitectura no generativa, licencia MIT) en vez de
un modelo de NER de HuggingFace: `huggingface.co` esta bloqueado en el
sandbox usado para construir este pipeline (ver informe_tecnico.pdf,
seccion de limitaciones), mientras que los modelos spaCy se instalan como
paquetes de pip sin tocar ese dominio. Es una alternativa equivalente en
espiritu al modelo originalmente recomendado
(`Babelscape/wikineural-multilingual-ner`): tambien es un encoder NO
generativo entrenado para NER, aqui como tres modelos monolingues en vez de
uno unico multilingue.
"""

from dataclasses import dataclass
from functools import lru_cache

import spacy

from ..config import SPACY_MODEL_BY_LANG

DEFAULT_LANG = "es"

# Los modelos es/pt_core_news_sm usan el esquema CoNLL (LOC, MISC, ORG, PER);
# en_core_web_sm usa el esquema OntoNotes, mucho mas granular (18 tipos).
# Para que el grafo capture entidades relevantes (personas, organizaciones,
# lugares, tecnologias, eventos -- sec. 7.1) y no ruido numerico/temporal,
# se excluyen explicitamente las etiquetas de OntoNotes que no describen una
# entidad nombrada propiamente dicha (CARDINAL, DATE, MONEY, ORDINAL,
# PERCENT, QUANTITY, TIME).
EXCLUDED_LABELS = {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}

# La etiqueta MISC del esquema CoNLL (es/pt) ocasionalmente captura clausulas
# completas en vez de una entidad nombrada real. En vez de descartar la etiqueta
# entera se acota la longitud de las entidades admitidas: 6 palabras / 60
# caracteres sobran de sobra para una entidad legitima y recortan las clausulas.
MAX_ENTIDAD_PALABRAS = 6
MAX_ENTIDAD_CARACTERES = 60


def _limpiar_entidad(texto: str) -> str | None:
    """Normaliza el span del NER y devuelve None si no queda una entidad limpia.

    Cuando el span de spaCy no coincide exactamente con un parentesis del texto
    original (puntuacion irregular de OCR/PDF), queda un caracter colgando en el
    nombre del nodo (ej. "Instituto Kroc) Objetivo"). Se recortan SOLO los
    parentesis colgantes de los bordes (los que estan desbalanceados; el ')' de
    "Cooperacion (ONU)" cierra un parentesis interno y no se toca), se eliminan
    los parentesis redundantes que envuelven todo el span y se descarta la
    entidad si al final queda un parentesis desbalanceado. Ademas se aplica el
    tope de longitud: asi una clausula completa etiquetada como MISC no llega a
    convertirse en nodo.
    """
    t = texto.strip()
    while t.endswith(")") and t.count("(") < t.count(")"):
        t = t[:-1].rstrip()
    while t.startswith("(") and t.count("(") > t.count(")"):
        t = t[1:].lstrip()
    t = t.strip()
    while t.startswith("(") and t.endswith(")") and t.count("(") == t.count(")"):
        t = t[1:-1].strip()
    if not t:
        return None
    if t.count("(") != t.count(")"):
        return None
    if len(t) > MAX_ENTIDAD_CARACTERES or len(t.split()) > MAX_ENTIDAD_PALABRAS:
        return None
    return t


@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int


@lru_cache(maxsize=None)
def _get_ner_pipeline(lang: str):
    model_name = SPACY_MODEL_BY_LANG.get(lang, SPACY_MODEL_BY_LANG[DEFAULT_LANG])
    return spacy.load(model_name)


def extract_entities(text: str, lang: str | None) -> list[Entity]:
    if not text or not text.strip():
        return []
    nlp = _get_ner_pipeline(lang or DEFAULT_LANG)
    doc = nlp(text)
    entidades: list[Entity] = []
    for ent in doc.ents:
        if not ent.text.strip() or ent.label_ in EXCLUDED_LABELS:
            continue
        entidad = _limpiar_entidad(ent.text)
        if entidad is None:
            continue
        entidades.append(Entity(text=entidad, label=ent.label_, start_char=ent.start_char, end_char=ent.end_char))
    return entidades
