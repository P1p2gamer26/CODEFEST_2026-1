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
    return [
        Entity(text=ent.text.strip(), label=ent.label_, start_char=ent.start_char, end_char=ent.end_char)
        for ent in doc.ents
        if ent.text.strip() and ent.label_ not in EXCLUDED_LABELS
    ]
