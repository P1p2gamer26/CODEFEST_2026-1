"""Extraccion de relaciones (RE), sec. 7.2 paso 2 -- via heuristicas basadas
en dependencias sintacticas de spaCy (estadistico, NO un modelo generativo),
tal como permite explicitamente la especificacion ("heuristicas basadas en
patrones linguisticos, o dependencias sintacticas").

Heuristica: para cada oracion con al menos dos entidades, se emparejan
entidades consecutivas (en orden de aparicion) y se busca un verbo/auxiliar
entre ambas para usar su lema como etiqueta de relacion; si no se encuentra
ninguno, se usa el verbo raiz de la oracion; si tampoco hay, se cae en la
relacion generica `relacionado_con` por co-ocurrencia (sec. 7.2, paso 2).
Solo se emparejan entidades consecutivas (no todas las combinaciones) para
evitar relaciones espurias entre entidades lejanas dentro de una misma
oracion larga.
"""

from .ner import DEFAULT_LANG, EXCLUDED_LABELS, _get_ner_pipeline

FALLBACK_RELATION = "relacionado_con"


def _connecting_verb_lemma(sent, ent1_end: int, ent2_start: int) -> str | None:
    between = sent.doc[ent1_end:ent2_start]
    for token in between:
        if token.pos_ in ("VERB", "AUX"):
            return token.lemma_.lower()
    if sent.root.pos_ in ("VERB", "AUX"):
        return sent.root.lemma_.lower()
    return None


def extract_triples(text: str, lang: str | None) -> list[tuple[str, str, str, str, str]]:
    """Devuelve tripletas (sujeto, relacion, objeto) encontradas en `text`.

    Cada tripleta incluye ademas la etiqueta NER de cada entidad (PER/ORG/LOC/
    MISC, u OntoNotes en el modelo de ingles) para poder tipar y colorear los
    nodos del grafo al visualizarlo.
    """
    if not text or not text.strip():
        return []

    nlp = _get_ner_pipeline(lang or DEFAULT_LANG)
    doc = nlp(text)

    triples: list[tuple[str, str, str, str, str]] = []
    for sent in doc.sents:
        ents = [e for e in sent.ents if e.text.strip() and e.label_ not in EXCLUDED_LABELS]
        for e1, e2 in zip(ents, ents[1:]):
            subj, obj = e1.text.strip(), e2.text.strip()
            if subj.lower() == obj.lower():
                continue
            relation = _connecting_verb_lemma(sent, e1.end, e2.start) or FALLBACK_RELATION
            triples.append((subj, relation, obj, e1.label_, e2.label_))

    return triples
