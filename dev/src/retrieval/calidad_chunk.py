"""Estima que fraccion de un fragmento es aparato bibliografico y no prosa.

Por que existe
--------------
La sec. 10.2.1 del PDF dice que la relevancia de cada fragmento se juzga sobre
su **contenido textual** (campo `text`) y que el `chunk_id` NO es la clave de
emparejamiento. O sea: el evaluador lee el texto que le entregamos. Un chunk
que es una lista de notas al pie de un documento relevante puntua 0 para el
evaluador humano y 1 para nuestro NDCG proxy, que hereda la relevancia del
documento. El proxy miente exactamente en la direccion que importa.

Medido sobre los 500 fragmentos de la entrega actual: hay chunks que arrancan
con la bibliografia entera ("[1047] "What Are the Risks from Artificial
Intelligence?," MIT AI Risk Initiative, accessed September 2025, https://...")
ocupando cupos del top-10.

Por que por segmento y no por chunk
-----------------------------------
El caso frecuente no es el chunk 100% bibliografia sino el **mixto**: prosa
util que termina en una cola de notas al pie, porque el chunker corta por
ventana de tokens y no respeta la frontera cuerpo/notas. Puntuar el chunk
entero con un umbral global los clasifica mal en los dos sentidos. Medido: de
16 fragmentos con >=2 URL/100 palabras, 4 empiezan con prosa legitima.

Por eso la funcion devuelve una **fraccion continua** (que parte del texto es
aparato) y no una etiqueta. Sirve para reordenar, que es lo unico que se puede
hacer: la sec. 9.3.2 exige exactamente 10 fragmentos, asi que descartar no es
una opcion.

Por que sin modelo
------------------
La sec. 8.3 restringe que puede intervenir en la recuperacion. Esto es
contar patrones tipograficos: no hay embedding, ni clasificador entrenado, ni
nada que se parezca a un decoder. Ademas tiene que correr en CPU dentro del
presupuesto del evaluador.

Un contraejemplo que guio el diseno: F1-CSET-098-c0070 tiene 22% de tokens con
digito y es contenido legitimo ("Conduct cybersecurity threat assessments to
identify potential threat actors [359, 360]"). La densidad numerica sola lo
marcaria como bibliografia. Por eso el peso fuerte esta en la URL y en el
marcador de nota al pie a principio de segmento, no en los digitos sueltos.
"""

from __future__ import annotations

import re

# Una URL dentro de un segmento corto es la senal mas fiable de entrada
# bibliografica: la prosa de estos informes cita por numero, no pegando el
# enlace en medio de la frase.
_URL = re.compile(r"https?://|www\.\w|doi\.org/|\bdoi:", re.IGNORECASE)

# Marcador de nota al pie al principio del segmento: "[1047] ", "140 ", "12 ".
# Exige que lo siga una mayuscula o una comilla de apertura, que es como
# empieza un nombre de autor o un titulo entrecomillado.
_MARCADOR = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4})[.,]?\s+[\"“«A-ZÀ-ÞА-Я]")

# Titulo entrecomillado seguido de coma: '"Firefly: Spoofing Earth ...,"'.
# Es la forma canonica de una cita en estos corpus (Chicago/Turabian).
_TITULO_CITADO = re.compile(r"[\"“«][^\"”»]{12,}[\"”»]\s*[,.]")

# Mes en ingles/espanol/portugues seguido de anio: "September 27, 2018".
_MES_ANIO = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December|enero|febrero|marzo|abril|mayo|junio|julio|agosto"
    r"|septiembre|setiembre|octubre|noviembre|diciembre|janeiro|fevereiro"
    r"|mar[cç]o|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

# Vocabulario de aparato: paginacion, edicion, acceso. Multilingue porque el
# corpus lo es.
_VOCAB_APARATO = re.compile(
    r"\bet al\.|\baccessed\b|\bconsultado\b|\bacesso em\b|\bvol\.|\bno\.\s*\d"
    r"|\bpp?\.\s*\d|\bibid\b|\bop\. cit\.|\bed[s]?\.\,|\bretrieved\b"
    r"|\bdisponible en\b|\bavailable at\b",
    re.IGNORECASE,
)

# Corta en fin de oracion, y tambien JUSTO ANTES de un marcador de nota al pie
# aunque no haya punto: el caso "...pero 03-21 144 Richard Fisher, Jr." tiene
# la frontera cuerpo/notas en medio de lo que parece una sola oracion.
#
# La segunda alternativa es la que mas trabajo hace en este corpus. La
# extraccion pega la llamada a la nota al final de la palabra sin espacio
# ("...miniaturization of electronics.222 Despite..."), asi que despues del
# punto no hay blanco y la regla de fin-de-oracion no dispara. Sin esta
# alternativa, el segmento de prosa se traga el bloque de notas que le sigue y
# hereda su URL: el chunk entero queda marcado como bibliografia.
_CORTE = re.compile(
    r"(?<=[.!?;])\s+(?=[\"“«A-ZÀ-ÞА-Я\[\d])"
    r"|(?<=[.!?][0-9])\s+|(?<=[.!?][0-9]{2})\s+|(?<=[.!?][0-9]{3})\s+"
    r"|\s+(?=\[\d{1,4}\]\s*[\"“«A-ZÀ-Þ])"
    r"|\s+(?=\d{1,4}\s+[\"“«])"
    r"|\s+(?=\d{1,4}\s+[A-ZÀ-Þ][a-zà-þ]+[,\s])"
)

# Un segmento mas corto que esto no se juzga por su cuenta: se arrastra al
# anterior. Evita que un "Jr." o un "S." partan una cita en trozos que
# individualmente no disparan ninguna senal.
_MIN_PALABRAS_SEGMENTO = 4


def segmentar(texto: str) -> list[str]:
    """Parte el texto en unidades juzgables (oracion o entrada bibliografica).

    Los chunks vienen sin saltos de linea -- la limpieza los colapsa -- asi que
    la unica frontera disponible es tipografica.
    """
    crudos = [s.strip() for s in _CORTE.split(texto) if s and s.strip()]
    if not crudos:
        return []
    fusionados: list[str] = [crudos[0]]
    for seg in crudos[1:]:
        if len(seg.split()) < _MIN_PALABRAS_SEGMENTO:
            fusionados[-1] = f"{fusionados[-1]} {seg}"
        else:
            fusionados.append(seg)
    return fusionados


def _es_referencia(segmento: str) -> bool:
    """True si el segmento parece aparato bibliografico y no prosa.

    La URL sola alcanza. El resto de senales necesitan dos coincidencias, para
    no marcar prosa que simplemente cita una fecha o usa comillas.
    """
    if _URL.search(segmento):
        return True
    senales = sum(
        bool(patron.search(segmento))
        for patron in (_MARCADOR, _TITULO_CITADO, _MES_ANIO, _VOCAB_APARATO)
    )
    return senales >= 2


def fraccion_aparato(texto: str) -> float:
    """Fraccion de palabras del fragmento que caen en segmentos bibliograficos.

    Devuelve un valor en [0, 1]. 0 = prosa limpia, 1 = solo bibliografia.
    Un texto vacio devuelve 1.0: no aporta nada al evaluador.
    """
    segmentos = segmentar(texto)
    total = sum(len(s.split()) for s in segmentos)
    if total == 0:
        return 1.0
    aparato = sum(len(s.split()) for s in segmentos if _es_referencia(s))
    return aparato / total


def calidad(texto: str) -> float:
    """Complemento de `fraccion_aparato`: 1 = fragmento util, 0 = puro aparato.

    Pensado como criterio de DESEMPATE al ordenar los 10 fragmentos, nunca
    como filtro: la sec. 9.3.2 exige exactamente 10 y quitar uno malo sin tener
    con que reemplazarlo invalida la entrega.
    """
    return 1.0 - fraccion_aparato(texto)
