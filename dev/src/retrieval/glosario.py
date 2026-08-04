"""Glosario bilingue ES->EN para expandir la consulta antes de vectorizarla.

Por que existe
--------------
Las 50 consultas oficiales estan **todas en espanol**, y el corpus de los
fenomenos 1 y 2 (IA y espacio: CSET, SWF, SIPRI, CSIS) esta **casi todo en
ingles**. Los encoders son multilingues y en general tienden el puente, pero
hay terminos de dominio donde el puente no existe en el dato. Contado sobre
los 128.526 chunks del indice entregado:

    termino de la consulta (ES)        chunks    forma inglesa       chunks
    NBQR                                    0    CBRN                   66
    maniobras de proximidad                 0    rendezvous and prox.  382
    reabastecimiento en orbita              0    on-orbit servicing    113
    enjambres de drones                     0    drone swarm            81
    operaciones espaciales dinamicas        0    dynamic space ops.      4
    desechos orbitales                      1    space debris        1.947
    semiconductores                         4    semiconductor         408
    antisatelite                            8    ASAT                3.813
    energia dirigida                       16    directed energy       522
    guerra electronica                     29    electronic warfare  1.226
    contraespacial                         60    counterspace        3.595

**Todos** los terminos de dominio que las consultas usan en espanol son entre
30 y 2.000 veces mas raros que su forma inglesa, y cinco estan literalmente
ausentes. Eso es una propiedad del corpus, medible sin ground truth y sin
mirar que documentos son relevantes.

Ojo con una nota vieja del proyecto: `las notas del proyecto` daba por descartada la
hipotesis "el recuperador ignora los terminos discriminantes" porque las
SIGLAS de las consultas si estan en el corpus (GDO, GAOR, GAO, RPO, GEO).
Eso sigue siendo cierto y no lo contradice: lo que faltaba revisar no eran
las siglas sino los **terminos multipalabra en espanol**, que es donde esta
la asimetria.

Por que es legal (sec. 8.3)
---------------------------
Es una tabla escrita a mano. No hay modelo generativo, ni traductor
automatico, ni clasificador: la consulta se concatena con las formas inglesas
de los terminos que contiene y se vectoriza una sola vez. La sec. 8.7 permite
ademas trabajar la consulta antes de la busqueda.

Por que no es sobreajuste
-------------------------
Los pares salen de contar formas sobre el CORPUS, no de mirar que documentos
marca el ground truth ni que consultas fallan. El criterio de entrada a la
tabla se fijo antes: el termino espanol aparece en la consulta, y su forma
inglesa es al menos un orden de magnitud mas frecuente en el corpus.

Efecto neutro por construccion
------------------------------
Una consulta que no contiene ningun termino de la tabla se devuelve
**identica**, asi que su vector -- y por tanto su respuesta -- no cambia.
"""

from __future__ import annotations

import unicodedata

# (termino en espanol tal como puede aparecer en la consulta, formas inglesas
# que el corpus si usa). La clave va sin acentos y en minusculas porque asi se
# normaliza la consulta antes de buscarla; el valor se concatena tal cual.
GLOSARIO: tuple[tuple[str, str], ...] = (
    ("nbqr", "CBRN chemical biological radiological nuclear"),
    ("contraespacial", "counterspace"),
    ("contraespaciales", "counterspace"),
    ("guerra electronica", "electronic warfare jamming"),
    # "maniobras de proximidad" -> "rendezvous and proximity operations" SE
    # QUITO. Cumplia la mitad del criterio (0 chunks en espanol contra 382 en
    # ingles) pero no la otra: la unica consulta que lo usa (q021) ya trae la
    # sigla **RPO** escrita, y RPO aparece en 871 chunks del corpus. O sea que
    # esa consulta ya hablaba el idioma del corpus y la expansion solo diluia:
    # medido, le costaba F1 0.33 -> 0.00. El criterio de entrada a la tabla es
    # que la CONSULTA no tenga ningun puente, no solo que el termino espanol
    # sea raro.
    ("energia dirigida", "directed energy weapons"),
    ("antisatelite", "anti-satellite ASAT"),
    ("desechos orbitales", "orbital debris space debris"),
    ("reabastecimiento en orbita", "on-orbit servicing refueling"),
    ("servicio y reabastecimiento", "on-orbit servicing"),
    ("operaciones espaciales dinamicas", "dynamic space operations"),
    ("enjambres de drones", "drone swarms"),
    ("semiconductores", "semiconductors"),
)


def _normalizar(texto: str) -> str:
    """Minusculas y sin acentos, para que la tabla no dependa de la tilde."""
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def terminos_expandidos(consulta: str) -> list[str]:
    """Formas inglesas que le corresponden a esta consulta, sin repetir.

    Se conserva el orden de la tabla para que la expansion sea determinista:
    dos corridas con la misma consulta tienen que producir el mismo texto y
    por tanto el mismo vector.
    """
    normalizada = _normalizar(consulta or "")
    vistos: set[str] = set()
    fuera: list[str] = []
    for termino, expansion in GLOSARIO:
        if termino in normalizada and expansion not in vistos:
            vistos.add(expansion)
            fuera.append(expansion)
    return fuera


def expandir_consulta(consulta: str) -> str:
    """La consulta con sus terminos ingleses anexados, o tal cual si no aplica.

    Se ANEXA en vez de reemplazar: la consulta en espanol sigue siendo la
    mayor parte del texto y los documentos en espanol (todo el fenomeno 3) no
    tienen por que perder. Es lo contrario del rescate lexico con BM25, donde
    el termino REEMPLAZA la consulta -- ahi hacia falta porque 20 palabras
    comunes acumulaban mas score que un termino raro; aca no, porque el
    encoder pondera semanticamente y no por conteo.
    """
    extras = terminos_expandidos(consulta)
    return f"{consulta} {' '.join(extras)}" if extras else consulta
