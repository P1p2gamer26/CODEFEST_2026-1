"""Prior de recencia para consultas con marcador temporal (sec. 8.7).

La sec. 8.7 autoriza post-filtros por metadata incluyendo **rango de fechas**.
El corpus lo permite sin indexar nada nuevo: `derive_fuente()` guarda en cada
chunk el nombre de archivo original ("Nombre estandarizado" del inventario de
ADL), y 509 de los 1826 documentos (28%) llevan el anio ahi.

Alcance real, medido antes de escribir esto
-------------------------------------------
Solo **6 de las 50 consultas** tienen marcador temporal, y de esas apenas dos
devuelven material claramente mas viejo que el relevante (q029: los relevantes
son 2021-2026 y se entregan 2019, 2019, 2025). **El techo son 2 consultas.** Se
implementa porque es barato y esta explicitamente autorizado, no porque sea una
palanca grande -- conviene tenerlo escrito para no volver a estimarlo de cero.

APAGADO POR DEFECTO. No se pudo validar contra el indice en la sesion en que se
escribio, y la leccion 11 del proyecto es que un componente de recuperacion sin
medir no se enciende. Encenderlo con `--prior-recencia` y decidir con
`eval_mini.py --comparar-con`.
"""

from __future__ import annotations

import re

# Marcadores de que la consulta pide material reciente. Deliberadamente
# conservador: activarlo de mas castiga documentos viejos que son la respuesta
# correcta (un tratado de 1967 sigue siendo el tratado). Cubre es/pt/en porque
# aunque las 50 oficiales estan en espanol, el script tiene que aguantar lo que
# le tiren.
_MARCADOR_TEMPORAL = re.compile(
    r"\brecient|\bactual|\búltim|\bultim|\bhoy\b|\bnuevo|\bemergent|\bcoyuntur"
    r"|\bvigente|\bahora\b|\bmoderno|\brecent\b|\bcurrent\b|\blatest\b"
    r"|\bde los ultimos\b|\bde los últimos\b|\b20[12]\d\b",
    re.IGNORECASE,
)

# Anios plausibles de publicacion. El limite inferior descarta numeros que
# parecen anio pero no lo son (codigos, paginas); el superior evita que una
# cifra futura invente un documento novisimo.
_ANIO_MIN, _ANIO_MAX = 1990, 2026
_ANIO = re.compile(r"(?:19|20)\d{2}")


def tiene_marcador_temporal(consulta: str) -> bool:
    """True si la consulta pide explicitamente material reciente."""
    return bool(_MARCADOR_TEMPORAL.search(consulta or ""))


def anio_de_fuente(fuente: str) -> int | None:
    """Anio de publicacion inferido del nombre de archivo, o None.

    Se toma el MAYOR de los anios presentes. Los nombres del corpus meten a
    veces dos ("ai-index-2024-ch1", "informe-2019-actualizado-2021") y el mas
    reciente es el que fecha el documento.
    """
    candidatos = [
        int(a) for a in _ANIO.findall(fuente or "") if _ANIO_MIN <= int(a) <= _ANIO_MAX
    ]
    return max(candidatos) if candidatos else None


def anios_por_documento(hits) -> dict[str, int]:
    """Mapa doc_id -> anio, a partir del campo `fuente` que ya trae cada Hit.

    Si dos chunks del mismo documento dieran anios distintos (no deberia: el
    `fuente` es el mismo archivo) gana el mayor, por coherencia con
    `anio_de_fuente`.
    """
    anios: dict[str, int] = {}
    for hit in hits:
        anio = anio_de_fuente(getattr(hit, "fuente", ""))
        if anio is not None and anio > anios.get(hit.doc_id, 0):
            anios[hit.doc_id] = anio
    return anios


def aplicar_prior_recencia(
    doc_hits: list,
    anios: dict[str, int],
    peso: float = 0.05,
    anio_referencia: int = _ANIO_MAX,
) -> list:
    """Reordena candidatos a documento favoreciendo los mas recientes.

    Es un **reordenamiento, no un filtro**: un documento sin anio conocido (el
    72% del corpus) conserva su puntaje tal cual en vez de irse al fondo. Con
    28% de cobertura, filtrar por fecha tiraria material bueno por no tener el
    anio en el nombre.

    El bonus es `peso * (1 - (referencia - anio) / 36)`, acotado a [0, peso]:
    lineal, saturado a los 36 anios (1990-2026) y siempre positivo, de modo que
    un documento fechado nunca queda por debajo de uno sin fecha con el mismo
    puntaje. `peso` es deliberadamente chico: la senal semantica manda y esto
    solo desempata.

    Devuelve una lista nueva con los `rank` renumerados. No muta la entrada.
    """
    if not doc_hits or peso <= 0:
        return list(doc_hits)

    rango = max(1, anio_referencia - _ANIO_MIN)

    def puntaje(dh):
        anio = anios.get(dh.doc_id)
        if anio is None:
            return dh.score
        cercania = max(0.0, min(1.0, 1.0 - (anio_referencia - anio) / rango))
        return dh.score + peso * cercania

    ordenados = sorted(doc_hits, key=puntaje, reverse=True)
    return [
        type(dh)(rank=i, doc_id=dh.doc_id, score=dh.score)
        for i, dh in enumerate(ordenados, start=1)
    ]
