"""Pruebas del prior de recencia (sec. 8.7).

Los nombres de archivo son reales, del inventario de ADL
(`dev/corpus_meta/Indice_Datos_Codefest.xlsx`, hoja "Inventario de Archivos").
"""

from dataclasses import dataclass

from src.retrieval.recencia import (
    anio_de_fuente,
    anios_por_documento,
    aplicar_prior_recencia,
    tiene_marcador_temporal,
)


@dataclass
class _Doc:
    rank: int
    doc_id: str
    score: float


@dataclass
class _Hit:
    doc_id: str
    fuente: str


def test_extrae_el_anio_de_nombres_reales_del_inventario():
    assert anio_de_fuente("AIINDEX_ai-index-2024-ch1-research-development.pdf") == 2024
    assert anio_de_fuente("SWF_global-counterspace-capabilities-2022.pdf") == 2022


def test_sin_anio_devuelve_none():
    assert anio_de_fuente("CSET_ai-and-cybersecurity.pdf") is None
    assert anio_de_fuente("") is None


def test_toma_el_anio_mayor_cuando_hay_varios():
    """Los nombres del corpus a veces traen dos: el mas reciente fecha el
    documento."""
    assert anio_de_fuente("informe-2019-actualizado-2021.pdf") == 2021


def test_ignora_numeros_que_no_son_anio_plausible():
    """Un codigo de cuatro cifras fuera del rango no puede fechar nada."""
    assert anio_de_fuente("doc-1503-anexo.pdf") is None
    assert anio_de_fuente("resolucion-2050-borrador.pdf") is None


def test_detecta_marcador_temporal_en_las_consultas_oficiales():
    """Enunciados reales de consultas_50_oficiales.jsonl."""
    assert tiene_marcador_temporal(
        "¿Qué capacidades operacionales evidencian las maniobras realizadas "
        "recientemente por satélites rusos en órbita GEO?"
    )
    assert tiene_marcador_temporal(
        "¿Qué innovaciones tácticas recientes han incorporado los grupos armados?"
    )


def test_no_marca_consultas_atemporales():
    """Activarlo de mas castiga documentos viejos que son la respuesta correcta:
    un tratado de 1967 sigue siendo el tratado."""
    assert not tiene_marcador_temporal(
        "¿Qué restricciones impone el Derecho Internacional en el Espacio en la "
        "regulación del uso de armas?"
    )


def test_anios_por_documento_sale_del_campo_fuente():
    hits = [
        _Hit("D1", "SWF_report-2022.pdf"),
        _Hit("D1", "SWF_report-2022.pdf"),
        _Hit("D2", "CSET_sin-anio.pdf"),
    ]
    assert anios_por_documento(hits) == {"D1": 2022}


def test_el_prior_sube_al_mas_reciente_con_puntajes_empatados():
    docs = [_Doc(1, "VIEJO", 0.80), _Doc(2, "NUEVO", 0.80)]
    salida = aplicar_prior_recencia(docs, {"VIEJO": 1995, "NUEVO": 2026}, peso=0.05)
    assert [d.doc_id for d in salida] == ["NUEVO", "VIEJO"]
    assert [d.rank for d in salida] == [1, 2]


def test_el_prior_no_da_vuelta_una_diferencia_semantica_clara():
    """El peso es chico a proposito: desempata, no manda. Si esto se rompe, el
    prior esta pisando la senal del encoder."""
    docs = [_Doc(1, "VIEJO_PERO_BUENO", 0.90), _Doc(2, "NUEVO", 0.80)]
    salida = aplicar_prior_recencia(docs, {"VIEJO_PERO_BUENO": 1995, "NUEVO": 2026})
    assert [d.doc_id for d in salida] == ["VIEJO_PERO_BUENO", "NUEVO"]


def test_un_documento_sin_anio_no_se_hunde():
    """Reordena, no filtra: con solo 28% del corpus fechado, castigar a los sin
    fecha tiraria material bueno."""
    docs = [_Doc(1, "SIN_FECHA", 0.90), _Doc(2, "FECHADO", 0.80)]
    salida = aplicar_prior_recencia(docs, {"FECHADO": 2026}, peso=0.05)
    assert [d.doc_id for d in salida] == ["SIN_FECHA", "FECHADO"]


def test_no_muta_la_entrada_ni_cambia_los_scores():
    docs = [_Doc(1, "A", 0.80), _Doc(2, "B", 0.80)]
    salida = aplicar_prior_recencia(docs, {"B": 2026}, peso=0.05)
    assert [d.doc_id for d in docs] == ["A", "B"], "la entrada no debe mutarse"
    assert {d.doc_id: d.score for d in salida} == {"A": 0.80, "B": 0.80}


def test_peso_cero_es_identidad():
    docs = [_Doc(1, "A", 0.80), _Doc(2, "B", 0.80)]
    salida = aplicar_prior_recencia(docs, {"B": 2026}, peso=0.0)
    assert [d.doc_id for d in salida] == ["A", "B"]
