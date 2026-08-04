"""Pruebas del diagnostico que separa "tema equivocado" de "hermano equivocado".

De esta clasificacion sale una decision de prioridad: el bucket grande (22% de
las consultas) es hermano equivocado, que ataca el grafo, y no tema equivocado,
que es lo que ataca el glosario bilingue. Si `coleccion()` empieza a agrupar
mal, los dos buckets se mezclan y la conclusion se da vuelta sin que nada falle.
"""

from scripts.diagnostico_colecciones import coleccion, texto_consulta


def test_coleccion_agrupa_hermanos_de_la_misma_serie():
    """Dos informes de la misma serie tienen que caer en el mismo grupo: es lo
    que define "hermano equivocado"."""
    assert coleccion("F3-MAPPOEA-030") == coleccion("F3-MAPPOEA-031") == "F3-MAPPOEA"
    assert coleccion("F1-CSET-098") == "F1-CSET"


def test_coleccion_separa_fases_distintas():
    """La fase forma parte del identificador: F1-CSET y un hipotetico F2-CSET
    son colecciones distintas y no deben confundirse."""
    assert coleccion("F1-CSET-001") != coleccion("F2-CSET-001")


def test_coleccion_tolera_un_doc_id_con_otra_forma():
    """No debe reventar con un doc_id que no siga el patron: el diagnostico se
    corre sobre salidas de configuraciones experimentales."""
    assert coleccion("raro") == "raro"
    assert coleccion("") == ""


def test_texto_consulta_ignora_el_query_id():
    """El campo del enunciado cambia de nombre entre versiones del archivo de
    consultas, asi que se busca por forma; el query_id no puede colarse."""
    fila = {
        "query_id": "q001",
        "texto": "¿Como esta transformando la inteligencia artificial la capacidad de los Estados?",
    }
    assert texto_consulta(fila).startswith("¿Como esta transformando")


def test_texto_consulta_sin_enunciado_devuelve_vacio():
    assert texto_consulta({"query_id": "q001"}) == ""
