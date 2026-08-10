"""Pruebas de la reconstruccion de oraciones para la rejilla de chunking (E38).

No hay checkpoint del texto crudo de los documentos, solo de los chunks ya
fragmentados. Re-chunkear a un presupuesto MAYOR obliga a reconstruir la
secuencia original de oraciones quitando el solape que el chunker introdujo.

Si esa reconstruccion pierde o duplica texto, las seis celdas de la rejilla
miden sobre un corpus corrupto y ninguna medicion valdria nada -- sin que nada
falle a la vista. De ahi que estos casos sean la puerta de entrada del
experimento y no un detalle de implementacion.
"""

from scripts.rechunkear_e38 import reconstruir_oraciones, reempaquetar


def test_quita_la_oracion_de_solape_entre_chunks_contiguos():
    # El segundo chunk repite la ultima oracion del primero, que es justo lo
    # que hace CHUNK_OVERLAP_SENTENCES=1.
    chunks = [
        {"posicion": 0, "titulo_seccion": "S1", "idioma": "es",
         "texto": "Alfa uno. Beta dos. Gamma tres."},
        {"posicion": 1, "titulo_seccion": "S1", "idioma": "es",
         "texto": "Gamma tres. Delta cuatro."},
    ]
    secciones = reconstruir_oraciones(chunks)
    assert len(secciones) == 1
    heading, oraciones = secciones[0]
    assert heading == "S1"
    assert oraciones == ["Alfa uno.", "Beta dos.", "Gamma tres.", "Delta cuatro."]


def test_no_deduplica_una_repeticion_legitima_no_contigua():
    # La misma oracion dos veces, pero NO formando el sufijo/prefijo de la
    # frontera: es contenido real del documento y tiene que conservarse.
    chunks = [
        {"posicion": 0, "titulo_seccion": None, "idioma": "es",
         "texto": "Alfa uno. Beta dos."},
        {"posicion": 1, "titulo_seccion": None, "idioma": "es",
         "texto": "Alfa uno. Delta cuatro."},
    ]
    _, oraciones = reconstruir_oraciones(chunks)[0]
    assert oraciones == ["Alfa uno.", "Beta dos.", "Alfa uno.", "Delta cuatro."]


def test_separa_por_seccion():
    chunks = [
        {"posicion": 0, "titulo_seccion": "A", "idioma": "es", "texto": "Uno."},
        {"posicion": 1, "titulo_seccion": "B", "idioma": "es", "texto": "Dos."},
    ]
    secciones = reconstruir_oraciones(chunks)
    assert [h for h, _ in secciones] == ["A", "B"]


def test_reconstruccion_conserva_toda_oracion_no_solapada():
    chunks = [
        {"posicion": 0, "titulo_seccion": "S", "idioma": "es",
         "texto": "Uno uno. Dos dos. Tres tres."},
        {"posicion": 1, "titulo_seccion": "S", "idioma": "es",
         "texto": "Tres tres. Cuatro cuatro."},
    ]
    _, oraciones = reconstruir_oraciones(chunks)[0]
    unidas = " ".join(oraciones)
    for esperada in ["Uno uno.", "Dos dos.", "Tres tres.", "Cuatro cuatro."]:
        assert esperada in unidas
    assert unidas.count("Tres tres.") == 1


def _doc(texto, formato="pdf"):
    return [{
        "doc_id": "D1", "posicion": 0, "titulo_seccion": None, "idioma": "es",
        "formato": formato, "fuente": "a.pdf", "fenomeno": 1, "url": "",
        "texto": texto,
    }]


def test_reempaquetar_conserva_la_identidad_del_documento():
    salida = reempaquetar(_doc("Uno uno uno. Dos dos dos. Tres tres tres."),
                          token_budget=280, overlap_sentences=1,
                          count_tokens=lambda t: len(t.split()))
    assert len(salida) == 1
    assert salida[0]["doc_id"] == "D1"
    assert salida[0]["posicion"] == 0
    assert salida[0]["chunk_id"] == "D1::0"
    assert "Tres tres tres." in salida[0]["texto"]
    assert sorted(salida[0]) == [
        "chunk_id", "doc_id", "fenomeno", "formato", "fuente", "idioma",
        "num_tokens", "posicion", "texto", "titulo_seccion", "url",
    ]


def test_presupuesto_menor_produce_mas_chunks():
    ct = lambda t: len(t.split())
    doc = _doc("Uno uno uno. Dos dos dos. Tres tres tres. Cuatro cuatro.")
    assert len(reempaquetar(doc, 4, 0, ct)) > len(reempaquetar(doc, 280, 1, ct))


def test_los_tabulares_no_se_re_empaquetan():
    """CSV y XLSX se fragmentan por filas, el presupuesto de tokens no aplica."""
    doc = _doc("fila uno. fila dos.", formato="csv")
    salida = reempaquetar(doc, 4, 0, lambda t: len(t.split()))
    assert len(salida) == 1
    assert salida[0]["texto"] == "fila uno. fila dos."


def test_la_puerta_acepta_borrar_el_solape():
    from scripts.rechunkear_e38 import _es_subsecuencia
    assert _es_subsecuencia("holamundo", "holaholamundo")


def test_la_puerta_RECHAZA_texto_inventado():
    """Control negativo: una puerta que nunca falla no sirve de nada."""
    from scripts.rechunkear_e38 import _es_subsecuencia
    assert not _es_subsecuencia("holaxmundo", "holamundo")


def test_la_puerta_RECHAZA_texto_reordenado():
    from scripts.rechunkear_e38 import _es_subsecuencia
    assert not _es_subsecuencia("mundohola", "holamundo")
