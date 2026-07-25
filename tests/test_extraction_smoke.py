"""Pruebas de humo de los extractores por formato (sec. 2.1): cada uno debe
devolver texto no vacio y, cuando aplica, descartar boilerplate/campos no
textuales."""

import json

import pandas as pd

from src.extraction.html_extractor import extract_html
from src.extraction.json_extractor import extract_json
from src.extraction.tabular_extractor import extract_csv, extract_xlsx
from src.extraction.text_extractor import extract_text_md


def test_html_extractor_strips_boilerplate_keeps_content(tmp_path):
    # trafilatura necesita una pagina con suficiente senal de contenido real
    # para distinguirlo del boilerplate (nav/footer); un fixture demasiado
    # corto no ejercita bien su heuristica -- de ahi el parrafo largo.
    html = """<!doctype html><html><head><title>Articulo de prueba</title></head><body>
    <nav><a href="/">Inicio</a> | <a href="/publicaciones">Publicaciones</a> |
    <a href="/contacto">Contacto</a> | <a href="/sobre-nosotros">Sobre nosotros</a></nav>
    <header><p class="site-name">Observatorio de Prueba</p></header>
    <article>
      <h1>Titulo del articulo de prueba</h1>
      <p class="byline">Publicado por Observatorio de Prueba</p>
      <p>Este es el contenido principal que debe conservarse integro en la
      extraccion. Describe en detalle un fenomeno relevante para el analisis,
      con suficiente longitud como para que las heuristicas de deteccion de
      contenido principal lo distingan claramente del menu de navegacion y
      del pie de pagina que rodean al articulo.</p>
      <p>Un segundo parrafo refuerza la senal de que este bloque es el
      cuerpo real del documento y no un elemento decorativo de la plantilla.</p>
    </article>
    <aside><p>Suscribase a nuestro boletin para recibir mas analisis como este.</p></aside>
    <footer>
      <p>Todos los derechos reservados 2026 Observatorio de Prueba.</p>
      <p>Politica de privacidad | Terminos de uso | Mapa del sitio</p>
    </footer>
    </body></html>"""
    path = tmp_path / "pagina.html"
    path.write_text(html, encoding="utf-8")

    doc = extract_html(path)

    assert "contenido principal" in doc.texto_crudo
    assert "Inicio" not in doc.texto_crudo
    assert "derechos reservados" not in doc.texto_crudo
    assert doc.formato == "html"


def test_json_extractor_uses_known_body_fields_and_keeps_metadata(tmp_path):
    data = {
        "title": "Titulo de prueba",
        "body_paragraphs": ["Primer parrafo.", "Segundo parrafo."],
        "url": "https://example.org/articulo",
        "authors": ["Autor Uno"],
    }
    path = tmp_path / "articulo.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    doc = extract_json(path)

    assert "Primer parrafo." in doc.texto_crudo
    assert "Segundo parrafo." in doc.texto_crudo
    assert doc.extra_metadata["url"] == "https://example.org/articulo"


def test_json_extractor_fallback_for_unknown_schema(tmp_path):
    data = {"foo": "Texto suelto sin campos conocidos.", "nested": {"bar": "Otro texto."}}
    path = tmp_path / "raro.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    doc = extract_json(path)

    assert "Texto suelto" in doc.texto_crudo
    assert "Otro texto" in doc.texto_crudo


def test_csv_extractor_produces_one_block_per_row(tmp_path):
    path = tmp_path / "datos.csv"
    pd.DataFrame({"pais": ["Colombia", "Brasil"], "indicador": [0.7, 0.6]}).to_csv(path, index=False)

    doc = extract_csv(path)
    bloques = doc.texto_crudo.split("\n\n")

    assert len(bloques) == 2
    assert "pais: Colombia" in bloques[0]
    assert "indicador: 0.7" in bloques[0]


def test_xlsx_extractor_produces_one_block_per_row(tmp_path):
    path = tmp_path / "datos.xlsx"
    pd.DataFrame({"region": ["LatAm"], "valor": [42]}).to_excel(path, index=False)

    doc = extract_xlsx(path)

    assert "region: LatAm" in doc.texto_crudo
    assert doc.formato == "xlsx"


def test_text_extractor_passthrough(tmp_path):
    path = tmp_path / "notas.md"
    path.write_text("# Encabezado\n\nContenido plano.", encoding="utf-8")

    doc = extract_text_md(path)

    assert doc.texto_crudo == "# Encabezado\n\nContenido plano."
    assert doc.formato == "md"
