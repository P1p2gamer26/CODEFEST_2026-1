"""Genera Entrega/informe_tecnico.pdf (maximo 8 paginas, sec. 1.4 punto 3 de
la especificacion): decisiones de diseno del pipeline -- chunking, encoder(s),
indice FAISS, grafo de conocimiento y limitaciones conocidas.

Uso: python scripts/gen_informe_tecnico.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]  # dev/scripts/ -> dev/ -> raiz del repo
OUT_PATH = ROOT / "Entrega" / "informe_tecnico.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1custom", parent=styles["Heading1"], spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a2b4c")))
styles.add(ParagraphStyle("H2custom", parent=styles["Heading2"], spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1a2b4c"), fontSize=12))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=6, leading=14))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle("TitlePage", parent=styles["Title"], fontSize=20, spaceAfter=4))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#555555"), spaceAfter=20))


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def h1(text: str) -> Paragraph:
    return Paragraph(text, styles["H1custom"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, styles["H2custom"])


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item), bulletColor=colors.HexColor("#1a2b4c")) for item in items],
        bulletType="bullet",
        leftIndent=14,
        spaceBefore=2,
        spaceAfter=8,
    )


def build_story() -> list:
    story = []

    story.append(p("CODEFEST AD ASTRA 2026", "TitlePage"))
    story.append(p("Documento tecnico &mdash; Etapa 1: Base de Conocimiento Vectorial", "Subtitle"))

    story.append(h1("1. Preprocesamiento y extraccion de texto"))
    story.append(p(
        "Cada formato de origen tiene un extractor dedicado que produce un objeto comun "
        "(<i>RawDocument</i>) con el texto crudo y metadata auxiliar: <b>PDF</b> "
        "(pdfplumber, preservando el orden de lectura por pagina y con deteccion "
        "heuristica de boilerplate &mdash; lineas identicas repetidas en >=30% de las "
        "paginas de un documento de 3 o mas paginas se descartan como cabecera/pie de "
        "pagina), <b>HTML</b> (trafilatura, que separa el contenido principal del "
        "menu de navegacion, anuncios y pie de pagina, y preserva encabezados como "
        "senales Markdown), <b>JSON</b> (mapeo explicito de campos conocidos como "
        "<i>title</i>/<i>body_paragraphs</i>/<i>body_text</i>, con fallback generico "
        "para esquemas no anticipados), <b>CSV/XLSX</b> (cada fila se convierte en "
        "pares <i>columna: valor</i> y se trata como unidad atomica de fragmentacion) "
        "e <b>imagenes</b> (OCR opcional via pytesseract, con degradacion controlada "
        "si el binario de tesseract no esta disponible). La limpieza posterior "
        "(normalizacion Unicode NFC, remocion de caracteres de control, deteccion de "
        "idioma) es deliberadamente conservadora sobre el contenido &mdash; sin "
        "lowercasing ni stemming &mdash; porque la evaluacion compara el campo "
        "<i>text</i> de forma textual contra el ground truth."
    ))

    story.append(h1("2. Estrategia de chunking y justificacion"))
    story.append(p(
        "Se implemento una estrategia <b>hibrida estructural + oracional con "
        "solapamiento</b>, en tres niveles:"
    ))
    story.append(bullets([
        "<b>Estructural:</b> el texto se separa primero por encabezados Markdown "
        "(que HTML/JSON/MD ya traen marcados por el extractor); si el documento no "
        "tiene encabezados (caso tipico de PDF), se trata como una unica seccion.",
        "<b>Segmentacion de oraciones multilingue:</b> dentro de cada seccion, se "
        "usa <i>pysbd</i> (reglas linguisticas dedicadas) para espanol e ingles, y el "
        "parser entrenado de spaCy (<i>pt_core_news_sm</i>) para portugues, que "
        "pysbd no cubre nativamente.",
        "<b>Empaquetado voraz por presupuesto de tokens</b> (~280 tokens, margen bajo "
        "el limite de 512 del encoder), contando tokens con el tokenizer real del "
        "encoder de indexacion &mdash; no una aproximacion por palabras &mdash; con "
        "<b>solapamiento de 1 oracion</b> entre chunks consecutivos para no perder "
        "ideas que caen en la frontera.",
    ]))
    story.append(p(
        "La unidad minima que se mueve entre chunks es siempre una oracion completa: "
        "el empaquetador nunca corta un caracter a mitad de oracion, cumpliendo el "
        "requisito de completitud linguistica (sec. 3.3). Esta garantia se verifica "
        "automaticamente en <i>tests/test_chunking.py</i>. La unica excepcion "
        "estructural es CSV/XLSX, donde cada fila (no una oracion en lenguaje "
        "natural) se indexa como chunk atomico, tal como permite explicitamente la "
        "especificacion."
    ))

    story.append(h1("3. Encoder(s) seleccionado(s) y criterios de eleccion"))
    story.append(p(
        "Encoder primario: <b>sentence-transformers/paraphrase-multilingual-"
        "MiniLM-L12-v2</b> (arquitectura encoder tipo BERT, licencia Apache 2.0)."
    ))
    cell_style = ParagraphStyle("TableCell", parent=styles["Small"], fontSize=8.5, leading=10.5, textColor=colors.black)
    header_style = ParagraphStyle("TableHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")
    raw_rows = [
        ["Criterio", "Justificacion"],
        ["Soporte multilingue", "Cubre nativamente espanol, ingles y portugues (50+ idiomas), "
         "necesario porque el corpus y las consultas mezclan los tres."],
        ["Dimensionalidad", "384 dimensiones: balance razonable entre costo de almacenamiento/"
         "busqueda y expresividad para el volumen esperado del reto."],
        ["Longitud maxima de entrada", "512 tokens; el presupuesto de chunking (~280 tokens) deja "
         "margen suficiente."],
        ["Licencia", "Apache 2.0, uso libre sin restricciones para el contexto del reto."],
        ["Eficiencia computacional", "Modelo liviano (backbone MiniLM), adecuado para los recursos "
         "de computo limitados del equipo."],
        ["Riesgo de reproducibilidad", "Ya validado en el material de apoyo entregado por el equipo "
         "organizador (01_embeddings.ipynb, 02_rag.ipynb), reduciendo el riesgo de "
         "incompatibilidades de ultimo momento."],
    ]
    table_data = [
        [Paragraph(c, header_style if i == 0 else cell_style) for c in row]
        for i, row in enumerate(raw_rows)
    ]
    tbl = Table(table_data, colWidths=[4.2 * cm, 11.3 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(h2("3.1 Encoder secundario y fusion de rankings (sec. 4.4)"))
    story.append(p(
        "El pipeline soporta construir la base con mas de un encoder. El "
        "secundario es <b>intfloat/multilingual-e5-base</b> (encoder tipo BERT, "
        "licencia MIT, 768 dimensiones, multilingue nativo, limite 512 tokens). "
        "El criterio de seleccion no fue elegir un modelo \"mejor\" en abstracto, "
        "sino uno con <b>errores decorrelacionados</b> respecto al primario: E5 "
        "esta entrenado para recuperacion densa mientras que el primario esta "
        "afinado para similitud de parafrasis. Fusionar dos rankings solo aporta "
        "cuando los modelos se equivocan en casos distintos; dos encoders de la "
        "misma familia habrian aportado poco."
    ))
    story.append(p(
        "La familia E5 exige prefijos distintos para consulta (<i>query:</i>) y "
        "pasaje (<i>passage:</i>); omitirlos degrada la calidad silenciosamente. "
        "Por eso la interfaz <i>Encoder</i> distingue codificacion asimetrica "
        "(<i>encode_query</i> / <i>encode_passages</i>), y los encoders que no "
        "los requieren heredan el comportamiento por defecto."
    ))
    story.append(p(
        "Las listas se combinan con <b>Reciprocal Rank Fusion</b> (RRF, ecuacion 7, "
        "con k0=60), elegido sobre CombSUM/CombMNZ porque opera sobre posiciones y "
        "no sobre puntuaciones absolutas: es robusto a que dos encoders produzcan "
        "similitudes en escalas distintas. El grafo de conocimiento se incorpora "
        "como una lista adicional a fusionar, tal como sugiere la sec. 8.5."
    ))
    story.append(p(
        "<b>Invariante de correccion:</b> la fusion empareja fragmentos por "
        "<i>chunk_id</i>, lo que solo es valido si todos los indices comparten los "
        "mismos fragmentos. Como el presupuesto de chunking depende del tokenizer, "
        "fragmentar por separado con cada encoder produciria <i>chunk_id</i> "
        "colisionantes apuntando a textos distintos. Por eso el chunking se ejecuta "
        "una unica vez y sus fragmentos se reutilizan para todos los encoders; "
        "<i>generador.py</i> ademas verifica la coincidencia al cargar y aborta si "
        "detecta indices desincronizados, en lugar de producir resultados "
        "silenciosamente incorrectos."
    ))
    story.append(p(
        "Dado que el <i>ground truth</i> no es publico durante el reto, no es "
        "posible medir NDCG@10 ni F1@3 para decidir entre uno y dos encoders. Se "
        "incluye <i>scripts/compare_encoders.py</i>, que cuantifica el solapamiento "
        "entre ambos rankings: un solapamiento alto indica que fusionar no aportaria "
        "y que conviene entregar un solo indice. La configuracion multi-encoder "
        "queda por tanto detras de un flag, desactivada por defecto."
    ))

    story.append(p(
        "No se emplea ningun modelo decoder/generativo en ninguna etapa de "
        "indexacion o recuperacion (sec. 4.2 y 8.3): en particular, se descarto "
        "explicitamente HyDE (requiere un LLM para generar una respuesta hipotetica) "
        "y el reranking con LLM. El uso de un <i>cross-encoder</i> (arquitectura "
        "encoder, no generativa, demostrado en el material de apoyo) quedo "
        "implementado detras de un flag desactivado por defecto, como exploracion "
        "documentada, para no depender de el en el flujo evaluado."
    ))

    story.append(h1("4. Indice FAISS"))
    story.append(p(
        "Se utiliza <b>IndexFlatIP</b> sobre vectores normalizados a norma unitaria, "
        "equivalente a similitud coseno (sec. 5.2 y 8.2). Es la opcion recomendada "
        "por la propia especificacion para el volumen de documentos esperado en "
        "esta etapa del reto: garantiza resultados exactos (sin la perdida de "
        "exactitud de IVFFlat/HNSW) y su costo computacional es totalmente asumible "
        "a esta escala. El identificador interno de FAISS se mantiene alineado "
        "linea a linea con <i>metadata.jsonl</i> (mismo orden de insercion que de "
        "escritura), invariante verificado en <i>tests/test_faiss_alignment.py</i>. "
        "Si el corpus real de ADL resultara mucho mayor al esperado, "
        "<i>src/config.py</i> centraliza el punto donde migrar a un indice "
        "aproximado sin tocar el resto del pipeline."
    ))

    story.append(h1("5. Grafo de conocimiento (componente bonus)"))
    story.append(p(
        "<b>NER:</b> se usa el componente de reconocimiento de entidades ya "
        "entrenado en los modelos spaCy <i>es/en/pt_core_news_sm</i> (licencia MIT, "
        "arquitectura no generativa), filtrando tipos de entidad puramente "
        "numericos/temporales (fechas, cantidades, porcentajes) para quedarse con "
        "personas, organizaciones, lugares y otras entidades nombradas relevantes."
    ))
    story.append(p(
        "<b>Extraccion de relaciones:</b> heuristica basada en dependencias "
        "sintacticas (permitida explicitamente por la especificacion como "
        "alternativa a un modelo entrenado de RE): para cada oracion con al menos "
        "dos entidades, se toma el verbo/auxiliar entre un par de entidades "
        "consecutivas como etiqueta de relacion (o el verbo raiz de la oracion si "
        "no hay uno intermedio), con una relacion generica de co-ocurrencia como "
        "ultimo recurso."
    ))
    story.append(p(
        "<b>Construccion e integracion:</b> las tripletas se acumulan en un "
        "<i>networkx.MultiDiGraph</i>, cada arista con <i>doc_id</i>, <i>chunk_id</i> "
        "y la relacion como evidencia trazable, exportado a GraphML. En "
        "recuperacion, las mismas entidades se detectan en la consulta, se buscan "
        "sus vecinos de primer orden en el grafo, y los chunks vinculados se "
        "fusionan con el ranking vectorial mediante Reciprocal Rank Fusion "
        "(el grafo se trata como un indice adicional, sec. 8.5), detras de un flag "
        "<i>--use-graph</i> opcional en <i>generador.py</i>."
    ))

    story.append(h1("6. Modulo de recuperacion"))
    story.append(bullets([
        "Mismo encoder para indexacion y consulta; vector de consulta normalizado "
        "igual que el indice.",
        "Sobre-recuperacion de candidatos antes de aplicar post-filtros de metadata "
        "(fenomeno, formato, idioma) o de umbral de similitud coseno (sec. 8.7).",
        "Agregacion a nivel documento por <i>max pooling</i> (configurable a suma o "
        "media) sobre un pool de candidatos mas amplio que los 10 fragmentos "
        "mostrados, para que la relevancia de un documento no dependa solo de si su "
        "mejor chunk aparece en el top-10 (sec. 8.6).",
        "Reciprocal Rank Fusion (k0=60) como mecanismo unico de combinacion, "
        "reusado tanto para multiples encoders como para vector+grafo (sec. 8.4).",
        "Limite de 250 palabras por fragmento aplicado dividiendo en limites "
        "oracionales completos cuando un chunk lo excede; los sub-fragmentos "
        "comparten <i>chunk_id</i> y ocupan su propio <i>rank</i> (sec. 9.2.1).",
    ]))
    story.append(p(
        "Ninguna etapa de recuperacion usa un modelo generativo: solo vectores, "
        "puntuaciones de similitud coseno y aritmetica sobre metadata."
    ))

    story.append(h1("7. Limitaciones conocidas y decisiones documentadas"))
    story.append(bullets([
        "<b>Corpus de ejemplo sintetico:</b> el entorno de desarrollo usado para "
        "construir este pipeline bloquea (proxy de salida, politica 403) la "
        "descarga de documentos reales de observatorios/centros de pensamiento. "
        "El corpus en <i>corpus_ejemplo/</i> (15 documentos, PDF/HTML, ES/EN/PT) fue "
        "redactado a mano, tematicamente inspirado en fuentes reales conocidas "
        "(SIPRI, ESA Space Debris Office, CEPAL, Instituto Kroc, entre otras) pero "
        "NO es una copia ni cita textual de ningun documento real. Se reemplaza "
        "integramente por el corpus oficial de ADL sin cambios en <i>src/</i>.",
        "<b>huggingface.co tambien bloqueado</b> en el mismo entorno: no fue posible "
        "descargar los pesos reales del encoder ni el modelo de NER de "
        "HuggingFace originalmente considerado para el grafo. El encoder esta "
        "detras de una interfaz intercambiable "
        "(<i>SentenceTransformerEncoder</i> real vs. <i>HashingFakeEncoder</i> "
        "determinista de pruebas); toda la suite de tests y las corridas de "
        "demostracion de este documento usan el encoder de pruebas, que valida la "
        "mecanica de indexacion/recuperacion/formato de salida pero NO la calidad "
        "semantica de la recuperacion. El indice con embeddings reales debe "
        "generarse corriendo <i>scripts/build_corpus_index.py</i> en un entorno con "
        "acceso normal a internet antes de la entrega final.",
        "<b>PBF (mapas):</b> no implementado en esta iteracion; es un formato nicho "
        "que requiere una libreria dedicada (osmium/pyrosm) y no se considero "
        "razonable fabricar datos geoespaciales sinteticos. La interfaz del "
        "extractor esta lista para completarse cuando el equipo tenga archivos PBF "
        "reales con los que validar el parseo.",
        "<b>Campo <i>fuente</i>:</b> se deriva hoy del nombre de archivo o la URL "
        "detectada por el extractor HTML. Es la clave real de emparejamiento con el "
        "ground truth a nivel documento (sec. 10.2.1), y esta aislada en una unica "
        "funcion (<i>derive_fuente()</i>) para ajustarla facilmente si ADL usa otra "
        "convencion.",
        "<b>Fusion de chunks cortos adyacentes</b> (sec. 9.2.1, mencionada como "
        "posible, no obligatoria) no esta implementada: solo se implemento la "
        "division obligatoria de chunks que exceden 250 palabras.",
        "<b>Boilerplate de PDF:</b> la heuristica de lineas repetidas (>=30% de "
        "paginas) requiere documentos de 3 o mas paginas; PDFs mas cortos no "
        "reciben este filtro.",
    ]))

    return story


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="CODEFEST AD ASTRA 2026 - Documento Tecnico Etapa 1",
    )
    doc.build(build_story())
    print(f"generado: {OUT_PATH}")


if __name__ == "__main__":
    main()
