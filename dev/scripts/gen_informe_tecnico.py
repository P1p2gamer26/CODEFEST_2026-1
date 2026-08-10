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
styles.add(ParagraphStyle("H1custom", parent=styles["Heading1"], spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a2b4c")))
styles.add(ParagraphStyle("H2custom", parent=styles["Heading2"], spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1a2b4c"), fontSize=12))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=4, leading=12.5))
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
        spaceAfter=7,
    )


def build_story() -> list:
    story = []

    story.append(p("CODEFEST AD ASTRA 2026", "TitlePage"))
    story.append(p("Documento tecnico &mdash; Etapa 1: Base de Conocimiento Vectorial", "Subtitle"))

    story.append(h1("1. El corpus y su preprocesamiento"))
    story.append(p(
        "El corpus entregado por ADL son <b>1826 documentos</b> (3 GB) de 21 "
        "observatorios repartidos en los tres fenomenos: 760 PDF, 964 JSON, 26 CSV, "
        "73 PBF, 8 imagenes, 6 XLSX y 1 TXT. La heterogeneidad es el primer problema "
        "de ingenieria del reto: un mismo indice tiene que absorber informes de 400 "
        "paginas de SIPRI o del AI Index, articulos web serializados en JSON, "
        "datasets bibliograficos de decenas de MB y teselas de mapas vectoriales. "
        "De los 1826 documentos, <b>1818 aportan texto</b>; los 8 restantes son 5 "
        "imagenes sin texto legible, un JSON vacio y dos archivos con extension "
        "<i>.pdf</i> que en realidad son paginas HTML de error de la descarga."
    ))
    story.append(p(
        "Cada formato de origen tiene un extractor dedicado que produce un objeto comun "
        "(<i>RawDocument</i>) con el texto crudo y metadata auxiliar: <b>PDF</b> "
        "(pypdfium2, preservando el orden de lectura por pagina y con deteccion "
        "heuristica de boilerplate &mdash; lineas identicas repetidas en >=30% de las "
        "paginas de un documento de 3 o mas paginas se descartan como cabecera/pie de "
        "pagina), <b>HTML</b> (trafilatura, que separa el contenido principal del "
        "menu de navegacion, anuncios y pie de pagina, y preserva encabezados como "
        "senales Markdown), <b>JSON</b> (mapeo explicito de campos conocidos como "
        "<i>title</i>/<i>body_paragraphs</i>/<i>body_text</i>, con fallback generico "
        "para esquemas no anticipados), <b>CSV/XLSX</b> (cada fila se convierte en "
        "pares <i>columna: valor</i> y se trata como unidad atomica de fragmentacion) "
        "e <b>imagenes</b> (OCR opcional via pytesseract, con degradacion controlada "
        "si el binario de tesseract no esta disponible). Tres decisiones merecen "
        "justificacion explicita porque las impuso el corpus real:"
    ))
    story.append(bullets([
        "<b>pypdfium2 en vez de pdfplumber</b> para los PDF. Con 760 PDF y del orden "
        "de 30.000 paginas, el costo de extraccion deja de ser irrelevante: medido "
        "sobre este corpus, pypdfium2 lee unas 3.000 paginas en 4,5 s, mientras que "
        "pdfplumber es dos ordenes de magnitud mas lento y convierte la fase offline "
        "de minutos en horas. Para texto plano de informes la calidad extraida es "
        "equivalente.",
        "<b>OCR selectivo.</b> Cerca del 5% de los PDF &mdash; casi todos los "
        "informes de riesgo de la Defensoria del Pueblo &mdash; son escaneos sin capa "
        "de texto. Cuando un documento rinde menos de 200 caracteres por pagina se "
        "asume escaneo y se rasteriza para pasarlo por tesseract, con un tope de 60 "
        "paginas por documento: sin ese tope un puñado de escaneos largos domina el "
        "tiempo total de la corrida.",
        "<b>Los .pbf no son OpenStreetMap.</b> Las 73 teselas de Amazon Underworld "
        "son <i>Mapbox Vector Tiles</i>, no el formato Protocol Buffer de OSM, asi "
        "que las librerias habituales (pyosmium, pyrosm) no las leen. Se decodifican "
        "con <i>mapbox-vector-tile</i>, se recorren las capas leyendo los atributos "
        "de cada elemento como pares <i>atributo: valor</i>, y se deduplican los "
        "elementos repetidos entre niveles de zoom, tal como indica la sec. 2.1.",
    ]))
    story.append(p(
        "Los CSV/XLSX llevan un tope de 500 filas por archivo: los datasets del AI "
        "Index son volcados bibliograficos de PubMed de hasta 35 MB, ajenos a las "
        "consultas del reto, y sin tope aportarian mas de 100.000 fragmentos de ruido "
        "que dominarian el espacio vectorial. Con tope el documento sigue indexado y "
        "recuperable a nivel documento. La limpieza posterior (normalizacion Unicode "
        "NFC, remocion de caracteres de control, deteccion de idioma) es "
        "deliberadamente conservadora sobre el contenido &mdash; sin lowercasing ni "
        "stemming &mdash; porque la evaluacion compara el campo <i>text</i> de forma "
        "textual contra el ground truth."
    ))
    story.append(p(
        "<b>Reparacion del guion de fin de linea.</b> Las fuentes incrustadas de "
        "muchos PDF no mapean el guion de corte a Unicode y el extractor lo entrega "
        "como U+FFFE, partiendo la palabra: <i>satel&#65534;lites</i>, "
        "<i>weap&#65534;ons</i>. Afectaba al <b>30% de los fragmentos</b> (38.397 de "
        "128.526, en 613 documentos) y degradaba las dos metricas a la vez: rompe "
        "para el tokenizer justo los terminos de dominio que discriminan la consulta, "
        "y entrega al evaluador un fragmento con el texto partido. La reparacion "
        "decide entre unir (<i>satellites</i>) y conservar el guion "
        "(<i>AI-related</i>) contando en el propio corpus cual de las dos formas "
        "aparece mas veces &mdash; 23.177 cortes quedaron resueltos con esa "
        "evidencia &mdash; y solo cae en una heuristica cuando ninguna forma esta "
        "atestiguada. Medido sobre el ground truth propio, la reparacion no cambia "
        "F1@3 de forma significativa (0.306 a 0.310; gana 3 consultas, pierde 3, "
        "empata 35): se adopta porque corrige un defecto del dato y limpia los 166 "
        "fragmentos entregados que salian partidos, no por una ganancia de "
        "recuperacion que la muestra no puede sostener."
    ))
    story.append(h2("1.1 Identificacion de documentos (doc_id)"))
    story.append(p(
        "El emparejamiento a nivel documento se hace con el <b>DOC_ID que suministra "
        "ADL</b> en el inventario del corpus (formato <i>F1-AIINDEX-001</i>), no con "
        "un identificador propio. El manifest se indexa por <b>ruta relativa</b> y no "
        "por nombre de archivo: 59 nombres del inventario aparecen en dos carpetas "
        "distintas con DOC_ID distintos &mdash; los informes de CSET bajo "
        "<i>pdfs/Reports</i> y <i>pdfs/Translation</i>, y las teselas que se repiten "
        "por nivel de zoom &mdash; de modo que indexar por nombre le habria asignado "
        "el identificador equivocado a 127 documentos y anulado su aporte al F1@3. "
        "Once archivos presentes en el corpus no figuran en el inventario de ADL "
        "(catalogos y registros del proceso de descarga); al no tener DOC_ID no "
        "pueden aparecer en el ground truth, y se excluyen del indice para que no "
        "ocupen uno de los tres cupos de documento por consulta."
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
        "<b>Actualizacion:</b> la entrega usa una cascada de <b>tres</b> encoders: "
        "MiniLM trae los candidatos y los re-puntuan <b>gte-multilingual-base</b> "
        "(encoder-only, Apache 2.0, 305 M, 768 dim) y E5, con peso 0,60 cada uno. "
        "Elegida midiendo cinco estructuras: NDCG@10 sube de <b>0,338 a 0,406</b> "
        "(41 consultas) y de 0,329 a 0,360 (10 independientes). GTE como "
        "<i>primario</i> se descarto pese a su mejor F1 en las 41 (0,385) porque "
        "cae a <b>0,200</b> en las independientes: sesgo de pooling."
    ))
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
        "los requieren heredan el comportamiento por defecto. Se verifico ademas "
        "que el bajo rendimiento del secundario en solitario es real y no un error "
        "de uso de esos prefijos: la indexacion aplica <i>passage:</i> y la "
        "consulta <i>query:</i>, tanto en el pipeline como en la copia "
        "autocontenida."
    ))
    story.append(p(
        "Los rankings no se fusionan simetricamente: los encoders secundarios "
        "<b>re-puntuan la lista del primario</b>, sumando su similitud a la del "
        "primario con un peso de 0,60 (los tres terminos son cosenos de la misma "
        "escala, de modo que sumarlos es legitimo). El RRF (ecuacion 7) queda en el "
        "codigo para la variante de varios primarios, que no es la entregada; "
        "fusionar listas distintas se midio y se rechazo (seccion 7), y el grafo "
        "no se incorpora como lista adicional sino como desempate (seccion 5)."
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
        "<b>Resultado de la medicion: la fusion simetrica no sirve, la cascada si.</b> "
        "Comparadas contra el primario solo (F1@3 0,306 sobre las 41 consultas "
        "anotadas), <i>multilingual-e5-base</i> en solitario rinde 0,182 y pierde "
        "7-17, la fusion RRF de ambos rinde 0,268 y pierde 8-13, y la <b>cascada "
        "0,352 ganando 5-0 sin perder ninguna</b>. Sobre las 10 consultas de "
        "anotacion independiente el reparto es el mismo en direccion."
    ))
    story.append(p(
        "<b>Re-medicion tras reparar los guiones (seccion 1).</b> Reconstruidos los "
        "indices sobre el texto reparado la comparacion se sostiene: el primario solo "
        "pasa a <b>0,310</b> y <b>0,300</b>, y la cascada a <b>0,344</b> y "
        "<b>0,333</b>, ganando <b>4-0 y 1-0 sin perder ninguna</b>. La prueba de "
        "signos no alcanza significancia (p = 0,125); se conserva la cascada por el "
        "criterio fijado antes de medir, no por el promedio."
    ))
    story.append(p(
        "<b>Por que la fusion simetrica no podia funcionar.</b> RRF premia el "
        "<i>acuerdo</i> entre listas. Medido sobre las 50 consultas, los dos encoders "
        "comparten apenas el <b>11,3%</b> de los documentos del top-3 y el <b>6,2%</b> "
        "de los fragmentos del top-10. Con ese nivel de desacuerdo RRF no fusiona: "
        "intercala la lista buena con la mala, y el resultado queda a medio camino "
        "entre ambas. Eso es exactamente lo observado (0,306 baja a 0,268)."
    ))
    story.append(p(
        "<b>La cascada, que es lo que se entrega.</b> Lo que E5 hace mal es el "
        "<i>recall</i>: su propio conjunto de candidatos rara vez contiene el "
        "documento correcto. Pero puntuar candidatos que el primario ya encontro es "
        "un trabajo distinto y mas facil. Asi que el primario genera 200 candidatos y "
        "E5 solo los <b>re-puntua</b>, sumando su similitud a la del primario con un "
        "peso de 0,60. Los dos terminos son cosenos, misma escala, de modo que "
        "sumarlos es legitimo. La operacion es barata porque los dos indices se "
        "construyen sobre los mismos fragmentos en el mismo orden: el vector de un "
        "fragmento en el segundo espacio se lee por su fila con <i>reconstruct</i>, "
        "sin volver a codificar ni un pasaje. El costo por consulta es una sola "
        "vectorizacion adicional."
    ))
    story.append(p(
        "El peso paso de 0,25 a <b>0,60</b>. El 0,25 se habia fijado cuando el pool "
        "era de 60 candidatos, la agregacion era <i>sum</i> y no habia glosario; las "
        "tres cosas cambiaron despues. Al ampliar el pool a 100 entran candidatos con "
        "similitud primaria mas baja, donde el mismo peso absoluto pesa mas en "
        "relacion a esa similitud, de modo que el punto de equilibrio se desplaza. Se "
        "barrio la grilla completa 0,10 / 0,25 / 0,40 / 0,60 / 0,75 / 0,90 y el "
        "resultado es una <b>meseta, no una tendencia</b>: 0,60 es el unico valor cuyo "
        "intervalo de confianza al 90% del delta pareado excluye una perdida de 0,02 "
        "en las seis lecturas (F1@3 y NDCG@10 sobre las 50 consultas, sobre las 41 "
        "humanas y sobre las 10 de anotacion independiente). Los valores 0,75 y 0,90 "
        "fallan el criterio sobre las 50. Que no haya tendencia monotona es lo que "
        "impide escalar el hallazgo a la hipotesis distinta de que el re-puntuador "
        "deberia pasar a ser el primario."
    ))

    story.append(p(
        "No se emplea ningun modelo decoder/generativo en ninguna etapa de "
        "indexacion o recuperacion (sec. 4.2 y 8.3): en particular, se descarto "
        "explicitamente HyDE (requiere un LLM para generar una respuesta hipotetica) "
        "y el reranking con LLM. El reranking de los candidatos lo resuelven dos "
        "encoders bi-encoder (gte y e5) que puntuan sobre los vectores ya indexados, "
        "sin releer texto. ADL confirmo que la restriccion de la sec. 8.3 aplica a "
        "las arquitecturas decoder y que un <i>cross-encoder</i> (arquitectura "
        "encoder, no generativa) como re-puntuador esta permitido; su evaluacion "
        "se documenta en la sec. 7."
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
    story.append(p(
        "La decision se verifico empiricamente antes de fijarla. Con vectores de "
        "384 dimensiones y k=120 sobre CPU, una busqueda exacta tarda 1,9 ms "
        "(mediana) con 20.000 vectores, 5,7 ms con 50.000 (9,8 ms en el percentil "
        "99) y 13,2 ms con 100.000. El corpus real produce <b>128.526 fragmentos</b>, "
        "es decir del orden de 15 ms por consulta: irrelevante frente a los ~27 ms "
        "que cuesta vectorizar la consulta y los ~19 s de carga del encoder. "
        "Migrar a IVFFlat o HNSW habria cambiado busqueda exacta por busqueda "
        "aproximada &mdash; perdiendo recall, que es justamente lo que mide la "
        "evaluacion &mdash; a cambio de ahorrar milisegundos que nadie percibe."
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
        "sus vecinos de primer orden en el grafo y se cuenta la evidencia de cada "
        "chunk. Como <b>integracion no desplazante</b>: en lugar de fusionar los "
        "candidatos del grafo como una lista mas via RRF (que medido degrada la "
        "recuperacion), el grafo entra como clave secundaria de orden sobre los "
        "candidatos que el pool vectorial ya recupero, rompiendo <i>solo</i> "
        "empates exactos de score por la evidencia de primer orden. El conjunto "
        "del pool no cambia, asi que la integracion no puede perjudicar la "
        "recuperacion vectorial. Activa con el flag <i>--use-graph</i> de "
        "<i>generador.py</i>."
    ))
    story.append(p(
        "<b>El grafo sobre el corpus real tiene 224.101 nodos y 754.876 aristas</b>, "
        "construidas a partir de 1.687 documentos. Dos decisiones lo hicieron "
        "utilizable. La primera: se excluyen los formatos sin narrativa (CSV, XLSX y "
        "las teselas vectoriales, el 17,7% de los fragmentos), porque el NER busca "
        "entidades nombradas en lenguaje natural y aplicado a una fila de tabla "
        "produce ruido &mdash; las entidades mas frecuentes de una version preliminar "
        "eran <i>FALSO</i>, <i>VERDADEIRO</i> y el nombre de una capa de mapa. La "
        "segunda: los nombres de entidad se limpian de caracteres de control, que el "
        "texto de OCR arrastra y que hacen fallar la exportacion a GraphML, un "
        "formato XML que no los admite."
    ))
    story.append(p(
        "<b>La fusion plena del grafo esta rechazada por medicion; la integracion "
        "adoptada es la no desplazante.</b> Con el ground truth propio (50 consultas), "
        "fusionar los candidatos del grafo con el ranking vectorial por RRF degrada "
        "la recuperacion: F1@3 0,358 contra 0,455 y NDCG@10 0,390 contra 0,516, con "
        "18 consultas en cero. En cambio, el grafo como clave secundaria de orden "
        "(seccion 5) es <b>neutro sobre las 50 consultas</b>: 34 de ellas generan "
        "evidencia de grafo (1.029 apuntes de evidencia en total), pero al no "
        "existir empates exactos de score en el pool, ninguna linea de "
        "<i>resultados.jsonl</i> cambia frente a la corrida sin grafo. Es decir: el "
        "componente bonus queda integrado a la recuperacion, como exige la seccion "
        "7, sin renunciar a ninguna fraccion de la metrica."
    ))

    story.append(h1("6. Modulo de recuperacion"))
    story.append(bullets([
        "Mismo encoder para indexacion y consulta; vector de consulta normalizado "
        "igual que el indice.",
        "Sobre-recuperacion de candidatos antes de aplicar post-filtros de metadata "
        "(fenomeno, formato, idioma) o de umbral de similitud coseno (sec. 8.7).",
        "Agregacion a nivel documento por <b>suma</b> de las puntuaciones de sus "
        "fragmentos, sumando los <b>5 mejores</b> de cada uno, sobre un pool de "
        "<b>100 candidatos</b> "
        "mas amplio que los 10 fragmentos mostrados, para que la relevancia de un "
        "documento no dependa solo de si su mejor chunk aparece en el top-10 "
        "(sec. 8.6). La eleccion de la suma sobre el maximo esta medida, no "
        "supuesta: ver seccion 7.",
        "Combinacion de encoders por <b>suma ponderada de similitudes</b> sobre "
        "una sola lista (la cascada, sec. 3.1): cada re-puntuador anade su coseno "
        "multiplicado por el peso 0,60; RRF (k0=60) queda disponible solo cuando "
        "se piden varios encoders primarios, que no es la configuracion "
        "entregada. El grafo no se fusiona por RRF (medido, degrada) sino que "
        "participa como clave secundaria de orden, ver seccion 5.",
        "<b>Post-filtrado por fenomeno dominante</b> (sec. 8.7): cuando un solo "
        "fenomeno acapara al menos el 80% del pool agregado a documento, el "
        "resultado se restringe a ese fenomeno. La sec. 10.2.2 empareja por "
        "<i>doc_id</i> y los documentos relevantes de casi todas las consultas "
        "viven en un unico fenomeno, asi que filtrar el 20% de candidatos ajenos "
        "concentra los cupos sin riesgo de vaciar la respuesta (medido: F1@3 "
        "0,440 a 0,455). El umbral 0,8 se eligio por robustez, no como argmax.",
        "Limite de 250 palabras por fragmento aplicado dividiendo en limites "
        "oracionales completos cuando un chunk lo excede; los sub-fragmentos "
        "comparten <i>chunk_id</i> y ocupan su propio <i>rank</i> (sec. 9.2.1). "
        "El corpus real obligo a agregar un ultimo recurso: en texto proveniente de "
        "OCR la puntuacion puede desaparecer por completo y el segmentador devuelve "
        "una unica \"oracion\" de mas de 250 palabras, que respetar el limite "
        "oracional dejaria pasar entera. En ese caso &mdash; y solo en ese &mdash; "
        "el corte se hace por palabras.",
        "<b>Supresion de fragmentos con texto repetido.</b> El corpus contiene "
        "documentos duplicados (el mismo informe de CSIS bajo dos <i>doc_id</i>, "
        "series que reeditan capitulos completos), de modo que dos fragmentos "
        "distintos del indice pueden tener texto identico. Entregar el mismo texto "
        "dos veces no puede aportar ganancia en NDCG@10 &mdash; el ranking ideal no "
        "lo contiene &mdash; y ademas desplaza fuera del top-10 a un candidato que si "
        "podria aportarla. Sobre las 50 consultas oficiales, 17 de los 500 "
        "fragmentos entregados eran duplicados exactos; suprimirlos y completar el "
        "cupo con el siguiente candidato recupera esos 17 espacios. La comparacion "
        "es de texto exacto (normalizando solo espacios y mayusculas) y no de "
        "similitud aproximada, que podria descartar fragmentos legitimamente "
        "distintos que comparten un parrafo.",
        "<b>Los fragmentos se ordenan despues de decidir los documentos</b>, con "
        "tres criterios de desempate y sin descartar nada, para que las dos "
        "mitades de la respuesta no se contradigan: primero los fragmentos de los "
        "3 documentos entregados; luego los que el evaluador puede leer (el corpus "
        "trae traducciones al coreano, ruso, arabe, chino y aleman que competian "
        "por los mismos cupos); por ultimo los que son aparato bibliografico, que "
        "la sec. 10.2.1 juzga por su texto y puntuan 0 aunque su documento sea el "
        "correcto. El primer criterio es la mejora individual mas grande del "
        "proyecto: los fragmentos procedentes del top-3 pasan del <b>32% al 98%</b> "
        "y el NDCG@10 sube <b>+0,131</b> (IC 90% [+0,066, +0,199]) sin mover el "
        "F1@3, como corresponde a un cambio que no toca los documentos.",
    ]))
    story.append(p(
        "Ninguna etapa de recuperacion usa un modelo generativo: solo vectores, "
        "puntuaciones de similitud coseno y aritmetica sobre metadata."
    ))

    story.append(h1("7. Como se tomaron las decisiones: medicion interna"))
    story.append(p(
        "El ground truth oficial no es publico durante el reto, de modo que ninguna "
        "decision de diseno puede validarse contra la metrica real. Para no elegir "
        "a ojo se construyo uno propio (<i>dev/eval/</i>) que cubre <b>las 50 "
        "consultas oficiales</b>: 41 con documentos relevantes marcados y 9 en las "
        "que se reviso candidato por candidato sin que ninguno respondiera. Se "
        "construyo en dos etapas y la distincion es importante para no enganarse:"
    ))
    story.append(bullets([
        "<b>10 consultas de anotacion independiente.</b> Los candidatos salieron de "
        "contar palabras clave sobre el texto extraido, <i>sin pasar por el "
        "recuperador</i>. Son las unicas validas para comparar dos encoders entre si.",
        "<b>31 consultas por <i>pooling</i></b> (la tecnica de TREC): los candidatos "
        "los propuso el propio sistema. Aportan volumen, pero una consulta cuyos "
        "candidatos propuso el encoder X favorece a X, porque un documento que X "
        "nunca recupero no pudo marcarse. Por eso cada linea registra su procedencia "
        "y la evaluacion tiene una opcion <i>--sin-pooling</i>.",
    ]))
    story.append(p(
        "El sesgo resulto menor de lo temido: el F1@3 da 0,306 sobre las 41 "
        "consultas y 0,300 sobre las 10 independientes, porque los candidatos se "
        "proponen con un pool de 200 mientras la configuracion entregada agrega "
        "100: las marcas no se cumplen solas."
    ))
    story.append(p(
        "La agregacion a documento <b>suma</b> las puntuaciones de sus fragmentos en "
        "lugar de tomar el <b>maximo</b>, y con el pool ampliado a 100 suma solo los "
        "5 mejores. La razon principal no es el promedio (0,306 frente a 0,226 sobre "
        "las 41 anotadas, pero contando por consulta 16-8 con 17 empates, "
        "<b>p = 0,15</b>: sin significancia) sino el argumento estructural: un "
        "documento relevante contiene <i>varios</i> pasajes relevantes, mientras "
        "que el maximo premia al que tuvo un unico fragmento afortunado. Se "
        "documenta asi y no como resultado medido; la diferencia entre \"lo "
        "medimos\" y \"lo argumentamos y el dato no lo contradice\" es lo que "
        "evita sobreajustar a un ground truth reducido."
    ))
    story.append(p(
        "<b>El promedio de F1@3 no basta para decidir.</b> El tamano del pool parecia "
        "importar (con 26 consultas, 30 daba 0,309 frente a 0,274 con 60, y la ventaja "
        "se repetia con 10, 19 y 26 consultas), pero contando por consulta el "
        "resultado es <b>30 gana en 5, 60 en 3 y empatan 18</b>, que es lo que "
        "produce el azar: con unas 30 consultas cada una pesa 0,033 en la media, y "
        "dos que cambien de lado la mueven mas que cualquier efecto real. El pool "
        "quedo en 60 hasta que una medicion posterior, con criterio de intervalo de "
        "confianza y no de promedio, justifico ampliarlo a 100."
    ))
    story.append(p(
        "El mismo criterio descarto dos hipotesis prometedoras: la recuperacion "
        "entre idiomas distintos (2 de 5 aciertos con documentos en ingles frente a "
        "3 de 5 en espanol) y la confusion entre grupos armados ilegales y fuerzas "
        "armadas estatales (medido con un patron justo, afecta al 8% de los cupos). "
        "El riesgo de un ground truth reducido no es medir de menos sino "
        "<i>sobreajustar</i>."
    ))
    story.append(p(
        "<b>Donde esta el cuello de botella.</b> Antes de seguir ajustando parametros "
        "se midio que impide acertar cuando el sistema falla, separando los tres "
        "motivos de un F1@3 nulo: documento no indexado, ningun fragmento suyo en el "
        "pool, o que entre y pierda la agregacion. De las 17 consultas en cero, "
        "ninguna por indexacion, dos porque el encoder no alcanza el documento y "
        "<b>quince con el documento correcto dentro del pool</b>. El limite no esta "
        "en la recuperacion sino en la agregacion."
    ))
    story.append(p(
        "De ese diagnostico salieron dos intentos, ambos implementados y descartados "
        "por la misma regla. El primero, sumar solo los <i>M</i> mejores fragmentos "
        "de cada documento: la suma sin tope deja que un documento con muchos "
        "fragmentos mediocres desplace a uno con un fragmento excelente. El promedio "
        "mejora (0,347 con pool 100), pero es el maximo de setenta combinaciones y "
        "por consulta reparte 8-3 con 30 empates (<b>p = 0,227</b>). El segundo, BM25 "
        "fusionado con el denso por RRF para rescatar los terminos raros que un "
        "vector de 384 dimensiones diluye: rinde 0,192 frente a 0,306 y pierde 15-4 "
        "(<b>p = 0,019</b>). La hipotesis lexica queda refutada como mejora general."
    ))
    story.append(p(
        "<b>La ultima mejora adoptada es el post-filtrado por fenomeno dominante</b> "
        "(sec. 8.7, detallado en la seccion 6): cuando un solo fenomeno ocupa al "
        "menos el 80% del pool agregado, el resultado se restringe a el. El umbral "
        "0,8 no es el argmax &mdash; 0,7 da mejor F1@3 sobre las 50 &mdash; sino el "
        "unico que no dispara el veto pre-registrado: por debajo, las consultas en "
        "cero suben de 11 a 12, y el filtro no toca las consultas con relevantes en "
        "dos fenomenos a la vez (q019). Cambia 6 de 50 lineas y lleva la entrega a "
        "su <b>estado final</b>: F1@3 = 0,455 y NDCG@10 = 0,516 (0,499 penalizado), "
        "con 11 consultas en cero. El F1@3 no tiende a 1: la sec. 10.2.2 fija "
        "P@3 = aciertos/3 con los tres cupos siempre llenos, asi que una consulta "
        "con un solo documento relevante topa en 0,50 y el techo sobre este ground "
        "truth es <b>0,906</b> &mdash; el 0,455 es el 50% de lo alcanzable, no el "
        "45% de 1."
    ))
    story.append(p(
        "<b>Los dos ultimos cambios ordenan los fragmentos por lo que la sec. 10.2.1 "
        "juzga: su campo</b> <font face='Courier'>text</font>. Un fragmento ilegible, "
        "o que no menciona el objeto de la consulta, vale cero por relevante que sea "
        "su documento. El <b>idioma legible sube por encima de la alineacion con el "
        "top-3</b> (antes un fragmento en coreano del documento n.&ordm; 1 desplazaba "
        "a uno legible del n.&ordm; 4: ilegibles <b>19 a 0</b>) y la <b>cobertura "
        "lexica de la consulta</b> entra como ultimo desempate (fragmentos sin una "
        "palabra de la consulta, <b>175 a 122</b> de 500). Juntos, NDCG@10 "
        "<b>+0,016</b> (IC 90% [+0,004, +0,029], 11-4) y F1@3 inalterado."
    ))
    story.append(p(
        "<b>Las dos ultimas mejoras adoptadas se componen.</b> La primera es "
        "<b>ampliar el pool de 60 a 100</b>: la cascada recuperaba 200 candidatos y "
        "se descartaban 140 antes de agregar, de modo que la profundidad extra solo "
        "reordenaba. La segunda es un <b>glosario biling&uuml;e</b> que expande la "
        "consulta antes de vectorizarla, justificado por una propiedad medible del "
        "corpus: las consultas estan en espanol y los fenomenos 1 y 2 en ingles, y "
        "<i>todos</i> los terminos de dominio que las consultas usan en espanol son "
        "entre 30 y 2.000 veces mas raros que su forma inglesa (NBQR 0 veces contra "
        "66 de CBRN; \"antisatelite\" 8 contra 3.813 de ASAT). Es una tabla escrita "
        "a mano, sin modelo generativo ni traductor (sec. 8.3); una consulta sin "
        "terminos de la tabla se devuelve identica."
    ))
    story.append(p(
        "Medidas sobre el ground truth propio, cada una por separado y las dos "
        "juntas (F1@3 y NDCG@10, primero sobre las 50 consultas y luego sobre las 41 "
        "de anotacion humana): la configuracion anterior daba <b>0,363 / 0,392</b> y "
        "<b>0,386 / 0,405</b>; solo el glosario, 0,379 / 0,403 y 0,396 / 0,406; solo "
        "el pool ampliado, 0,366 / 0,434 y 0,398 / 0,447; y <b>las dos juntas, que "
        "son las que se entregan, 0,402 / 0,457 y 0,424 / 0,456</b>. Los efectos se "
        "componen: ninguna de las dos anula a la otra."
    ))
    story.append(p(
        "<b>La ultima ronda anadio dos palancas mas, y tambien se componen.</b> El "
        "peso del re-puntuador paso de 0,25 a 0,60 (sec. 3.1) y el glosario gano tres "
        "entradas &mdash; <i>derecho internacional en el espacio</i>, <i>dominio "
        "espacial</i> y <i>sistemas no tripulados</i> &mdash; medidas una por una, "
        "cada una ganando una consulta sin perder ninguna. Que se compusieran no era "
        "obvio: el glosario cambia <i>que</i> candidatos entran al pool y el peso "
        "reordena el pool ya recuperado, de modo que un re-puntuador con mas peso "
        "podia hundir justo los documentos que el glosario acababa de rescatar. "
        "Medido, no ocurre: partiendo de 0,402 / 0,457 sobre las 50, solo el peso da "
        "0,425 / 0,476, solo el glosario 0,423 / 0,486, y <b>las dos juntas 0,440 / "
        "0,490</b>, mejor que cualquiera por separado en las cuatro lecturas "
        "comparadas."
    ))
    story.append(p(
        "<b>La salvedad, que se declara junto al numero:</b> contra el glosario solo, "
        "la ganancia sobre las 50 consultas es estrecha (NDCG@10 +0,004). Lo que "
        "sostiene la combinacion es el F1@3 sobre las 10 consultas de anotacion "
        "independiente, +0,067 con intervalo al 90% de [+0,000, +0,133] &mdash; la "
        "unica muestra libre del sesgo de pooling, y tambien la mas pequena: una sola "
        "consulta la mueve 0,033. Sobre las 50, el cambio gana 9 consultas y "
        "<i>pierde</i> 4."
    ))
    story.append(p(
        "El glosario dejo ademas un hallazgo que acota su propio alcance: <b>la "
        "asimetria espanol/ingles existe solo en los fenomenos 1 y 2</b>. En el "
        "fenomeno 3, territorial y colombiano, la relacion se invierte "
        "(\"reclutamiento\" en 682 fragmentos contra 2 de <i>child recruitment</i>), "
        "de modo que expandir al ingles una consulta de ese fenomeno la alejaria de "
        "sus documentos: es una herramienta de dos fenomenos, no de tres. Por la "
        "misma razon se rechazo <i>capacidades laser</i> &rarr; <i>laser weapons</i>: "
        "\"laser\" es cognado y la consulta ya tenia puente al corpus ingles."
    ))
    story.append(p(
        "Con el pool ampliado la agregacion pasa de <i>suma</i> a <b>sumar los 5 "
        "mejores fragmentos</b> de cada documento, por robustez: con pool 60 las dos "
        "son identicas, pero al ampliarlo la suma pierde el tope y un documento con "
        "muchos fragmentos mediocres desplaza al bueno &mdash; con pool 200 cae a "
        "<b>0,298</b> mientras el tope de 5 sube a 0,406."
    ))
    story.append(p(
        "<b>Lo que cuestan, dicho explicitamente:</b> sobre las 10 consultas de "
        "anotacion independiente el F1@3 baja de 0,333 a 0,300 y el NDCG@10 de "
        "0,360 a 0,338. Es <i>una sola consulta</i> la que cambia de lado y el "
        "intervalo de confianza con esa muestra no distingue la perdida de cero, "
        "pero es la unica muestra sin sesgo de pooling y corresponde decir que ahi "
        "no confirman. Se adoptan porque ganan de forma consistente en las 41 de "
        "anotacion humana &mdash; F1@3 +0,038 (IC 90% [+0,002, +0,073]) y NDCG@10 "
        "+0,051 ([+0,008, +0,095]) &mdash; y porque las dos tienen explicacion "
        "mecanica previa a la medicion. Otros dos ajustes se midieron y "
        "<i>no</i> se adoptaron: reservar cupos de fragmento degrada el NDCG@10 de "
        "forma monotona (&minus;0,025 con 8 cupos, &minus;0,082 con 4) sin mover el "
        "F1@3, y el prior de recencia resulta inerte; quedan implementados y "
        "apagados. El grafo como fusion pierde 11-0 (seccion 5); como clave "
        "secundaria de orden no desplaza nada y se adopta sin tocar las metricas."
    ))
    story.append(p(
        "La entrega se verifica de punta a punta antes de empaquetarse con "
        "<i>scripts/validar_entrega.py</i>, que comprueba la estructura de "
        "carpetas, las 50 lineas, los 3 documentos y 10 fragmentos por consulta, el "
        "limite de 250 palabras, la alineacion entre el indice FAISS y "
        "<i>metadata.jsonl</i>, y que todo <i>doc_id</i> reportado pertenezca al "
        "inventario de ADL. La suite de pruebas (161 casos) cubre los invariantes "
        "criticos, incluida la ejecucion de <i>generador.py</i> como subproceso con "
        "<i>PYTHONPATH</i> vacio para garantizar que es autocontenido y que los "
        "resultados son reproducibles, requisito del punto 4 de la sec. 1.4."
    ))

    story.append(h1("8. Limitaciones conocidas y decisiones documentadas"))
    story.append(bullets([
        "<b>El ground truth propio es la limitacion dominante del proyecto.</b> "
        "Los valores de F1@3 de la seccion 7 sirven para ordenar configuraciones "
        "entre si, no para estimar la nota: la anotacion es parcial frente a un "
        "corpus de 1826 documentos, asi que un documento relevante no anotado cuenta "
        "como error y el valor real sera mayor. Ademas se juzga el documento viendo "
        "un solo fragmento &mdash; el mejor de cada documento, que no siempre es la "
        "mejor evidencia &mdash; lo que hace marcar de menos, nunca de mas.",
        "<b>El NDCG@10 se mide con un proxy:</b> la relevancia de cada fragmento "
        "se hereda de su documento, porque anotarla de verdad exigiria relevancia "
        "graduada fragmento por fragmento. El proxy sobreestima &mdash; da 1 a la "
        "bibliografia de un documento relevante &mdash; y cuanto, esta acotado por "
        "abajo: descontando el aparato bibliografico da 0,499 frente a 0,516. El "
        "otro modo de fallo, un pasaje que no responde dentro de un documento que "
        "si es relevante, no lo ve ninguna medicion automatica.",
        "<b>Documentos sin texto recuperable:</b> 8 de los 1826. Las cinco imagenes "
        "no aportan texto (una en AVIF, que Pillow no lee sin plugin) pero quedan "
        "presentes en <i>metadata.jsonl</i> con una fila sin vector (<i>en_indice</i> "
        "= false), manteniendo la alineacion indice&ndash;metadata completa. Los "
        "otros tres &mdash; un JSON de 0 bytes y dos HTML de error con extension "
        ".pdf, detectados por contenido &mdash; no aportan texto y quedan sin "
        "posibilidad de ser recuperados.",
        "<b>Campo <i>fuente</i>:</b> se reporta el nombre de archivo estandarizado "
        "del inventario de ADL, nunca la URL (que se conserva en un campo <i>url</i> "
        "adicional). La sec. 10.2.1 sugeria <i>fuente</i> como clave de "
        "emparejamiento y ADL aclaro que es <i>doc_id</i>; el nombre de archivo "
        "funciona bajo cualquiera de las dos lecturas.",
        "<b>Fusion de chunks adyacentes</b> (sec. 9.2.1, opcional): implementada y "
        "revertida &mdash; el chunker solapa una oracion y concatenar duplica texto.",
        "<b>Boilerplate de PDF:</b> el filtro de lineas repetidas pide 3 o mas "
        "paginas.",
    ]))

    return story


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=LETTER,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="CODEFEST AD ASTRA 2026 - Documento Tecnico Etapa 1",
    )
    doc.build(build_story())
    print(f"generado: {OUT_PATH}")


if __name__ == "__main__":
    main()
