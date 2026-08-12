"""Vigila que Entrega/generador.py no se quede atras de dev/src/.

`Entrega/generador.py` es autocontenido a proposito: no importa nada de
`dev/src/`, porque el evaluador de ADL corre ese unico archivo desde fuera del
repo. El precio de esa decision es que **todo cambio en la logica de
recuperacion vive por duplicado**, y un cambio aplicado solo en `dev/src/`
mejora los barridos internos y NO llega a la entrega.

Ya paso: el gate de aparato bibliografico se aplico en
`dev/src/retrieval/truncate.py` y en `Entrega/generador.py` quedo a medias
(solo el `import re`), asi que las dos mitades del repo puntuaban distinto
sobre los mismos fragmentos. Esta prueba existe para que la proxima vez lo diga
pytest y no una entrega ya subida.

Compara COMPORTAMIENTO sobre el texto real de los 500 fragmentos entregados, no
el codigo fuente: dos implementaciones pueden estar escritas distinto (la
aplanada no puede importar) y aun asi ser correctas. Lo que no puede pasar es
que den numeros distintos.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DEV_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEV_DIR.parent
GENERADOR_PATH = REPO_ROOT / "Entrega" / "generador.py"
RESULTADOS_PATH = REPO_ROOT / "Entrega" / "resultados.jsonl"


def _cargar_generador():
    """Importa el script de entrega como modulo.

    Hay que registrarlo en sys.modules ANTES de ejecutarlo: los `@dataclass`
    del archivo resuelven sus anotaciones mirando
    `sys.modules[cls.__module__].__dict__`, y sin el registro previo eso es
    None y la importacion revienta.
    """
    spec = importlib.util.spec_from_file_location("generador_entrega", GENERADOR_PATH)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["generador_entrega"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def _textos_reales() -> list[tuple[str, str]]:
    if not RESULTADOS_PATH.exists():
        pytest.skip(f"no existe {RESULTADOS_PATH}")
    return [
        (frag["chunk_id"], frag["text"])
        for linea in RESULTADOS_PATH.read_text(encoding="utf-8").splitlines()
        if linea.strip()
        for frag in json.loads(linea)["fragments"]
    ]


def test_el_detector_de_aparato_puntua_igual_en_las_dos_copias():
    from src.retrieval.calidad_chunk import fraccion_aparato as en_dev

    entrega = _cargar_generador()
    textos = _textos_reales()
    assert textos, "resultados.jsonl no tiene fragmentos"

    discrepantes = [
        chunk_id
        for chunk_id, texto in textos
        if abs(en_dev(texto) - entrega.fraccion_aparato(texto)) > 1e-12
    ]
    assert not discrepantes, (
        f"{len(discrepantes)} de {len(textos)} fragmentos puntuan distinto en "
        f"dev/src/retrieval/calidad_chunk.py y Entrega/generador.py "
        f"(primeros: {discrepantes[:5]}). Hay que re-aplanar el cambio."
    )


def test_el_umbral_de_aparato_es_el_mismo():
    """Un umbral distinto en cada copia da el mismo puntaje por fragmento y aun
    asi ordena distinto, que es justo lo que la prueba anterior no ve."""
    from src.retrieval.truncate import UMBRAL_APARATO as en_dev

    assert en_dev == _cargar_generador().UMBRAL_APARATO


def test_los_idiomas_legibles_son_los_mismos():
    from src.retrieval.truncate import IDIOMAS_LEGIBLES as en_dev

    assert en_dev == _cargar_generador().IDIOMAS_LEGIBLES


def test_el_prior_de_recencia_se_comporta_igual_en_las_dos_copias():
    """Mismo criterio que el detector de aparato: se compara comportamiento.

    Se prueba con nombres reales del inventario de ADL y con los enunciados de
    las consultas oficiales que llevan marcador temporal.
    """
    from src.retrieval import recencia as en_dev

    entrega = _cargar_generador()

    nombres = [
        "AIINDEX_ai-index-2024-ch1-research-development.pdf",
        "SWF_global-counterspace-capabilities-2022.pdf",
        "CSET_ai-and-cybersecurity.pdf",
        "informe-2019-actualizado-2021.pdf",
        "doc-1503-anexo.pdf",
        "",
    ]
    for nombre in nombres:
        assert en_dev.anio_de_fuente(nombre) == entrega.anio_de_fuente(nombre), nombre

    enunciados = [
        "¿Qué capacidades operacionales evidencian las maniobras realizadas "
        "recientemente por satélites rusos en órbita GEO?",
        "¿Qué restricciones impone el Derecho Internacional en el Espacio en la "
        "regulación del uso de armas?",
        "¿Qué innovaciones tácticas recientes han incorporado los grupos armados?",
    ]
    for enunciado in enunciados:
        assert en_dev.tiene_marcador_temporal(enunciado) == entrega.tiene_marcador_temporal(
            enunciado
        ), enunciado


def test_el_prior_de_recencia_viene_apagado_en_la_entrega():
    """Se escribio sin poder validarlo contra el indice. Encenderlo por
    descuido cambiaria el ranking de documentos -- o sea el F1@3 -- sin ninguna
    medicion que lo respalde."""
    import inspect

    entrega = _cargar_generador()
    firma = inspect.signature(entrega.build_result_object)
    assert firma.parameters["prior_recencia"].default == 0.0

    # El default del CLI no se puede leer sin ejecutar main(): el parser se
    # construye ahi dentro. Lo cubre la corrida real por subprocess de
    # test_retrieval_schema.py, que corre sin flags y valida el esquema.


def test_el_glosario_expande_igual_en_las_dos_copias():
    """El glosario cambia el VECTOR de la consulta, asi que una tabla que se
    desfase entre dev/src y la entrega produce resultados distintos sin que
    nada falle. Se comparan las 50 consultas oficiales, que es el universo
    real, y no solo la tabla."""
    from src.retrieval.glosario import GLOSARIO as en_dev
    from src.retrieval.glosario import expandir_consulta as expandir_dev

    entrega = _cargar_generador()
    assert en_dev == entrega.GLOSARIO

    consultas = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
    if not consultas.exists():
        pytest.skip(f"no existe {consultas}")
    textos = [
        json.loads(linea)["text"]
        for linea in consultas.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    for texto in textos:
        assert expandir_dev(texto) == entrega.expandir_consulta(texto), texto[:60]


def test_los_defaults_de_recuperacion_son_los_que_se_midieron():
    """k_pool 100 y top5 salieron de scripts/barrido_pool.py. Volverlos a 60 o
    a `sum` cambia la entrega en silencio: con pool 100 las dos estrategias YA
    NO son equivalentes (a 60 si lo eran)."""
    entrega = _cargar_generador()
    assert entrega.DEFAULT_K_POOL == 100


def test_el_desempate_con_grafo_se_comporta_igual_en_las_dos_copias():
    """El grafo no se fusiona como lista (pierde 11-0 medido): entra como
    desempate por evidencia sobre el pool ya recuperado. La copia aplanada y
    `dev/src/` tienen que reordenar identico o la entrega se desfasa."""
    from src.graph.graph_retrieval import GraphHit, desempatar_con_grafo as en_dev

    entrega = _cargar_generador()

    def hit(chunk_id, score):
        return entrega.Hit(rank=1, score=score, chunk_id=chunk_id,
                           doc_id=chunk_id.split("-c")[0], fuente="f.pdf",
                           texto="x", formato="pdf", fenomeno=1, idioma="es")

    casos = [
        (  # scores distintos: el grafo no puede desplazar
            [hit("d1-c0", 0.9), hit("d2-c0", 0.8), hit("d3-c0", 0.7)],
            [GraphHit(rank=1, score=3.0, chunk_id="d3-c0", doc_id="d3")],
        ),
        (  # empate exacto: decide la evidencia
            [hit("d1-c0", 0.5), hit("d2-c0", 0.5)],
            [GraphHit(rank=1, score=5.0, chunk_id="d2-c0", doc_id="d2")],
        ),
        (  # sin evidencia: inerte
            [hit("d1-c0", 0.5), hit("d2-c0", 0.5)],
            [],
        ),
    ]
    for hits, graph_hits in casos:
        dev = en_dev(list(hits), graph_hits)
        plano = entrega.desempatar_con_grafo(list(hits), graph_hits)
        assert [h.chunk_id for h in dev] == [h.chunk_id for h in plano], (
            f"desempate con grafo distinto: dev={[h.chunk_id for h in dev]} "
            f"entrega={[h.chunk_id for h in plano]}"
        )


def test_grafo_activo_por_defecto():
    """El grafo entra en la corrida oficial, no solo con un flag opcional.

    La sec. 7 concede el bonus si el grafo esta INTEGRADO a la recuperacion;
    construirlo y no usarlo no cuenta. Entra como desempate no desplazante
    (`desempatar_con_grafo`), NO por fusion RRF -- la fusion esta medida y
    pierde 11-0.

    Medido el 12 ago 2026 sobre las 50 oficiales: activarlo cambia 0 de 50
    lineas de `resultados.jsonl` y deja F1@3 0.455 / NDCG@10 0.516 /
    penalizado 0.499 intactos. Es integracion del artefacto, no una mejora
    de metrica.
    """
    entrega = _cargar_generador()
    parser = entrega.build_arg_parser()

    assert parser.parse_args(["--consultas", "x.jsonl"]).use_graph is True
    assert parser.parse_args(["--consultas", "x.jsonl", "--sin-grafo"]).use_graph is False
    # `--use-graph` sigue aceptandose para no romper los scripts que lo pasan.
    assert parser.parse_args(["--consultas", "x.jsonl", "--use-graph"]).use_graph is True
