"""Envoltorios sincronos de la fase OFFLINE y ONLINE (secs. 6 y 8) pensados
para correr en un hilo de fondo desde la GUI (`scripts/gui_app.py`), con
callback de progreso por documento/consulta.

No reimplementa logica de negocio: llama a las mismas funciones que ya usan
`scripts/build_corpus_index.py` y `Entrega/generador.py`. Estos dos scripts
CLI siguen funcionando exactamente igual sin cambios -- la GUI es una segunda
forma de invocar el mismo pipeline, no una reimplementacion.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import (
    CORPUS_DIR,
    ENCODER_GTE_NAME,
    ENCODER_PRIMARY_NAME,
    ENCODER_SECONDARY_NAME,
    GRAFO_PATH,
    RESULTADOS_PATH,
)
from ..embedding.build_index import build_and_persist, load_index
from ..embedding.encoders import get_encoder
from ..ingestion.pipeline import iter_corpus_files, process_document, write_chunks_jsonl
from ..retrieval.fusion import rebuild_hits_from_fusion, reciprocal_rank_fusion
from ..retrieval.rerank import rerank_por_segundo_encoder, verificar_alineacion
from ..retrieval.search import search

# Mismos valores que Entrega/generador.py (DEFAULT_RERANK_*): la GUI tiene que
# responder lo mismo que se entrega, no una configuracion parecida.
RERANK_NAMES = (ENCODER_GTE_NAME, ENCODER_SECONDARY_NAME)
RERANK_WEIGHT = 0.25
RERANK_DEPTH = 200

ENTREGA_DIR = Path(__file__).resolve().parents[3] / "Entrega"  # <raiz>/Entrega

ProgressCallback = Optional[Callable[[dict], None]]


@dataclass
class RunSummary:
    """Un resumen = una linea del historial de corridas (src/gui/history.py)."""

    tipo: str  # "offline" | "online"
    encoder: str
    detalle: str  # corpus_dir (offline) o ruta de consultas (online)
    n_items: int  # documentos procesados (offline) o consultas respondidas (online)
    tokens_totales: int
    duracion_seg: float
    estado: str  # "ok" | "error"
    out_path: str
    mensaje_error: str | None = None


def run_offline(
    corpus_dir: Path = CORPUS_DIR,
    encoder_name: str = ENCODER_PRIMARY_NAME,
    use_fake_encoder: bool = False,
    with_graph: bool = True,
    out_dir: Path | None = None,
    graph_out_path: Path = GRAFO_PATH,
    progress_cb: ProgressCallback = None,
) -> RunSummary:
    start = time.time()
    encoder = get_encoder(name=encoder_name, use_fake=use_fake_encoder)
    if progress_cb:
        progress_cb({"tipo": "inicio", "fase": "offline", "encoder": encoder.name})

    records = []
    tokens_totales = 0
    n_documentos = 0
    for path in iter_corpus_files(corpus_dir):
        try:
            recs = process_document(path, count_tokens=encoder.count_tokens, corpus_dir=corpus_dir)
        except Exception as exc:  # noqa: BLE001 -- se reporta a la GUI y se sigue con el resto del corpus
            if progress_cb:
                progress_cb({"tipo": "error_documento", "nombre": path.name, "mensaje": str(exc)})
            continue
        records.extend(recs)
        n_documentos += 1
        tokens_totales += sum(r.num_tokens for r in recs)
        if progress_cb:
            progress_cb(
                {
                    "tipo": "documento",
                    "nombre": path.name,
                    "n_chunks": len(recs),
                    "tokens_acumulados": tokens_totales,
                }
            )

    write_chunks_jsonl(records)
    out_dir_final = build_and_persist(records, encoder, out_dir=out_dir)

    if with_graph:
        from ..graph.build_graph import build_knowledge_graph, export_graphml

        graph = build_knowledge_graph(records)
        export_graphml(graph, graph_out_path)
        if progress_cb:
            progress_cb(
                {
                    "tipo": "grafo",
                    "nodos": graph.number_of_nodes(),
                    "aristas": graph.number_of_edges(),
                }
            )

    return RunSummary(
        tipo="offline",
        encoder=encoder.name,
        detalle=str(corpus_dir),
        n_items=n_documentos,
        tokens_totales=tokens_totales,
        duracion_seg=time.time() - start,
        estado="ok",
        out_path=str(out_dir_final),
    )


def _load_generador_module():
    """`Entrega/generador.py` es un script (sin __init__.py), pero define
    funciones puras a nivel de modulo (`load_consultas`, `build_result_object`)
    que se pueden importar directamente sin ejecutar su CLI -- se reutilizan
    aqui en vez de duplicar el esquema de salida de la seccion 9."""
    sys.path.insert(0, str(ENTREGA_DIR))
    import generador  # noqa: PLC0415

    return generador


def _answer_one(
    query_text: str,
    query_id: str,
    encoder,
    index,
    metadata: list[dict],
    metadata_by_chunk_id: dict,
    graph,
    generador,
    k_pool: int,
    reranks: list = (),
) -> tuple[dict, int, float]:
    """Responde una sola consulta -- misma logica que usa `run_online()` en
    lote y `Session.ask()` en modo interactivo (chat), para no duplicar la
    fusion con el grafo ni el esquema de salida en dos lugares distintos.

    Devuelve tambien `best_score` (similitud coseno del mejor fragmento antes
    de la fusion con el grafo): el sistema es un motor de RECUPERACION, no un
    chatbot -- SIEMPRE devuelve los top-k mas parecidos, incluso si la
    consulta no tiene relacion con el corpus (sec. 8.3 prohibe cualquier
    modelo generativo que pudiera responder "no se"). `best_score` es la
    unica senal disponible para distinguir una respuesta bien fundamentada de
    una donde no habia nada realmente parecido."""
    tokens = encoder.count_tokens(query_text)
    # Mismo glosario bilingue que usa la entrega, y aplicado en el MISMO punto:
    # antes de vectorizar y para todos los encoders. Se hace aca, en el unico
    # sitio por el que pasan tanto el lote como el chat, para que la GUI no
    # pueda desfasarse de resultados.jsonl. Una consulta sin terminos del
    # glosario se devuelve identica.
    query_text = generador.expandir_consulta(query_text)
    k_busqueda = max(k_pool, RERANK_DEPTH) if reranks else k_pool
    hits = search(query_text, encoder, index, metadata, k=k_busqueda)
    best_score = hits[0].score if hits else 0.0

    # Cascada triple, igual que generador.py: el recorte a RERANK_DEPTH se hace
    # UNA sola vez, antes del primer re-puntuador (ver punto 0 de CLAUDE.md).
    if reranks:
        hits = hits[:RERANK_DEPTH]
        for enc_r, idx_r in reranks:
            hits = rerank_por_segundo_encoder(
                hits, idx_r, enc_r.encode_query(query_text), peso=RERANK_WEIGHT
            )

    if graph is not None:
        from ..graph.graph_retrieval import graph_search

        query_lang = hits[0].idioma if hits else None
        graph_hits = graph_search(query_text, graph, lang=query_lang, k=k_pool)
        if graph_hits:
            fused = reciprocal_rank_fusion([hits, graph_hits], key=lambda h: h.chunk_id)
            # Se reutiliza rebuild_hits_from_fusion en vez de rehacerlo aqui:
            # es la misma reconstruccion que hace generador.py, y tenerla dos
            # veces garantiza que tarde o temprano diverjan.
            hits = rebuild_hits_from_fusion(fused, metadata_by_chunk_id, limit=k_pool)

    # La agregacion a documento usa siempre k_pool: la profundidad extra de la
    # cascada sirve para REORDENAR, no para ampliar el pool.
    hits = hits[:k_pool]
    return generador.build_result_object(query_id, hits), tokens, best_score


def run_online(
    consultas_path: Path,
    encoder_name: str = ENCODER_PRIMARY_NAME,
    use_fake_encoder: bool = False,
    use_graph: bool = False,
    index_dir: Path | None = None,
    graph_path: Path = GRAFO_PATH,
    out_path: Path = RESULTADOS_PATH,
    k_pool: int = 100,
    progress_cb: ProgressCallback = None,
) -> RunSummary:
    generador = _load_generador_module()

    start = time.time()
    encoder = get_encoder(name=encoder_name, use_fake=use_fake_encoder)
    if progress_cb:
        progress_cb({"tipo": "inicio", "fase": "online", "encoder": encoder.name})

    index, metadata = load_index(encoder.name, index_dir=index_dir)
    metadata_by_chunk_id = {m["chunk_id"]: m for m in metadata}

    graph = None
    if use_graph:
        import networkx as nx

        graph = nx.read_graphml(graph_path)

    consultas = generador.load_consultas(consultas_path)
    resultados = []
    tokens_totales = 0

    for consulta in consultas:
        resultado, tokens, _best_score = _answer_one(
            consulta["text"], consulta["query_id"], encoder, index, metadata,
            metadata_by_chunk_id, graph, generador, k_pool,
        )
        tokens_totales += tokens
        resultados.append(resultado)
        if progress_cb:
            progress_cb(
                {
                    "tipo": "consulta",
                    "query_id": consulta["query_id"],
                    "tokens_acumulados": tokens_totales,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return RunSummary(
        tipo="online",
        encoder=encoder.name,
        detalle=str(consultas_path),
        n_items=len(consultas),
        tokens_totales=tokens_totales,
        duracion_seg=time.time() - start,
        estado="ok",
        out_path=str(out_path),
    )


class Session:
    """Sesion de chat interactiva: carga el encoder + indice + grafo UNA sola
    vez (son los pasos lentos) y responde consultas sueltas sobre eso, para
    `scripts/gui_app.py`. Reusa `_answer_one()` -- la misma logica de fusion y
    esquema de salida que usa `run_online()` en lote."""

    def __init__(
        self,
        encoder_name: str = ENCODER_PRIMARY_NAME,
        use_fake_encoder: bool = False,
        use_graph: bool = False,
        index_dir: Path | None = None,
        graph_path: Path = GRAFO_PATH,
        k_pool: int = 100,
        rerank_names: tuple = RERANK_NAMES,
        progress_cb: ProgressCallback = None,
    ):
        self.generador = _load_generador_module()
        self.encoder = get_encoder(name=encoder_name, use_fake=use_fake_encoder)
        self.index, self.metadata = load_index(self.encoder.name, index_dir=index_dir)
        self.metadata_by_chunk_id = {m["chunk_id"]: m for m in self.metadata}

        # Los dos re-puntuadores de la cascada entregada. Si alguno no se puede
        # cargar (gte baja ~1,2 GB de HuggingFace) se sigue sin el y se avisa:
        # aca degradar es aceptable porque la GUI es exploratoria, a diferencia
        # de generador.py, que aborta para no entregar algo distinto a lo medido.
        self.reranks: list = []
        for nombre_r in rerank_names if not use_fake_encoder else ():
            try:
                enc_r = get_encoder(name=nombre_r)
                idx_r, meta_r = load_index(enc_r.name)
                verificar_alineacion(self.metadata, meta_r)
            except Exception as exc:  # noqa: BLE001
                if progress_cb:
                    progress_cb({"tipo": "encoder_omitido", "nombre": nombre_r, "mensaje": str(exc)})
                continue
            self.reranks.append((enc_r, idx_r))
            if progress_cb:
                progress_cb({"tipo": "encoder_listo", "nombre": enc_r.name})

        self.graph = None
        if use_graph:
            import networkx as nx

            self.graph = nx.read_graphml(graph_path)
        self.k_pool = k_pool
        self.tokens_totales = 0

    def ask(self, query_text: str, query_id: str = "chat") -> tuple[dict, int, float, float]:
        """Devuelve (resultado, tokens, tiempo_seg, best_score). `best_score`
        es la similitud coseno del mejor fragmento -- ver docstring de
        `_answer_one()`: util para que la GUI avise cuando la consulta no
        tiene relacion real con el corpus, en vez de mostrar la lista de
        top-k como si fuera una respuesta confiable."""
        start = time.time()
        resultado, tokens, best_score = _answer_one(
            query_text, query_id, self.encoder, self.index, self.metadata,
            self.metadata_by_chunk_id, self.graph, self.generador, self.k_pool,
            self.reranks,
        )
        self.tokens_totales += tokens
        return resultado, tokens, time.time() - start, best_score
