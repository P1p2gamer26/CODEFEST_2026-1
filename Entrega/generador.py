#!/usr/bin/env python3
"""generador.py -- script de recuperacion ONLINE (sec. 6, sec. 8, sec. 9).

Lee el indice FAISS + metadata ya construidos (fase OFFLINE) y el archivo de
consultas, y genera `resultados.jsonl` con exactamente el esquema exigido en
la seccion 9: por cada consulta, los 3 documentos y los 10 fragmentos (<=250
palabras cada uno) mas relevantes.

Ninguna etapa de este script usa un modelo generativo/decoder: solo el
encoder (mismo que en la indexacion), FAISS, y aritmetica sobre
puntuaciones/metadata (sec. 8.3).

ESTE ARCHIVO ES AUTOCONTENIDO A PROPOSITO. No importa nada del repositorio:
la carpeta `Entrega/` debe poder copiarse sola a cualquier maquina y correr.
Es una version aplanada del camino online que en el repositorio de desarrollo
vive en `dev/src/` (config, embedding, retrieval, chunking, graph); cada
seccion de abajo indica su modulo de origen. Al modificar cualquiera de esos
modulos hay que re-aplanar aqui y correr
`dev/tests/test_entrega_standalone.py`.

Rutas: todo lo que este script necesita se resuelve RELATIVO A ESTE ARCHIVO
(`base_vectorial/`, `resultados.jsonl`), asi que la carpeta es reubicable.

Uso:
    python generador.py --consultas consultas.jsonl
    python generador.py --consultas consultas.jsonl --out resultados.jsonl
    python generador.py --consultas consultas.jsonl --use-graph
    python generador.py --consultas consultas.jsonl --use-fake-encoder  # sin red, solo pruebas

    # Multi-encoder (sec. 4.4): busca en ambos indices y fusiona con RRF (sec. 8.4).
    python generador.py --consultas consultas.jsonl \
        --encoder-name paraphrase-multilingual-MiniLM-L12-v2 multilingual-e5-base

Dependencias: faiss-cpu, numpy, sentence-transformers. Solo con `--use-graph`
hacen falta ademas networkx y spacy (se importan perezosamente).

Formato esperado del archivo de consultas (PROVISIONAL -- ver
`load_consultas()`; el formato oficial de q001-q050 aun no lo entrega ADL):
JSON Lines, un objeto por linea, con los campos `query_id` (o `id`) y `text`
(o `query`/`consulta`).
"""

import argparse
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import faiss
import numpy as np

logger = logging.getLogger("generador")

# ---------------------------------------------------------------------------
# Configuracion -- de src/config.py (solo lo que usa el camino ONLINE)
# ---------------------------------------------------------------------------

ENTREGA_DIR = Path(__file__).resolve().parent
BASE_VECTORIAL_DIR = ENTREGA_DIR / "base_vectorial"
RESULTADOS_PATH = ENTREGA_DIR / "resultados.jsonl"
GRAFO_PATH = BASE_VECTORIAL_DIR / "grafo" / "grafo.graphml"

# --- Encoders (HuggingFace, arquitectura encoder, sin modelos generativos) ---
ENCODER_PRIMARY_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_PRIMARY_HF_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_SECONDARY_NAME = "multilingual-e5-base"
ENCODER_SECONDARY_HF_ID = "intfloat/multilingual-e5-base"

# La familia E5 exige estos prefijos: sin ellos la calidad cae de forma
# silenciosa (el modelo fue entrenado siempre con ellos). Consulta y pasaje
# llevan prefijos DISTINTOS -- por eso `Encoder` expone codificacion asimetrica.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# --- Formato de salida (resultados.jsonl) ---
MAX_FRAGMENT_WORDS = 250
N_DOCUMENTS_PER_QUERY = 3
N_FRAGMENTS_PER_QUERY = 10

# --- Recuperacion ---
OVERFETCH_FACTOR = 4  # sobre-recuperar antes de post-filtros/agregacion
RRF_K0 = 60

# --- Segmentacion de oraciones / NER (grafo) ---
SPACY_MODEL_BY_LANG = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "pt": "pt_core_news_sm",
}


def encoder_dir(encoder_name: str) -> Path:
    return BASE_VECTORIAL_DIR / f"encoder_{encoder_name}"


# ---------------------------------------------------------------------------
# Encoders -- de src/embedding/encoders.py
# ---------------------------------------------------------------------------
# `count_tokens()` no se replica aqui: solo lo usa el chunking (fase offline).


class Encoder(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Devuelve una matriz (n, dim) float32 normalizada a norma unitaria,
        para que el producto interno (IndexFlatIP) equivalga a similitud
        coseno (sec. 8.2)."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # --- Codificacion asimetrica (consulta vs. pasaje) ---
    # Algunos encoders (familia E5) se entrenaron con prefijos DISTINTOS para
    # la consulta y para el documento indexado, y omitirlos degrada la calidad
    # de forma silenciosa. Los encoders que no los necesitan heredan estas
    # implementaciones por defecto.

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Codifica fragmentos para INDEXAR (solo fase offline)."""
        return self.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Codifica una consulta para BUSCAR (`search()`)."""
        return self.encode_one(text)


class SentenceTransformerEncoder(Encoder):
    """Encoder real de produccion. Requiere descargar los pesos desde
    huggingface.co la primera vez; despues funciona desde la cache local."""

    def __init__(
        self,
        hf_id: str = ENCODER_PRIMARY_HF_ID,
        name: str = ENCODER_PRIMARY_NAME,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(hf_id)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode([self.passage_prefix + t for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_one(self.query_prefix + text)


class HashingFakeEncoder(Encoder):
    """Encoder determinista por hashing, SIN modelo de lenguaje real.

    Uso EXCLUSIVO para pruebas sin red. Los vectores son reproducibles pero NO
    capturan significado semantico: validan la mecanica (formato de salida,
    agregacion, fusion), NO la calidad de la recuperacion. Nunca debe usarse
    para generar los resultados de la entrega.
    """

    def __init__(self, dim: int = 384, name: str = "hashing-fake-encoder"):
        self.dim = dim
        self.name = name

    def _hash_vector(self, text: str) -> np.ndarray:
        # El nombre entra en la semilla para que dos encoders falsos distintos
        # produzcan rankings distintos (necesario para probar la fusion RRF
        # multi-encoder sin red). Sigue siendo determinista.
        seed_material = f"{self.name}\x00{text}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_vector(t) for t in texts]).astype("float32")


# Encoders conocidos: nombre corto (el que se usa en la carpeta de entrega
# `encoder_<nombre>/` y en la CLI) -> como instanciarlo.
KNOWN_ENCODERS: dict[str, dict] = {
    ENCODER_PRIMARY_NAME: {
        "hf_id": ENCODER_PRIMARY_HF_ID,
        "query_prefix": "",
        "passage_prefix": "",
    },
    ENCODER_SECONDARY_NAME: {
        "hf_id": ENCODER_SECONDARY_HF_ID,
        "query_prefix": E5_QUERY_PREFIX,
        "passage_prefix": E5_PASSAGE_PREFIX,
    },
}


@lru_cache(maxsize=None)
def get_encoder(name: str = ENCODER_PRIMARY_NAME, use_fake: bool = False) -> Encoder:
    if use_fake:
        return HashingFakeEncoder(name=name)

    spec = KNOWN_ENCODERS.get(name)
    if spec is None:
        raise ValueError(
            f"encoder desconocido: {name!r}. Conocidos: {sorted(KNOWN_ENCODERS)}. "
            f"Agregar una entrada en KNOWN_ENCODERS."
        )
    return SentenceTransformerEncoder(
        hf_id=spec["hf_id"],
        name=name,
        query_prefix=spec["query_prefix"],
        passage_prefix=spec["passage_prefix"],
    )


# ---------------------------------------------------------------------------
# Carga del indice -- de src/embedding/build_index.py
# ---------------------------------------------------------------------------


def load_index(encoder_name: str, index_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    """FAISS solo guarda vectores + un id interno entero: `metadata.jsonl`
    debe tener una linea por vector, en el mismo orden en que se insertaron
    (sec. 5.3). Si los tamanos no cuadran, el mapeo id->metadata seria
    silenciosamente incorrecto, asi que se aborta."""
    index_dir = index_dir or encoder_dir(encoder_name)

    index = faiss.read_index(str(index_dir / "index.faiss"))

    metadata: list[dict] = []
    with (index_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    if index.ntotal != len(metadata):
        raise ValueError(
            f"indice desalineado al cargar: index.ntotal={index.ntotal} vs "
            f"lineas de metadata={len(metadata)} ({index_dir})"
        )

    return index, metadata


# ---------------------------------------------------------------------------
# Busqueda -- de src/retrieval/search.py
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    fuente: str
    texto: str
    formato: str
    fenomeno: int | None
    idioma: str | None
    # Fila del chunk en el indice FAISS. La usa la cascada de dos encoders para
    # leer el vector del MISMO chunk en el otro indice sin recodificar el
    # pasaje. Los hits que no vienen de FAISS (grafo, fusion) no tienen fila.
    fila: int = -1


def search(
    query: str,
    encoder: Encoder,
    index: faiss.Index,
    metadata: list[dict],
    k: int = 10,
    fenomeno: int | None = None,
    formato: str | None = None,
    idioma: str | None = None,
    min_score: float | None = None,
    overfetch_factor: int = OVERFETCH_FACTOR,
) -> list[Hit]:
    """Busca los `k` fragmentos mas relevantes para `query`.

    Post-filtros (sec. 8.7) operan directamente sobre metadata (`fenomeno`,
    `formato`, `idioma`) o sobre el score de similitud coseno (`min_score`),
    nunca via un modelo generativo. Se sobre-recupera `k * overfetch_factor`
    candidatos de FAISS antes de filtrar, ya que los filtros pueden descartar
    algunos de los top-k crudos.
    """
    # encode_query (no encode_one) para que los encoders que lo requieran
    # apliquen su prefijo de consulta.
    query_vec = encoder.encode_query(query).reshape(1, -1)
    fetch_k = min(index.ntotal, max(k * overfetch_factor, k))
    scores, ids = index.search(query_vec, fetch_k)

    hits: list[Hit] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        if fenomeno is not None and meta.get("fenomeno") != fenomeno:
            continue
        if formato is not None and meta.get("formato") != formato:
            continue
        if idioma is not None and meta.get("idioma") != idioma:
            continue
        if min_score is not None and score < min_score:
            continue

        hits.append(
            Hit(
                rank=0,  # se asigna abajo, tras aplicar todos los filtros
                score=float(score),
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                fuente=meta["fuente"],
                texto=meta["texto"],
                formato=meta["formato"],
                fenomeno=meta.get("fenomeno"),
                idioma=meta.get("idioma"),
                fila=int(idx),
            )
        )
        if len(hits) >= k:
            break

    for i, hit in enumerate(hits, start=1):
        hit.rank = i
    return hits


# ---------------------------------------------------------------------------
# Cascada de dos encoders -- de src/retrieval/rerank.py
# ---------------------------------------------------------------------------


def rerank_por_segundo_encoder(
    hits: list[Hit],
    index_secundario: faiss.Index,
    vector_consulta: np.ndarray,
    peso: float,
) -> list[Hit]:
    """Reordena `hits` mezclando su score con el del encoder secundario:

        score = cos_primario + peso * cos_secundario

    El primario genera los candidatos (conserva su recall) y el secundario solo
    los re-puntua. Se hace asi, y no fusionando las dos listas con RRF, porque
    RRF premia el acuerdo entre listas y los dos encoders casi no coinciden
    (11,3% de los documentos del top-3): fusionarlos intercala la lista buena
    con la mala. Ver la seccion 3.1 del informe tecnico.

    Los dos terminos son cosenos, misma escala, asi que sumarlos es legitimo.
    El vector del chunk se lee del otro indice por su FILA, sin recodificar el
    pasaje: los dos indices se construyen sobre los mismos records en el mismo
    orden. Los hits sin fila (grafo, fusion) conservan su score primario.
    """
    if not hits:
        return []

    consulta = np.asarray(vector_consulta, dtype="float32").reshape(-1)

    puntuados: list[tuple[float, Hit]] = []
    for hit in hits:
        score = hit.score
        if peso and hit.fila >= 0:
            vector = index_secundario.reconstruct(hit.fila)
            score += peso * float(np.dot(consulta, vector))
        puntuados.append((score, hit))

    # sorted es estable: ante empates se conserva el orden del primario.
    puntuados.sort(key=lambda par: par[0], reverse=True)

    return [
        Hit(
            rank=rank,
            score=score,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            fuente=hit.fuente,
            texto=hit.texto,
            formato=hit.formato,
            fenomeno=hit.fenomeno,
            idioma=hit.idioma,
            fila=hit.fila,
        )
        for rank, (score, hit) in enumerate(puntuados, start=1)
    ]


def verificar_alineacion(metadata_a: list[dict], metadata_b: list[dict]) -> None:
    """Aborta si los dos indices no describen los mismos chunks en el mismo
    orden: la cascada re-puntuaria el chunk equivocado en silencio."""
    if len(metadata_a) != len(metadata_b):
        raise ValueError(
            f"los indices tienen distinto numero de chunks: {len(metadata_a)} vs {len(metadata_b)}"
        )
    for i, (a, b) in enumerate(zip(metadata_a, metadata_b)):
        if a["chunk_id"] != b["chunk_id"]:
            raise ValueError(
                f"los indices divergen en la fila {i}: {a['chunk_id']!r} vs {b['chunk_id']!r}"
            )


# ---------------------------------------------------------------------------
# Agregacion a nivel documento -- de src/retrieval/aggregate.py
# ---------------------------------------------------------------------------


@dataclass
class DocumentHit:
    rank: int
    doc_id: str
    score: float


def aggregate_documents(
    hits: list[Hit], top_n: int = N_DOCUMENTS_PER_QUERY, strategy: str = "sum"
) -> list[DocumentHit]:
    """Agrega las puntuaciones de los fragmentos por documento (sec. 8.6).
    Solo aritmetica sobre los scores de FAISS.

    Por defecto "sum", no "max": un documento relevante suele tener VARIOS
    pasajes relevantes, mientras que "max" premia al que tiene un unico chunk
    afortunado. La eleccion se apoya en ese argumento, no en la medicion: sobre
    el ground truth propio sum promedia mas que max, pero contando por consulta
    el reparto es 16-8 con 17 empates (prueba de signos p=0.15), o sea que la
    ventaja no alcanza significancia con esa muestra."""
    scores_by_doc: dict[str, list[float]] = defaultdict(list)
    for hit in hits:
        scores_by_doc[hit.doc_id].append(hit.score)

    if strategy == "max":
        agg_scores = {doc_id: max(scores) for doc_id, scores in scores_by_doc.items()}
    elif strategy == "sum":
        agg_scores = {doc_id: sum(scores) for doc_id, scores in scores_by_doc.items()}
    elif strategy == "mean":
        agg_scores = {doc_id: sum(scores) / len(scores) for doc_id, scores in scores_by_doc.items()}
    elif strategy.startswith("top") and strategy[3:].isdigit():
        # Suma solo los M mejores fragmentos del documento: "sum" no tiene tope
        # y un documento con muchos fragmentos mediocres desplaza a uno con un
        # solo fragmento excelente. Medido y no adoptado (ver informe, sec. 7);
        # se mantiene aqui para que esta copia no diverja de src/retrieval/.
        m = int(strategy[3:])
        agg_scores = {
            doc_id: sum(sorted(scores, reverse=True)[:m]) for doc_id, scores in scores_by_doc.items()
        }
    else:
        raise ValueError(f"estrategia de agregacion desconocida: {strategy}")

    ranked = sorted(agg_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        DocumentHit(rank=i, doc_id=doc_id, score=score)
        for i, (doc_id, score) in enumerate(ranked, start=1)
    ]


# ---------------------------------------------------------------------------
# Segmentacion de oraciones -- de src/chunking/sentence_split.py
# ---------------------------------------------------------------------------
# `pysbd` (reglas linguisticas dedicadas) para es/en, que es donde tiene
# soporte nativo; spaCy (parser entrenado) para portugues, que pysbd no cubre;
# y un sentencizer generico de spaCy como ultimo recurso. Se importan
# perezosamente: solo hacen falta si algun chunk recuperado supera las 250
# palabras.

PYSBD_SUPPORTED_LANGS = {"en", "es"}


@lru_cache(maxsize=None)
def _get_pysbd_segmenter(lang: str):
    import pysbd

    return pysbd.Segmenter(language=lang, clean=False)


@lru_cache(maxsize=None)
def _get_spacy_trained_sentencizer(model_name: str):
    import spacy

    return spacy.load(
        model_name,
        disable=["tagger", "ner", "lemmatizer", "attribute_ruler", "morphologizer"],
    )


@lru_cache(maxsize=None)
def _get_spacy_blank_sentencizer(lang: str):
    import spacy

    try:
        nlp = spacy.blank(lang)
    except Exception:
        nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str, lang: str | None) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if lang in PYSBD_SUPPORTED_LANGS:
        sentences = _get_pysbd_segmenter(lang).segment(text)
    elif lang and lang in SPACY_MODEL_BY_LANG:
        nlp = _get_spacy_trained_sentencizer(SPACY_MODEL_BY_LANG[lang])
        sentences = [s.text for s in nlp(text).sents]
    else:
        nlp = _get_spacy_blank_sentencizer(lang or "xx")
        sentences = [s.text for s in nlp(text).sents]

    return [s.strip() for s in sentences if s and s.strip()]


# ---------------------------------------------------------------------------
# Limite de 250 palabras -- de src/retrieval/truncate.py
# ---------------------------------------------------------------------------
# Si un chunk recuperado supera el limite (sec. 9.2.1) se divide en
# sub-fragmentos que respetan limites oracionales completos (nunca se corta
# una oracion) y comparten el mismo `chunk_id` de origen, cada uno con su
# propio `rank` en la lista final (sec. 9.2.1 y 9.3.1).


def _split_oversized(hit: Hit, max_words: int) -> list[dict]:
    sentences = split_sentences(hit.texto, hit.idioma)
    if not sentences:
        sentences = [hit.texto]

    # Una sola "oracion" puede superar por si sola el limite: pasa con texto
    # de OCR o de PDF mal extraido, donde no queda puntuacion aprovechable y
    # el splitter devuelve un bloque entero. Ahi el corte por palabras es la
    # unica salida; sin esto el fragmento sale con mas de 250 palabras y la
    # sec. 9.3.2 lo penaliza o lo descarta.
    trozos: list[str] = []
    for sent in sentences:
        palabras = sent.split()
        if len(palabras) > max_words:
            trozos += [
                " ".join(palabras[i : i + max_words])
                for i in range(0, len(palabras), max_words)
            ]
        else:
            trozos.append(sent)
    sentences = trozos

    sub_fragments: list[dict] = []
    current: list[str] = []
    current_words = 0
    for sent in sentences:
        sent_words = len(sent.split())
        if current and current_words + sent_words > max_words:
            sub_fragments.append(
                {"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": " ".join(current)}
            )
            current, current_words = [], 0
        current.append(sent)
        current_words += sent_words

    if current:
        sub_fragments.append(
            {"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": " ".join(current)}
        )
    return sub_fragments


def _clave_dedup(texto: str) -> str:
    """Normaliza solo espacios y mayusculas: la comparacion sigue siendo de
    texto exacto. Deliberadamente NO se hace similitud difusa, que podria
    descartar fragmentos distintos que comparten un parrafo."""
    return " ".join(texto.split()).lower()


def enforce_word_limit(
    hits: list[Hit],
    max_fragments: int = N_FRAGMENTS_PER_QUERY,
    max_words: int = MAX_FRAGMENT_WORDS,
) -> list[dict]:
    """Los 10 fragmentos de la consulta, sin repetir texto.

    El corpus de ADL trae documentos duplicados (el mismo informe de CSIS bajo
    dos doc_id, series que reeditan capitulos enteros), asi que dos chunks
    distintos pueden tener texto identico. Entregar el mismo texto dos veces
    no puede sumar en NDCG@10 -- el ranking ideal no lo contiene -- y ademas
    desplaza fuera del top-10 a un candidato que si podria sumar. Medido sobre
    las 50 consultas oficiales: 17 de 500 fragmentos eran duplicados exactos."""
    fragments: list[dict] = []
    vistos: set[str] = set()

    for hit in hits:
        if len(fragments) >= max_fragments:
            break

        n_words = len(hit.texto.split())
        if n_words <= max_words:
            candidatos = [{"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "text": hit.texto}]
        else:
            candidatos = _split_oversized(hit, max_words)

        for sub in candidatos:
            if len(fragments) >= max_fragments:
                break
            clave = _clave_dedup(sub["text"])
            if clave in vistos:
                continue
            vistos.add(clave)
            fragments.append(sub)

    fragments = fragments[:max_fragments]
    for i, frag in enumerate(fragments, start=1):
        frag["rank"] = i
    return fragments


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion -- de src/retrieval/fusion.py
# ---------------------------------------------------------------------------
# RRF (sec. 8.4, ecuacion 7) combina varias listas ya ordenadas (multiples
# encoders, o vector+grafo con el grafo tratado como un "indice" adicional,
# sec. 8.5) usando solo la POSICION de cada item en cada lista: es robusto a
# que dos encoders produzcan puntuaciones en escalas distintas, que es la
# razon por la que se elige sobre CombSUM/CombMNZ.
#
# INVARIANTE: fusionar por `chunk_id` solo es correcto si todos los indices
# comparten los mismos chunks. Se valida en `main()` antes de buscar.

T = TypeVar("T")


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[T]],
    key: Callable[[T], str],
    k0: int = RRF_K0,
) -> list[tuple[T, float]]:
    """Cada lista en `ranked_lists` debe venir ya ordenada de mayor a menor
    relevancia (rank 1 = primer elemento). `key` extrae el identificador
    (p. ej. chunk_id) usado para reconocer el mismo item entre listas.

    Devuelve pares (item, score_rrf) ordenados de mayor a menor score_rrf; el
    item devuelto es su primera aparicion entre las listas fusionadas.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    representative: dict[str, T] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_key = key(item)
            rrf_scores[item_key] += 1.0 / (k0 + rank)
            representative.setdefault(item_key, item)

    ordered_keys = sorted(rrf_scores, key=lambda k_: rrf_scores[k_], reverse=True)
    return [(representative[k_], rrf_scores[k_]) for k_ in ordered_keys]


def rebuild_hits_from_fusion(
    fused: Sequence[tuple[object, float]],
    metadata_by_chunk_id: dict[str, dict],
    limit: int,
) -> list[Hit]:
    """Reconstruye `Hit`s a partir del resultado de `reciprocal_rank_fusion`.

    Los items fusionados pueden venir de fuentes distintas (otro encoder, el
    grafo), asi que el texto y la metadata se releen SIEMPRE desde
    `metadata_by_chunk_id` en vez de confiar en el item representativo: es lo
    que garantiza que el texto devuelto corresponda de verdad al chunk_id
    reportado. Los items sin metadata conocida se descartan.
    """
    hits: list[Hit] = []
    for item, score in fused:
        if len(hits) >= limit:
            break
        meta = metadata_by_chunk_id.get(getattr(item, "chunk_id", None))
        if meta is None:
            continue
        hits.append(
            Hit(
                rank=len(hits) + 1,
                score=score,
                chunk_id=meta["chunk_id"],
                doc_id=meta["doc_id"],
                fuente=meta["fuente"],
                texto=meta["texto"],
                formato=meta["formato"],
                fenomeno=meta.get("fenomeno"),
                idioma=meta.get("idioma"),
            )
        )
    return hits


# ---------------------------------------------------------------------------
# Grafo de conocimiento (bonus) -- de src/graph/ner.py y graph_retrieval.py
# ---------------------------------------------------------------------------
# NER sobre la consulta -> nodos coincidentes -> vecinos de primer orden ->
# pool de candidatos rankeado por evidencia, que se fusiona con FAISS via RRF
# tratando el grafo como un indice adicional (sec. 8.5).
#
# El NER usa el componente ya entrenado de los modelos spaCy es/en/pt (no
# generativo, MIT). `spacy` y `networkx` se importan perezosamente: sin
# `--use-graph` no hacen falta.

NER_DEFAULT_LANG = "es"

# Los modelos es/pt_core_news_sm usan el esquema CoNLL (LOC, MISC, ORG, PER);
# en_core_web_sm usa OntoNotes, mucho mas granular. Se excluyen las etiquetas
# de OntoNotes que no describen una entidad nombrada propiamente dicha.
NER_EXCLUDED_LABELS = {"CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}


@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class GraphHit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str


@lru_cache(maxsize=None)
def _get_ner_pipeline(lang: str):
    import spacy

    model_name = SPACY_MODEL_BY_LANG.get(lang, SPACY_MODEL_BY_LANG[NER_DEFAULT_LANG])
    return spacy.load(model_name)


def extract_entities(text: str, lang: str | None) -> list[Entity]:
    if not text or not text.strip():
        return []
    nlp = _get_ner_pipeline(lang or NER_DEFAULT_LANG)
    doc = nlp(text)
    return [
        Entity(
            text=ent.text.strip(),
            label=ent.label_,
            start_char=ent.start_char,
            end_char=ent.end_char,
        )
        for ent in doc.ents
        if ent.text.strip() and ent.label_ not in NER_EXCLUDED_LABELS
    ]


def _normalize(text: str) -> str:
    return text.strip().lower()


def graph_search(query: str, graph, lang: str | None = None, k: int = 10) -> list[GraphHit]:
    query_entities = {_normalize(e.text) for e in extract_entities(query, lang)}
    if not query_entities:
        return []

    node_by_normalized = {_normalize(n): n for n in graph.nodes}
    matched_nodes = [node_by_normalized[e] for e in query_entities if e in node_by_normalized]
    if not matched_nodes:
        return []

    evidence_count: dict[tuple[str, str], int] = defaultdict(int)
    for node in matched_nodes:
        for _, _, data in graph.out_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1
        for _, _, data in graph.in_edges(node, data=True):
            evidence_count[(data["doc_id"], data["chunk_id"])] += 1

    ranked = sorted(evidence_count.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        GraphHit(rank=i, score=float(count), doc_id=doc_id, chunk_id=chunk_id)
        for i, ((doc_id, chunk_id), count) in enumerate(ranked, start=1)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_K_POOL = 60  # candidatos usados para agregar a nivel documento (sec. 8.6);
# mayor que los 10 fragmentos que se devuelven, para que la relevancia de un
# documento no dependa solo de si su mejor chunk entro en el top-10 mostrado.

# Cascada de dos encoders. Ambos valores estan MEDIDOS, no supuestos, con
# scripts/barrido_dos_encoders.py sobre el mini ground truth (ver informe,
# sec. 7). Peso 0,25: los pesos mayores rinden mas en promedio sobre las 41
# consultas anotadas pero empiezan a perder en las 10 de anotacion
# independiente, que es la senal clasica de sobreajuste al pooling. Con 0,25 la
# cascada no empeora NINGUNA consulta de ninguna de las dos muestras.
DEFAULT_RERANK_WEIGHT = 0.25
DEFAULT_RERANK_DEPTH = 200

# La cascada viene ACTIVADA por defecto porque es la configuracion con la que
# se genero resultados.jsonl: correr este script sin flags tiene que reproducir
# la entrega (punto 4 de la sec. 1.4), no una variante parecida. Se apaga con
# --rerank-encoder none, y se apaga sola con --use-fake-encoder.
DEFAULT_RERANK_ENCODER = ENCODER_SECONDARY_NAME


def load_consultas(path: Path) -> list[dict]:
    consultas = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query_id = obj.get("query_id") or obj.get("id")
            text = obj.get("text") or obj.get("query") or obj.get("consulta")
            if query_id is None or text is None:
                raise ValueError(
                    f"{path}:{line_number}: se esperaban los campos "
                    f"'query_id'/'id' y 'text'/'query'/'consulta'; linea: {obj}"
                )
            consultas.append({"query_id": str(query_id), "text": str(text)})
    return consultas


def build_result_object(
    query_id: str,
    hits: list[Hit],
    top_docs: int = N_DOCUMENTS_PER_QUERY,
    max_fragments: int = N_FRAGMENTS_PER_QUERY,
    max_words: int = MAX_FRAGMENT_WORDS,
    # "sum" es la configuracion entregada, asi que tiene que ser el default:
    # src/gui/runner.py llama a esta funcion sin pasar la estrategia y con el
    # default anterior ("max") la GUI mostraba documentos distintos a los de
    # resultados.jsonl. El CLI siempre la pasa explicita, por eso no se veia.
    agg_strategy: str = "sum",
) -> dict:
    doc_hits = aggregate_documents(hits, top_n=top_docs, strategy=agg_strategy)
    fragments = enforce_word_limit(hits, max_fragments=max_fragments, max_words=max_words)

    return {
        "query_id": query_id,
        "documents": [{"rank": d.rank, "doc_id": d.doc_id} for d in doc_hits],
        "fragments": [
            {
                "rank": f["rank"],
                "chunk_id": f["chunk_id"],
                "doc_id": f["doc_id"],
                "text": f["text"],
            }
            for f in fragments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--consultas", type=Path, required=True)
    parser.add_argument(
        "--encoder-name",
        nargs="+",
        default=[ENCODER_PRIMARY_NAME],
        help="Uno o varios encoders (sec. 4.4). Con varios, se busca en el indice de "
        "cada uno y los rankings se fusionan con RRF (sec. 8.4). Los indices deben "
        "haberse construido en la misma corrida de la fase offline.",
    )
    parser.add_argument(
        "--use-fake-encoder",
        action="store_true",
        help="Usa el encoder determinista sin red (solo pruebas de mecanica, no de calidad de recuperacion).",
    )
    parser.add_argument(
        "--index-dir", type=Path, default=None, help="Por defecto base_vectorial/encoder_<nombre>"
    )
    parser.add_argument(
        "--index-base",
        type=Path,
        default=None,
        help="Carpeta base que contiene las subcarpetas encoder_<nombre>/. Por defecto "
        "base_vectorial/ junto a este script.",
    )
    parser.add_argument(
        "--rerank-encoder",
        default=DEFAULT_RERANK_ENCODER,
        help="Segundo encoder en CASCADA (sec. 4.4): el primario genera los candidatos "
        "y este los re-puntua. Es la forma de usar dos encoders que si mejora; "
        "fusionar sus dos listas con RRF (--encoder-name A B) medido empeora. "
        "Activado por defecto (es la configuracion de la entrega); 'none' lo apaga.",
    )
    parser.add_argument(
        "--rerank-index-dir",
        type=Path,
        default=None,
        help="Indice del encoder en cascada. Por defecto base_vectorial/encoder_<nombre>. "
        "Existe aparte de --index-base para poder probar la cascada con el indice "
        "primario oficial y el secundario en otra carpeta.",
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=DEFAULT_RERANK_WEIGHT,
        help="Autoridad del segundo encoder en la mezcla de scores.",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=DEFAULT_RERANK_DEPTH,
        help="Candidatos que genera el primario antes del re-puntaje. Con la misma "
        "profundidad que --k-pool la cascada apenas cambia nada: el margen esta en "
        "dejar que el secundario ascienda documentos que el primario dejo mas abajo.",
    )
    parser.add_argument("--out", type=Path, default=RESULTADOS_PATH)
    parser.add_argument("--k-pool", type=int, default=DEFAULT_K_POOL)
    parser.add_argument("--fenomeno", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--formato", default=None)
    parser.add_argument("--idioma", default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--agg-strategy", default="sum", choices=["max", "sum", "mean"])
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Fusiona la recuperacion vectorial con el grafo de conocimiento (bonus, sec. 8.5) via RRF.",
    )
    parser.add_argument("--graph-path", type=Path, default=GRAFO_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.index_dir is not None and len(args.encoder_name) > 1:
        parser.error("--index-dir solo vale con un unico --encoder-name")

    # Un indice por encoder (sec. 4.4). Con uno solo, el flujo es el de siempre.
    encoders_indices = []
    for name in args.encoder_name:
        enc = get_encoder(name=name, use_fake=args.use_fake_encoder)
        if args.index_dir is not None:
            index_dir = args.index_dir
        elif args.index_base is not None:
            index_dir = args.index_base / f"encoder_{enc.name}"
        else:
            index_dir = None  # load_index resuelve la ruta oficial de entrega
        idx, meta = load_index(enc.name, index_dir=index_dir)
        logger.info(
            "encoder: %s -> indice de %d vectores%s",
            enc.name,
            idx.ntotal,
            " [FAKE -- solo pruebas]" if args.use_fake_encoder else "",
        )
        encoders_indices.append((enc, idx, meta))

    metadata = encoders_indices[0][2]
    metadata_by_chunk_id = {m["chunk_id"]: m for m in metadata}

    # Encoder en cascada (opcional): no aporta candidatos propios, solo
    # re-puntua los del primario, asi que se carga aparte de encoders_indices.
    rerank = None
    if args.rerank_encoder and args.rerank_encoder.lower() == "none":
        args.rerank_encoder = None
    if args.use_fake_encoder:
        # El encoder falso no tiene un indice secundario real que re-puntuar.
        args.rerank_encoder = None
    if args.rerank_encoder:
        enc_r = get_encoder(name=args.rerank_encoder, use_fake=args.use_fake_encoder)
        if args.rerank_index_dir is not None:
            rerank_dir = args.rerank_index_dir
        elif args.index_base is not None:
            rerank_dir = args.index_base / f"encoder_{enc_r.name}"
        else:
            rerank_dir = None
        idx_r, meta_r = load_index(enc_r.name, index_dir=rerank_dir)
        # La cascada lee el vector del chunk por su FILA en el otro indice: si
        # los indices no describen los mismos chunks en el mismo orden,
        # re-puntuaria el chunk equivocado y los resultados serian plausibles
        # pero mal fundados. Mejor abortar.
        try:
            verificar_alineacion(metadata, meta_r)
        except ValueError as exc:
            parser.exit(2, f"error: {exc}\n")
        rerank = (enc_r, idx_r)
        logger.info(
            "cascada: %s re-puntua los %d candidatos de %s (peso %.2f)",
            enc_r.name,
            args.rerank_depth,
            encoders_indices[0][0].name,
            args.rerank_weight,
        )

    # Los indices deben compartir exactamente los mismos chunks: RRF fusiona
    # por chunk_id y la metadata se lee de uno solo. Si no coinciden, alguien
    # construyo los indices por separado (fragmentando dos veces) y los
    # resultados serian silenciosamente incorrectos -- mejor abortar.
    for enc, _, meta in encoders_indices[1:]:
        if {m["chunk_id"] for m in meta} != set(metadata_by_chunk_id):
            parser.exit(
                2,
                f"error: los indices de {encoders_indices[0][0].name} y {enc.name} no "
                f"comparten los mismos chunk_id. Reconstruir ambos en una sola corrida "
                f"de la fase offline con --encoder-name {' '.join(args.encoder_name)}\n",
            )

    graph = None
    if args.use_graph:
        import networkx as nx

        # --use-graph es opcional (bonus): si el grafo no esta, conviene decirlo
        # en una linea y no con un traceback de networkx.
        if not args.graph_path.is_file():
            parser.exit(2, f"error: no existe el grafo {args.graph_path}; correr sin --use-graph\n")
        graph = nx.read_graphml(args.graph_path)
        logger.info(
            "grafo cargado: %d nodos, %d aristas",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    consultas = load_consultas(args.consultas)
    logger.info("consultas cargadas: %d", len(consultas))

    resultados = []
    # Con cascada el primario genera mas candidatos de los que se agregan, para
    # que el secundario pueda ascender documentos que quedaron por debajo del
    # pool. Sin cascada, k_pool de siempre.
    k_busqueda = max(args.k_pool, args.rerank_depth) if rerank else args.k_pool
    for consulta in consultas:
        # Una lista de candidatos por encoder (sec. 8.4).
        ranked_lists = [
            search(
                consulta["text"],
                enc,
                idx,
                meta,
                k=k_busqueda,
                fenomeno=args.fenomeno,
                formato=args.formato,
                idioma=args.idioma,
                min_score=args.min_score,
            )
            for enc, idx, meta in encoders_indices
        ]

        if rerank is not None:
            enc_r, idx_r = rerank
            ranked_lists = [
                rerank_por_segundo_encoder(
                    lista[: args.rerank_depth],
                    idx_r,
                    enc_r.encode_query(consulta["text"]),
                    peso=args.rerank_weight,
                )
                for lista in ranked_lists
            ]

        # El grafo entra como una lista mas a fusionar (sec. 8.5: "RRF
        # tratando el grafo como un indice adicional").
        if graph is not None:
            first = ranked_lists[0]
            query_lang = first[0].idioma if first else None
            graph_hits = graph_search(consulta["text"], graph, lang=query_lang, k=args.k_pool)
            if graph_hits:
                ranked_lists.append(graph_hits)

        if len(ranked_lists) == 1:
            hits = ranked_lists[0]
        else:
            fused = reciprocal_rank_fusion(ranked_lists, key=lambda h: h.chunk_id)
            hits = rebuild_hits_from_fusion(fused, metadata_by_chunk_id, limit=args.k_pool)

        # La agregacion a documento usa siempre k_pool candidatos. Con cascada
        # la lista viene con rerank_depth (200): recortarla aqui es lo que hace
        # que la profundidad extra sirva para REORDENAR y no para ampliar el
        # pool, que es un cambio distinto y no medido.
        hits = hits[: args.k_pool]

        resultados.append(
            build_result_object(consulta["query_id"], hits, agg_strategy=args.agg_strategy)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for resultado in resultados:
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")

    logger.info("resultados escritos en: %s (%d lineas)", args.out, len(resultados))


if __name__ == "__main__":
    main()
