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

COMPATIBILIDAD: ADL evalua con **Python >= 3.9.5** (Q&A final), y este archivo
anota tipos con la sintaxis `X | None` de PEP 604, que es 3.10+. En 3.9 las
anotaciones se EVALUAN al definir la funcion, asi que sin la linea de abajo el
script muere con `TypeError: unsupported operand type(s) for |` en el import
-- antes de leer una sola consulta, y la sec. 1.4 excluye lo que no es
reproducible. `from __future__ import annotations` las difiere a texto (PEP
563) y funciona desde 3.7. Auditado con AST: las 11 uniones del archivo son
todas de anotacion, ninguna se evalua en ejecucion, y no hay otra sintaxis
posterior a 3.9. **No quitar esta linea ni mover imports por encima de ella.**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
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
# Re-puntuador de la cascada. Encoder-only tipo BERT, NO decoder: cumple la
# sec. 8.3 igual que los otros dos. Apache 2.0, 305 M parametros, 768 dim.
# Reemplazo a multilingual-e5-base tras medirlo: NDCG@10 +0.055 sobre las 41
# consultas anotadas y +0.033 sobre las 10 independientes.
ENCODER_RERANK_NAME = "gte-multilingual-base"
ENCODER_RERANK_HF_ID = "Alibaba-NLP/gte-multilingual-base"

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


def _reparar_buffers_no_persistentes(model) -> None:
    """Re-inicializa los buffers posicionales de los modelos con codigo remoto.

    Fallo real y medido, no codigo defensivo. `gte-multilingual-base` declara
    sus buffers de RoPE con `persistent=False`, asi que NO viajan en el
    checkpoint; y transformers los carga con `low_cpu_mem_usage`, que crea los
    tensores en el dispositivo `meta` y los materializa desde el checkpoint.
    Lo que no esta en el checkpoint queda apuntando a **memoria sin
    inicializar**:

        position_ids[0] = 2635600166912   (deberia ser 0)
        inv_freq        = [6.4e+20, 1.6e-42, 0.0]

    El primero revienta con `IndexError` al codificar. **El segundo NO
    revienta**: el modelo codifica sin informacion posicional y devuelve
    vectores normalizados, de la forma correcta y semanticamente basura. Sin
    esta funcion, este script produciria resultados degradados en silencio.

    Solo reescribe si el contenido difiere del esperado, asi que en una
    version de transformers que ya no tenga el problema es un no-op.
    """
    import torch

    for modulo in model.modules():
        pid = getattr(modulo, "position_ids", None)
        if pid is not None and pid.dim() == 1 and pid.numel():
            esperado = torch.arange(pid.numel(), device=pid.device, dtype=pid.dtype)
            if not torch.equal(pid, esperado):
                modulo.register_buffer("position_ids", esperado, persistent=False)

        inv = getattr(modulo, "inv_freq", None)
        if inv is None or inv.dim() != 1 or not inv.numel():
            continue
        dim = getattr(modulo, "dim", inv.numel() * 2)
        base = getattr(modulo, "base", 10000.0)
        esperado = 1.0 / (base ** (torch.arange(0, dim, 2, device=inv.device).float() / dim))
        if torch.allclose(inv.float(), esperado.to(inv.dtype).float(), rtol=1e-3, atol=1e-6):
            continue
        modulo.register_buffer("inv_freq", esperado.to(inv.dtype), persistent=False)
        if hasattr(modulo, "_set_cos_sin_cache"):  # las cachas derivan de inv_freq
            modulo._set_cos_sin_cache(
                seq_len=getattr(modulo, "max_position_embeddings", 512),
                device=inv.device,
                dtype=torch.get_default_dtype(),
            )


class SentenceTransformerEncoder(Encoder):
    """Encoder real de produccion. Requiere descargar los pesos desde
    huggingface.co la primera vez; despues funciona desde la cache local."""

    def __init__(
        self,
        hf_id: str = ENCODER_PRIMARY_HF_ID,
        name: str = ENCODER_PRIMARY_NAME,
        query_prefix: str = "",
        passage_prefix: str = "",
        trust_remote_code: bool = False,
        config_kwargs: dict | None = None,
        max_seq_length: int | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.name = name
        # trust_remote_code lo exige gte-multilingual-base, que define su
        # arquitectura (NewModel) en el repo del modelo. Sigue siendo
        # encoder-only tipo BERT: el flag es de empaquetado, no reabre la
        # prohibicion de modelos generativos de la sec. 8.3.
        extra = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
        if config_kwargs:
            # GTE trae activadas `unpad_inputs` y `use_memory_efficient_attention`,
            # que asumen GPU con atencion eficiente. En CPU su ruta de
            # desempaquetado arma position_ids con basura y revienta.
            extra["config_kwargs"] = config_kwargs
        self._model = SentenceTransformer(hf_id, **extra)
        if trust_remote_code:
            _reparar_buffers_no_persistentes(self._model)
        if max_seq_length:
            # GTE declara 8192. El p99 de los chunks es 466 tokens, y con la
            # atencion densa el coste es cuadratico en el padding.
            self._model.max_seq_length = max_seq_length
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
    # GTE NO lleva prefijos: ponerselos degrada la calidad en silencio, igual
    # que omitirlos degrada a E5. Son dos errores simetricos.
    ENCODER_RERANK_NAME: {
        "hf_id": ENCODER_RERANK_HF_ID,
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": True,
        "config_kwargs": {"unpad_inputs": False, "use_memory_efficient_attention": False},
        "max_seq_length": 512,
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
        trust_remote_code=spec.get("trust_remote_code", False),
        config_kwargs=spec.get("config_kwargs"),
        max_seq_length=spec.get("max_seq_length"),
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


# --- Prior de recencia (sec. 8.7) -------------------------------------------
#
# La sec. 8.7 autoriza post-filtros por metadata incluyendo rango de fechas. No
# hace falta indexar nada nuevo: `fuente` ya guarda el nombre de archivo
# original y 509 de los 1826 documentos (28%) llevan el anio ahi.
#
# ALCANCE MEDIDO: solo 6 de las 50 consultas tienen marcador temporal, y apenas
# dos devuelven material claramente mas viejo que el relevante (q029: los
# relevantes son 2021-2026 y se entregan 2019, 2019, 2025). El techo son 2
# consultas.
#
# APAGADO POR DEFECTO (--prior-recencia 0). No se pudo validar contra el indice
# cuando se escribio; encenderlo y decidir con eval_mini.py --comparar-con.

_MARCADOR_TEMPORAL = re.compile(
    r"\brecient|\bactual|\búltim|\bultim|\bhoy\b|\bnuevo|\bemergent|\bcoyuntur"
    r"|\bvigente|\bahora\b|\bmoderno|\brecent\b|\bcurrent\b|\blatest\b"
    r"|\bde los ultimos\b|\bde los últimos\b|\b20[12]\d\b",
    re.IGNORECASE,
)

_ANIO_MIN, _ANIO_MAX = 1990, 2026
_ANIO = re.compile(r"(?:19|20)\d{2}")


def tiene_marcador_temporal(consulta: str) -> bool:
    """True si la consulta pide explicitamente material reciente.

    Deliberadamente conservador: activarlo de mas castiga documentos viejos que
    son la respuesta correcta (un tratado de 1967 sigue siendo el tratado).
    """
    return bool(_MARCADOR_TEMPORAL.search(consulta or ""))


def anio_de_fuente(fuente: str) -> int | None:
    """Anio de publicacion inferido del nombre de archivo, o None.

    Se toma el MAYOR de los presentes: los nombres del corpus a veces traen dos
    ("informe-2019-actualizado-2021") y el mas reciente fecha el documento.
    """
    candidatos = [
        int(a) for a in _ANIO.findall(fuente or "") if _ANIO_MIN <= int(a) <= _ANIO_MAX
    ]
    return max(candidatos) if candidatos else None


def anios_por_documento(hits: list[Hit]) -> dict[str, int]:
    """Mapa doc_id -> anio, a partir del `fuente` que ya trae cada Hit."""
    anios: dict[str, int] = {}
    for hit in hits:
        anio = anio_de_fuente(getattr(hit, "fuente", ""))
        if anio is not None and anio > anios.get(hit.doc_id, 0):
            anios[hit.doc_id] = anio
    return anios


def aplicar_prior_recencia(
    doc_hits: list[DocumentHit],
    anios: dict[str, int],
    peso: float = 0.05,
    anio_referencia: int = _ANIO_MAX,
) -> list[DocumentHit]:
    """Reordena candidatos a documento favoreciendo los mas recientes.

    Es un REORDENAMIENTO, no un filtro: un documento sin anio conocido (el 72%
    del corpus) conserva su puntaje tal cual. Con 28% de cobertura, filtrar por
    fecha tiraria material bueno por no tener el anio en el nombre.

    El bonus es `peso * (1 - (referencia - anio) / 36)`, acotado a [0, peso], de
    modo que un documento fechado nunca queda por debajo de uno sin fecha con el
    mismo puntaje. `peso` es chico a proposito: la senal semantica manda y esto
    solo desempata.
    """
    if not doc_hits or peso <= 0:
        return list(doc_hits)

    rango = max(1, anio_referencia - _ANIO_MIN)

    def puntaje(dh: DocumentHit) -> float:
        anio = anios.get(dh.doc_id)
        if anio is None:
            return dh.score
        cercania = max(0.0, min(1.0, 1.0 - (anio_referencia - anio) / rango))
        return dh.score + peso * cercania

    ordenados = sorted(doc_hits, key=puntaje, reverse=True)
    return [
        DocumentHit(rank=i, doc_id=dh.doc_id, score=dh.score)
        for i, dh in enumerate(ordenados, start=1)
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


# --- Deteccion de aparato bibliografico -------------------------------------
#
# La sec. 10.2.1 dice que la relevancia de cada fragmento se juzga sobre su
# **contenido textual** (campo `text`) y que el `chunk_id` NO es la clave de
# emparejamiento. O sea: el evaluador lee el texto que le entregamos, y un
# chunk que es una lista de notas al pie de un documento relevante vale 0
# aunque el documento sea el correcto. Medido sobre los 500 fragmentos de la
# entrega: 11 lo son, dos de ellos en rank 2 y 3.
#
# Se puntua POR SEGMENTO y no por chunk porque el caso frecuente no es el chunk
# 100% bibliografia sino el mixto: prosa util que termina en una cola de notas,
# porque el chunker corta por ventana de tokens y no respeta la frontera
# cuerpo/notas. Un umbral sobre el chunk entero los clasifica mal en los dos
# sentidos.
#
# Sin modelo: son patrones tipograficos. No hay embedding, ni clasificador
# entrenado, ni nada que se parezca a un decoder (sec. 8.3).

# Una URL dentro de un segmento corto es la senal mas fiable: la prosa de estos
# informes cita por numero, no pegando el enlace en medio de la frase.
_URL = re.compile(r"https?://|www\.\w|doi\.org/|\bdoi:", re.IGNORECASE)

# Marcador de nota al pie al principio del segmento: "[1047] ", "140 ", "12 ".
_MARCADOR = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4})[.,]?\s+[\"“«A-ZÀ-ÞА-Я]")

# Titulo entrecomillado seguido de coma: la forma canonica de cita en el corpus.
_TITULO_CITADO = re.compile(r"[\"“«][^\"”»]{12,}[\"”»]\s*[,.]")

# Mes en ingles/espanol/portugues seguido de anio: "September 27, 2018".
_MES_ANIO = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December|enero|febrero|marzo|abril|mayo|junio|julio|agosto"
    r"|septiembre|setiembre|octubre|noviembre|diciembre|janeiro|fevereiro"
    r"|mar[cç]o|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)

# Vocabulario de aparato: paginacion, edicion, acceso. Multilingue.
_VOCAB_APARATO = re.compile(
    r"\bet al\.|\baccessed\b|\bconsultado\b|\bacesso em\b|\bvol\.|\bno\.\s*\d"
    r"|\bpp?\.\s*\d|\bibid\b|\bop\. cit\.|\bed[s]?\.\,|\bretrieved\b"
    r"|\bdisponible en\b|\bavailable at\b",
    re.IGNORECASE,
)

# Corta en fin de oracion y tambien justo antes de un marcador de nota al pie.
# La segunda alternativa es la que mas trabajo hace: la extraccion pega la
# llamada a la nota al final de la palabra sin espacio ("...miniaturization of
# electronics.222 Despite..."), asi que despues del punto no hay blanco y la
# regla de fin-de-oracion no dispara. Sin ella el segmento de prosa se traga el
# bloque de notas que le sigue y hereda su URL: F2-SWF-078-c0242 daba 0.92 de
# aparato en vez de 0.44.
_CORTE = re.compile(
    r"(?<=[.!?;])\s+(?=[\"“«A-ZÀ-ÞА-Я\[\d])"
    r"|(?<=[.!?][0-9])\s+|(?<=[.!?][0-9]{2})\s+|(?<=[.!?][0-9]{3})\s+"
    r"|\s+(?=\[\d{1,4}\]\s*[\"“«A-ZÀ-Þ])"
    r"|\s+(?=\d{1,4}\s+[\"“«])"
    r"|\s+(?=\d{1,4}\s+[A-ZÀ-Þ][a-zà-þ]+[,\s])"
)

# Un segmento mas corto que esto se arrastra al anterior: evita que un "Jr." o
# un "S." partan una cita en trozos que por si solos no disparan nada.
_MIN_PALABRAS_SEGMENTO = 4


def segmentar(texto: str) -> list[str]:
    """Parte el texto en unidades juzgables (oracion o entrada bibliografica).

    Los chunks vienen sin saltos de linea -- la limpieza los colapsa -- asi que
    la unica frontera disponible es tipografica.
    """
    crudos = [s.strip() for s in _CORTE.split(texto) if s and s.strip()]
    if not crudos:
        return []
    fusionados: list[str] = [crudos[0]]
    for seg in crudos[1:]:
        if len(seg.split()) < _MIN_PALABRAS_SEGMENTO:
            fusionados[-1] = f"{fusionados[-1]} {seg}"
        else:
            fusionados.append(seg)
    return fusionados


def _es_referencia(segmento: str) -> bool:
    """True si el segmento parece aparato bibliografico y no prosa.

    La URL sola alcanza. El resto de senales necesitan dos coincidencias, para
    no marcar prosa que simplemente cita una fecha o usa comillas. El
    contraejemplo que guio el diseno es F1-CSET-098-c0070: 22% de tokens con
    digito y contenido legitimo ("Conduct cybersecurity threat assessments to
    identify potential threat actors [359, 360]"). Por eso el peso fuerte esta
    en la URL y el marcador de nota, no en los digitos sueltos.
    """
    if _URL.search(segmento):
        return True
    senales = sum(
        bool(patron.search(segmento))
        for patron in (_MARCADOR, _TITULO_CITADO, _MES_ANIO, _VOCAB_APARATO)
    )
    return senales >= 2


def fraccion_aparato(texto: str) -> float:
    """Fraccion de palabras del fragmento que caen en segmentos bibliograficos.

    Devuelve un valor en [0, 1]. 0 = prosa limpia, 1 = solo bibliografia. Un
    texto vacio devuelve 1.0: no aporta nada al evaluador.
    """
    segmentos = segmentar(texto)
    total = sum(len(s.split()) for s in segmentos)
    if total == 0:
        return 1.0
    aparato = sum(len(s.split()) for s in segmentos if _es_referencia(s))
    return aparato / total


# Idiomas que el evaluador de ADL puede leer. El corpus trae informes de
# SIPRI y SWF traducidos al coreano, ruso, arabe, chino y aleman, y sus chunks
# compiten por los mismos 10 cupos: 45 de los 500 fragmentos entregados salian
# en un idioma ilegible, y q006 gastaba 5 de sus 10 cupos asi.
IDIOMAS_LEGIBLES = frozenset({"es", "en", "pt"})

# A partir de que fraccion de aparato se considera que el fragmento no le dice
# nada al evaluador. Barrido sobre los 500 fragmentos de la entrega, modelando
# que ADL puntua 0 un fragmento asi: 0.50 y 0.60 dan exactamente el mismo
# resultado (ningun fragmento cae entre ambos), 0.75 degrada 3 menos y gana
# menos, 0.90 se queda corto. Se toma 0.60 por ser el mas conservador de los
# dos empatados.
UMBRAL_APARATO = 0.60


def ordenar_para_fragmentos(
    hits: list[Hit],
    doc_ids_prioritarios: list[str] | None = None,
    priorizar_idioma: bool = True,
    degradar_aparato: bool = True,
) -> list[Hit]:
    """Reordena los hits ANTES de armar los fragmentos. No filtra nada.

    Dos criterios, en este orden:

    1. **Alineacion con los documentos entregados.** `build_result_object`
       calculaba las dos mitades de la respuesta por caminos independientes:
       los documentos por agregacion del pool, los fragmentos por score crudo
       de chunk. Resultado medido sobre las 41 consultas anotadas a mano:
       **solo el 31% de los fragmentos venia de los 3 documentos que la propia
       respuesta declaraba mas relevantes**, con una mediana de 9 documentos
       distintos entre los 10 fragmentos. La respuesta se contradecia a si
       misma, y un fragmento de un documento que uno mismo dejo fuera del
       top-3 es casi seguro un cero en NDCG@10.
    2. **Idioma legible**, dentro de cada grupo (ver IDIOMAS_LEGIBLES). Es un
       post-filtro por metadata, que la sec. 8.7 autoriza; no interviene
       ningun modelo generativo.
    3. **No ser aparato bibliografico** (ver `fraccion_aparato`). La sec. 10.2.1
       juzga el fragmento por su campo `text`, asi que una lista de notas al pie
       puntua 0 aunque venga de un documento relevante.

       EFECTO PEQUENO Y MEDIDO: +0.003 de NDCG@10 modelando que el evaluador les
       da 0. Se aplica porque no puede perder (solo mueve casos claros al fondo
       de los mismos 10) y cuesta cero en CPU, no porque mueva la aguja. El
       proxy binario de eval_mini no puede ver esta mejora -- le da 1 a la
       bibliografia de un documento relevante --, por eso se justifica contra el
       reglamento y se mide con `ndcg_penalizado`.

    Reordena, NO descarta: si el top-3 no tiene 10 chunks, los de mas abajo
    completan igual, y si todos los chunks de una consulta fueran ilegibles se
    entregan esos. Asi el esquema de la sec. 1.4 (exactamente 10 fragmentos)
    no puede romperse por este cambio.

    `sorted` es estable, asi que el ultimo criterio de desempate es el orden
    de entrada, o sea el score. No hace falta pasarlo explicito.
    """
    if not doc_ids_prioritarios and not priorizar_idioma and not degradar_aparato:
        return hits
    top = set(doc_ids_prioritarios or ())

    def clave(hit: Hit) -> tuple[int, int, int]:
        fuera_del_top = 1 if (top and hit.doc_id not in top) else 0
        ilegible = 1 if (priorizar_idioma and hit.idioma not in IDIOMAS_LEGIBLES) else 0
        aparato = (
            1
            if (degradar_aparato and fraccion_aparato(hit.texto) >= UMBRAL_APARATO)
            else 0
        )
        return (fuera_del_top, ilegible, aparato)

    return sorted(hits, key=clave)


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
#
# Cascada de TRES encoders: MiniLM trae los candidatos y los re-puntuan gte y
# e5, cada uno sumando su similitud con el mismo peso. Estructura elegida
# midiendo cinco alternativas con los tres indices ya construidos
# (scripts/barrido_estructuras.py), coste cero de codificacion:
#
#   | estructura      | F1(41) | NDCG(41) | F1(10) | NDCG(10) |
#   |-----------------|--------|----------|--------|----------|
#   | MiniLM->e5      | 0.344  |  0.338   | 0.333  |  0.329   |
#   | MiniLM->gte     | 0.378  |  0.393   | 0.333  |  0.362   |
#   | gte solo        | 0.311  |  0.309   | 0.333  |  0.287   |
#   | gte->MiniLM     | 0.385  |  0.393   | 0.200  |  0.202   |
#   | gte->e5         | 0.339  |  0.341   | 0.400  |  0.322   |
#   | MiniLM->gte+e5  | 0.386  |  0.406   | 0.333  |  0.360   |
#
# **gte como primario se derrumba en la muestra independiente** (F1 0.200)
# mientras luce bien en las 41: es sesgo de pooling, porque esas 41 se
# anotaron sobre candidatos que propuso MiniLM. Descartada por eso, no por el
# promedio.
#
# La triple es la unica ADOPTABLE en las cuatro mediciones por el criterio del
# proyecto (IC al 90% que excluya una perdida de 0,02). Ojo: la ganancia sobre
# MiniLM->gte es CHICA -- +0,014 de NDCG en las 41 y -0,002 en las 10.
#
# **El orden de los re-puntuadores no altera el resultado**: cada uno suma su
# similitud sobre la misma lista y la suma es conmutativa. El recorte a
# rerank_depth se hace una sola vez, antes del primero.
DEFAULT_RERANK_ENCODERS = [ENCODER_RERANK_NAME, ENCODER_SECONDARY_NAME]


def load_consultas(path: Path) -> list[dict]:
    """Lee el archivo de consultas. Tolerante en la ENTRADA, estricto en la
    salida: el formato exacto de q001-q050 lo define ADL y no lo controlamos.

    `utf-8-sig` en vez de `utf-8` NO es cosmetico. Cualquier archivo guardado
    con Excel, Bloc de notas o PowerShell en Windows lleva BOM, y con `utf-8`
    la primera linea falla con "Unexpected UTF-8 BOM" -- el script muere antes
    de la primera consulta y la sec. 1.4 excluye lo que no se puede
    reproducir. `utf-8-sig` se come el BOM si esta y no molesta si no.

    Los errores salen como mensaje, no como traceback: quien reciba solo esta
    carpeta necesita saber QUE arreglar, y un volcado de pila parece un script
    roto.
    """
    if not path.is_file():
        raise SystemExit(f"error: no existe el archivo de consultas: {path}")

    consultas = []
    with path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"error: {path}:{line_number} no es JSON valido ({exc.msg}).\n"
                    f"  Se espera JSON Lines: un objeto por linea, sin comas entre lineas.\n"
                    f"  Linea: {line[:120]}"
                ) from exc
            if not isinstance(obj, dict):
                raise SystemExit(
                    f"error: {path}:{line_number}: se esperaba un objeto JSON, "
                    f"no {type(obj).__name__}"
                )
            query_id = obj.get("query_id") or obj.get("id")
            text = obj.get("text") or obj.get("query") or obj.get("consulta")
            if query_id is None or text is None:
                raise SystemExit(
                    f"error: {path}:{line_number}: se esperaban los campos "
                    f"'query_id'/'id' y 'text'/'query'/'consulta'; campos "
                    f"encontrados: {sorted(obj)}"
                )
            consultas.append({"query_id": str(query_id), "text": str(text)})

    if not consultas:
        raise SystemExit(
            f"error: {path} no tiene ninguna consulta legible.\n"
            f"  Formato esperado (JSON Lines):\n"
            f'    {{"query_id": "q001", "text": "..."}}'
        )
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
    alinear_fragmentos: bool = True,
    priorizar_idioma: bool = True,
    cupo_alineado: int = N_FRAGMENTS_PER_QUERY,
    # Peso del prior de recencia (sec. 8.7). 0 = apagado, que es el default:
    # no se pudo validar contra el indice cuando se escribio. Ver
    # `aplicar_prior_recencia`.
    prior_recencia: float = 0.0,
) -> dict:
    doc_hits = aggregate_documents(hits, top_n=top_docs, strategy=agg_strategy)
    if prior_recencia > 0:
        doc_hits = aplicar_prior_recencia(
            doc_hits, anios_por_documento(hits), peso=prior_recencia
        )
    # Los fragmentos se ordenan DESPUES de saber cuales son los documentos
    # entregados, para que las dos mitades de la respuesta no se contradigan.
    # Solo reordena el pool: `doc_hits` ya esta calculado y no se toca, asi
    # que este cambio NO puede mover el F1@3.
    top_ids = [d.doc_id for d in doc_hits] if alinear_fragmentos else None
    if top_ids and 0 < cupo_alineado < max_fragments:
        # Cobertura: reservar `cupo_alineado` cupos para el top-3 y dejar el
        # resto abierto al pool. Alinear los 10 cupos amplifica el ranking de
        # documentos en las dos direcciones -- medido sobre las 41 anotadas,
        # gana en 25 consultas (todas con F1@3 > 0) y pierde en 12, de las
        # cuales 11 tienen F1@3 = 0. O sea que cuando el top-3 se equivoca,
        # concentrar ahi los 10 fragmentos entrega diez ceros seguros. Los
        # cupos libres son el seguro contra eso.
        del_top = enforce_word_limit(
            ordenar_para_fragmentos(
                [h for h in hits if h.doc_id in set(top_ids)],
                doc_ids_prioritarios=top_ids,
                priorizar_idioma=priorizar_idioma,
            ),
            max_fragments=cupo_alineado,
            max_words=max_words,
        )
        vistos = {f["chunk_id"] for f in del_top}
        del_resto = enforce_word_limit(
            ordenar_para_fragmentos(
                [h for h in hits if h.chunk_id not in vistos],
                doc_ids_prioritarios=None,
                priorizar_idioma=priorizar_idioma,
            ),
            max_fragments=max_fragments - len(del_top),
            max_words=max_words,
        )
        fragments = del_top + del_resto
        for i, frag in enumerate(fragments, start=1):
            frag["rank"] = i
    else:
        hits_ordenados = ordenar_para_fragmentos(
            hits, doc_ids_prioritarios=top_ids, priorizar_idioma=priorizar_idioma
        )
        fragments = enforce_word_limit(
            hits_ordenados, max_fragments=max_fragments, max_words=max_words
        )

    # La sec. 9.2 exige EXACTAMENTE 3 documentos. Si el pool no contiene 3
    # documentos distintos no hay de donde sacarlos, pero emitir una linea
    # corta en silencio es peor: se avisa para que quien corra el script sepa
    # que esa consulta sale malformada.
    #
    # Con las 50 consultas oficiales no ocurre (las 50 dan 3 documentos). Se
    # reproduce con entradas degeneradas -- una consulta de 3.000 caracteres
    # repetidos produce un vector que cae en un rincon del espacio donde los
    # 60 chunks del pool salen todos del mismo documento.
    if len(doc_hits) < top_docs:
        logger.warning(
            "%s: solo %d documento(s) distinto(s) en el pool, se esperaban %d; "
            "la linea no cumple el esquema de la sec. 9.2",
            query_id,
            len(doc_hits),
            top_docs,
        )

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
        nargs="+",
        default=list(DEFAULT_RERANK_ENCODERS),
        help="Encoders en CASCADA (sec. 4.4): el primario genera los candidatos y "
        "estos los re-puntuan, cada uno sumando su similitud con el mismo peso. "
        "Acepta varios. Es la forma de usar mas de un encoder que si mejora; "
        "fusionar sus listas con RRF (--encoder-name A B) medido empeora. "
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
    # topM (top2, top3, top5...) suma solo los M mejores fragmentos de cada
    # documento. La rama ya existia en aggregate_documents; el CLI no la
    # exponia. El default no cambia: sigue siendo "sum".
    parser.add_argument(
        "--agg-strategy",
        default="sum",
        choices=["max", "sum", "mean", "top2", "top3", "top5", "top10"],
    )
    # Las dos banderas siguientes APAGAN comportamientos activos por defecto.
    # Existen para poder medir cada uno por separado contra la entrega
    # anterior; correr el script sin flags reproduce lo que se entrega.
    parser.add_argument(
        "--sin-alinear-fragmentos",
        action="store_true",
        help="Vuelve al comportamiento viejo: los 10 fragmentos se toman por score "
        "crudo de chunk, sin preferir los de los 3 documentos entregados.",
    )
    parser.add_argument(
        "--cupo-alineado",
        type=int,
        default=N_FRAGMENTS_PER_QUERY,
        help="Cuantos de los 10 cupos de fragmento se reservan para los 3 documentos "
        "entregados. El resto queda abierto al pool, como seguro para las consultas "
        "cuyo top-3 esta equivocado.",
    )
    parser.add_argument(
        "--sin-priorizar-idioma",
        action="store_true",
        help="No manda al final los fragmentos en idiomas que el evaluador no lee "
        "(ver IDIOMAS_LEGIBLES).",
    )
    parser.add_argument(
        "--prior-recencia",
        type=float,
        default=0.0,
        metavar="PESO",
        help="Peso del prior de recencia (sec. 8.7), aplicado SOLO a las consultas "
        "con marcador temporal. 0 = apagado, que es el default: afecta a 6 de las 50 "
        "consultas y no se pudo validar contra el indice. Probar con 0.05 y decidir "
        "con eval_mini.py --comparar-con.",
    )
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

    # Encoders en cascada (opcional): no aportan candidatos propios, solo
    # re-puntuan los del primario, asi que se cargan aparte de
    # encoders_indices. Pueden ser VARIOS: cada uno suma su similitud con el
    # mismo peso sobre la misma lista de candidatos. Medido, no supuesto --
    # ver DEFAULT_RERANK_ENCODERS.
    reranks: list = []
    nombres_rerank = list(args.rerank_encoder or [])
    if len(nombres_rerank) == 1 and nombres_rerank[0].lower() == "none":
        nombres_rerank = []
    if args.use_fake_encoder:
        # El encoder falso no tiene un indice secundario real que re-puntuar.
        nombres_rerank = []
    for nombre_r in nombres_rerank:
        try:
            enc_r = get_encoder(name=nombre_r, use_fake=args.use_fake_encoder)
        except Exception as exc:  # noqa: BLE001
            # gte-multilingual-base necesita `trust_remote_code` y descargar
            # ~1,2 GB de huggingface.co. Si el entorno lo impide, un traceback
            # crudo no le dice al evaluador que hacer. Esto si:
            raise SystemExit(
                f"\nNo se pudo cargar el re-puntuador {nombre_r!r}: "
                f"{type(exc).__name__}: {exc}\n\n"
                "gte-multilingual-base define su arquitectura en el repositorio del "
                "modelo, asi que necesita `trust_remote_code=True` y acceso a "
                "huggingface.co.\n"
                "Si este entorno no lo permite, la entrega incluye el indice del "
                "re-puntuador anterior; volve a correr con:\n\n"
                f"    python generador.py --consultas <archivo> "
                f"--rerank-encoder {ENCODER_SECONDARY_NAME}\n\n"
                "Los resultados NO seran identicos a resultados.jsonl (esa variante "
                "da NDCG@10 0.338 en vez de 0.406), pero el sistema funciona.\n"
                "Para desactivar la cascada del todo: --rerank-encoder none\n"
            ) from exc
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
        reranks.append((enc_r, idx_r))
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
    k_busqueda = max(args.k_pool, args.rerank_depth) if reranks else args.k_pool
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

        # Cada re-puntuador suma su similitud sobre la MISMA lista, uno tras
        # otro. El recorte a rerank_depth se hace una sola vez, antes del
        # primero: repetirlo por cada re-puntuador iria descartando candidatos
        # entre pasadas y el resultado dependeria del orden de los encoders.
        if reranks:
            ranked_lists = [lista[: args.rerank_depth] for lista in ranked_lists]
            for enc_r, idx_r in reranks:
                qv = enc_r.encode_query(consulta["text"])
                ranked_lists = [
                    rerank_por_segundo_encoder(lista, idx_r, qv, peso=args.rerank_weight)
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
            build_result_object(
                consulta["query_id"],
                hits,
                agg_strategy=args.agg_strategy,
                alinear_fragmentos=not args.sin_alinear_fragmentos,
                priorizar_idioma=not args.sin_priorizar_idioma,
                cupo_alineado=args.cupo_alineado,
                # Solo donde la consulta pide material reciente. Aplicarlo a
                # todas castigaria documentos viejos que son la respuesta
                # correcta -- un tratado de 1967 sigue siendo el tratado.
                prior_recencia=(
                    args.prior_recencia
                    if tiene_marcador_temporal(consulta["text"])
                    else 0.0
                ),
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for resultado in resultados:
            f.write(json.dumps(resultado, ensure_ascii=False) + "\n")

    logger.info("resultados escritos en: %s (%d lineas)", args.out, len(resultados))


if __name__ == "__main__":
    main()
