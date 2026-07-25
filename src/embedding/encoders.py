"""Encoders de texto (sec. 4). Interfaz comun `Encoder` para poder
intercambiar implementaciones sin tocar indexacion ni recuperacion.

Solo arquitecturas encoder tipo BERT (sec. 4.2) -- ningun decoder/LLM
generativo en ninguna parte del pipeline.
"""

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from ..config import ENCODER_PRIMARY_HF_ID, ENCODER_PRIMARY_NAME


class Encoder(ABC):
    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Devuelve una matriz (n, dim) float32 normalizada a norma unitaria,
        para que el producto interno (IndexFlatIP) equivalga a similitud
        coseno (sec. 8.2)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Cuenta tokens tal como los veria este encoder -- se inyecta en
        src/chunking/chunker.py para que el presupuesto de tokens del
        chunking sea exacto respecto al limite de entrada del modelo
        (sec. 4.3), no una aproximacion por palabras."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class SentenceTransformerEncoder(Encoder):
    """Encoder real de produccion: `sentence-transformers/paraphrase-
    multilingual-MiniLM-L12-v2` (Apache 2.0, 384 dim, ES/EN/PT nativo, limite
    512 tokens -- ver config.py y el informe tecnico para la justificacion
    completa de la eleccion).

    Requiere descargar los pesos desde huggingface.co la primera vez. Eso NO
    es posible dentro del sandbox de desarrollo usado para construir este
    pipeline (el proxy de salida bloquea huggingface.co, igual que bloquea
    los dominios de descarga del corpus real -- ver informe_tecnico.pdf,
    seccion de limitaciones), pero funciona sin cambios en cualquier entorno
    con acceso normal a internet: `python scripts/build_corpus_index.py`.
    """

    def __init__(self, hf_id: str = ENCODER_PRIMARY_HF_ID, name: str = ENCODER_PRIMARY_NAME):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self._model = SentenceTransformer(hf_id)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def count_tokens(self, text: str) -> int:
        tokens = self._model.tokenizer(text, add_special_tokens=False)
        return len(tokens["input_ids"])


class HashingFakeEncoder(Encoder):
    """Encoder determinista por hashing, SIN modelo de lenguaje real.

    Uso EXCLUSIVO para desarrollo y pruebas dentro de este sandbox (donde
    huggingface.co esta bloqueado) y para que la suite de pytest corra
    rapido y sin red. Los vectores son reproducibles (mismo texto -> mismo
    vector) pero NO capturan significado semantico: sirven para validar la
    mecanica de indexacion/persistencia/agregacion/formato de salida, NO la
    calidad de la recuperacion. No debe usarse para construir el indice de
    la entrega final -- eso requiere `SentenceTransformerEncoder`.
    """

    def __init__(self, dim: int = 384, name: str = "hashing-fake-encoder"):
        self.dim = dim
        self.name = name

    def _hash_vector(self, text: str) -> np.ndarray:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_vector(t) for t in texts]).astype("float32")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@lru_cache(maxsize=None)
def get_encoder(name: str = ENCODER_PRIMARY_NAME, use_fake: bool = False) -> Encoder:
    if use_fake:
        return HashingFakeEncoder(name=name)
    return SentenceTransformerEncoder(name=name)
