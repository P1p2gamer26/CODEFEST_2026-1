"""Encoders de texto (sec. 4). Interfaz comun `Encoder` para poder
intercambiar implementaciones sin tocar indexacion ni recuperacion.

Solo arquitecturas encoder tipo BERT (sec. 4.2) -- ningun decoder/LLM
generativo en ninguna parte del pipeline.
"""

import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from ..config import (
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    ENCODER_E5_SMALL_HF_ID,
    ENCODER_E5_SMALL_NAME,
    ENCODER_GTE_HF_ID,
    ENCODER_GTE_NAME,
    ENCODER_PRIMARY_HF_ID,
    ENCODER_PRIMARY_NAME,
    ENCODER_SECONDARY_HF_ID,
    ENCODER_SECONDARY_NAME,
)


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

    # --- Codificacion asimetrica (consulta vs. pasaje) ---
    # Algunos encoders (familia E5) se entrenaron con prefijos DISTINTOS para
    # la consulta y para el documento indexado, y omitirlos degrada la calidad
    # de forma silenciosa. Los encoders que no los necesitan heredan estas
    # implementaciones por defecto y se comportan igual que antes.

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Codifica fragmentos para INDEXAR (src/embedding/build_index.py)."""
        return self.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Codifica una consulta para BUSCAR (src/retrieval/search.py)."""
        return self.encode_one(text)


def _reparar_buffers_no_persistentes(model) -> None:
    """Re-inicializa `position_ids` en los modelos con codigo remoto.

    Fallo real, no defensivo. `gte-multilingual-base` declara su buffer de
    posiciones con `persistent=False` (modeling.py:309), asi que NO viaja en
    el checkpoint; y transformers lo carga con `low_cpu_mem_usage` activado,
    que crea los tensores en el dispositivo `meta` y los materializa desde el
    checkpoint. Lo que no esta en el checkpoint queda apuntando a **memoria
    sin inicializar**: medido aqui, `position_ids[0] = 2635600166912` en vez
    de 0.

    El sintoma es una excepcion, no una degradacion silenciosa -- por suerte:

        IndexError: index 4104411873280 is out of bounds for dimension 0
        with size 569   (modeling.py:392, rope_cos[position_ids])

    La alternativa seria cargar con `low_cpu_mem_usage=False`, que materializa
    el state dict entero y duplica el pico de RAM (~2,4 GB para 305 M). En una
    maquina de 7,6 GB con 1,9 GB libres eso es cambiar un fallo por un OOM.
    Reescribir el buffer cuesta 8192 enteros.
    """
    import torch

    for modulo in model.modules():
        pid = getattr(modulo, "position_ids", None)
        if pid is not None and pid.dim() == 1 and pid.numel():
            esperado = torch.arange(pid.numel(), device=pid.device, dtype=pid.dtype)
            if not torch.equal(pid, esperado):
                modulo.register_buffer("position_ids", esperado, persistent=False)

        # RoPE: `inv_freq` sufre lo mismo y es MUCHO mas peligroso, porque no
        # revienta -- degrada en silencio. Medido aqui antes de repararlo:
        # inv_freq = [6.4e+20, 1.6e-42, 0.0] y cos_cached/sin_cached todo en
        # ceros, o sea el modelo codificaba sin informacion posicional. Los
        # vectores salian normalizados y de la forma correcta, y aun asi un
        # pasaje irrelevante puntuaba por encima de uno relevante. Sin esta
        # reparacion se habria descartado el encoder por "mala calidad"
        # cuando en realidad estaba roto al cargarse.
        inv = getattr(modulo, "inv_freq", None)
        if inv is None or inv.dim() != 1 or not inv.numel():
            continue
        dim = getattr(modulo, "dim", inv.numel() * 2)
        base = getattr(modulo, "base", 10000.0)
        esperado = 1.0 / (base ** (torch.arange(0, dim, 2, device=inv.device).float() / dim))
        if torch.allclose(inv.float(), esperado.to(inv.dtype).float(), rtol=1e-3, atol=1e-6):
            continue
        modulo.register_buffer("inv_freq", esperado.to(inv.dtype), persistent=False)
        # Las cachas cos/sin se derivan de inv_freq, asi que hay que rehacerlas.
        if hasattr(modulo, "_set_cos_sin_cache"):
            modulo._set_cos_sin_cache(
                seq_len=getattr(modulo, "max_position_embeddings", 512),
                device=inv.device,
                dtype=torch.get_default_dtype(),
            )


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
        # arquitectura (NewModel) en el repo del modelo en vez de usar una de
        # transformers. Sigue siendo encoder-only tipo BERT -- el flag es de
        # empaquetado, no cambia la arquitectura ni reabre la sec. 8.3.
        #
        # config_kwargs existe por un fallo concreto: GTE trae activadas por
        # defecto `unpad_inputs` y `use_memory_efficient_attention`, que
        # asumen GPU con atencion eficiente. En CPU su ruta de desempaquetado
        # arma `position_ids` con basura y revienta con
        # "IndexError: index 4104411873280 is out of bounds". Hay que
        # apagarlas por configuracion; no es opcional en esta maquina.
        extra = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
        if config_kwargs:
            extra["config_kwargs"] = config_kwargs
        self._model = SentenceTransformer(hf_id, **extra)
        if trust_remote_code:
            _reparar_buffers_no_persistentes(self._model)
        # GTE declara max_seq_length=8192. No lo necesitamos: el p99 de los
        # chunks es 466 tokens y el maximo 2.642. Y si importa: en CPU hay que
        # desactivar la atencion eficiente, con lo que el coste es CUADRATICO
        # en la longitud de padding. Recortarlo a 512 cubre el 99,3% de los
        # chunks enteros -- la misma ventana que e5-base, que ya sabemos
        # costear.
        if max_seq_length:
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
        # El nombre entra en la semilla para que dos encoders falsos distintos
        # produzcan rankings distintos (necesario para probar la fusion RRF
        # multi-encoder sin red). Sigue siendo determinista: mismo nombre +
        # mismo texto -> mismo vector.
        seed_material = f"{self.name}\x00{text}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._hash_vector(t) for t in texts]).astype("float32")

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# Encoders conocidos: nombre corto (el que se usa en la carpeta de entrega
# `encoder_<nombre>/` y en la CLI) -> como instanciarlo. Agregar un encoder
# nuevo es agregar una entrada aqui.
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
    ENCODER_GTE_NAME: {
        "hf_id": ENCODER_GTE_HF_ID,
        "query_prefix": "",
        "passage_prefix": "",
        "trust_remote_code": True,
        "config_kwargs": {"unpad_inputs": False, "use_memory_efficient_attention": False},
        "max_seq_length": 512,
    },
    # E02: mismos prefijos que e5-base, es la misma familia.
    ENCODER_E5_SMALL_NAME: {
        "hf_id": ENCODER_E5_SMALL_HF_ID,
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
            f"Agregar una entrada en KNOWN_ENCODERS (src/embedding/encoders.py)."
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
