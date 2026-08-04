"""BM25 Okapi sobre los mismos chunks del indice FAISS: recuperacion lexica
para complementar la densa (sec. 8.4, se fusiona por RRF).

Motivacion medida: el encoder denso pierde los terminos discriminantes raros
que no alinea entre idiomas. El caso documentado es NBQR/CBRN, pero el efecto
es general -- una sigla o un toponimo poco frecuente pesa poco en un vector
de 384 dimensiones y mucho en el idf de BM25.

No usa ninguna dependencia externa (`rank_bm25`, sklearn): `Entrega/generador.py`
tiene que seguir siendo autocontenido, y BM25 son cuarenta lineas. Tampoco hay
stemming ni stopwords: el corpus es ES/EN/PT mezclado y un stemmer por idioma
seria complejidad sin medicion que la respalde.
"""

import heapq
import math
import re
from array import array

from .search import Hit

K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(texto: str) -> list[str]:
    return _TOKEN_RE.findall(texto.lower())


class BM25:
    """Indice invertido en memoria construido desde la metadata del indice
    FAISS. El indice `i` de un chunk es el mismo en ambos, asi que los `Hit`
    que devuelve `search()` son fusionables por `chunk_id` con los densos.
    """

    def __init__(self, metadata: list[dict]) -> None:
        self.metadata = metadata
        self.n = len(metadata)
        # postings[t] = (array de indices de chunk, array de frecuencias).
        # Arrays en vez de listas de tuplas: con ~19M de postings la diferencia
        # es de ~115 MB contra mas de 1 GB.
        self.postings: dict[str, tuple[array, array]] = {}

        longitudes = array("i")
        for i, meta in enumerate(metadata):
            tokens = tokenize(meta["texto"])
            longitudes.append(len(tokens))
            frecuencias: dict[str, int] = {}
            for t in tokens:
                frecuencias[t] = frecuencias.get(t, 0) + 1
            for t, tf in frecuencias.items():
                entrada = self.postings.get(t)
                if entrada is None:
                    entrada = (array("i"), array("H"))
                    self.postings[t] = entrada
                entrada[0].append(i)
                entrada[1].append(min(tf, 65535))

        self.doc_len = longitudes
        self.avgdl = (sum(longitudes) / self.n) if self.n else 0.0

    def scores(self, query: str) -> dict[int, float]:
        acumulado: dict[int, float] = {}
        for termino in set(tokenize(query)):
            entrada = self.postings.get(termino)
            if entrada is None:
                continue
            indices, frecuencias = entrada
            df = len(indices)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in zip(indices, frecuencias):
                norma = K1 * (1.0 - B + B * self.doc_len[i] / self.avgdl)
                acumulado[i] = acumulado.get(i, 0.0) + idf * tf * (K1 + 1.0) / (tf + norma)
        return acumulado

    def search(self, query: str, k: int = 10) -> list[Hit]:
        acumulado = self.scores(query)
        mejores = heapq.nlargest(k, acumulado.items(), key=lambda par: par[1])
        hits = []
        for rank, (i, score) in enumerate(mejores, start=1):
            meta = self.metadata[i]
            hits.append(
                Hit(
                    rank=rank,
                    score=float(score),
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
