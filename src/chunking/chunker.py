"""Chunking hibrido (sec. 3 de la especificacion).

Estrategia, de mayor a menor granularidad:
1. Estructural: el texto se separa por encabezados Markdown ('#'..'######'),
   que ya vienen marcados por los extractores de HTML/JSON/MD (sec. 2.1). Si
   no hay encabezados (caso tipico de PDF), todo el documento es una sola
   "seccion" sin titulo.
2. Dentro de cada seccion, segmentacion de oraciones multilingue
   (sentence_split.split_sentences).
3. Empaquetado voraz por presupuesto de tokens (config.CHUNK_TOKEN_BUDGET),
   contando tokens con el tokenizer del encoder que se vaya a usar para
   indexar (se inyecta via `count_tokens`; por defecto, un conteo
   aproximado por palabras para poder usar el modulo de forma standalone).
4. Solape de `overlap_sentences` oraciones entre chunks consecutivos.

Garantia de completitud linguistica (sec. 3.3): la unidad minima que se
mueve entre chunks es siempre una oracion completa devuelta por
`split_sentences`; el chunker nunca corta un caracter en medio de una
oracion. La unica excepcion estructural es CSV/XLSX, donde cada fila ya es
una unidad atomica (no son oraciones en lenguaje natural) y se indexa
completa sin importar el presupuesto de tokens.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..config import CHUNK_OVERLAP_SENTENCES, CHUNK_TOKEN_BUDGET
from .sentence_split import split_sentences

_HEADER_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)\s*$")

CountTokensFn = Callable[[str], int]


@dataclass
class Chunk:
    texto: str
    posicion: int
    num_tokens: int
    titulo_seccion: str | None


def _default_count_tokens(texto: str) -> int:
    return len(texto.split())


def split_into_sections(texto: str) -> list[tuple[str | None, str]]:
    """Separa el texto en (titulo_seccion, contenido) usando encabezados '#'.
    Sin encabezados, devuelve una unica seccion sin titulo."""
    matches = list(_HEADER_RE.finditer(texto))
    if not matches:
        return [(None, texto)] if texto.strip() else []

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        leading = texto[: matches[0].start()].strip()
        if leading:
            sections.append((None, leading))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        content = texto[start:end].strip()
        if content:
            sections.append((heading, content))

    return sections


def _section_sentences(content: str, lang: str | None) -> list[str]:
    sentences: list[str] = []
    for parrafo in content.split("\n\n"):
        parrafo = parrafo.strip()
        if parrafo:
            sentences.extend(split_sentences(parrafo, lang))
    return sentences


def _pack_sentences(
    sentences: list[str],
    heading: str | None,
    posicion_inicial: int,
    count_tokens: CountTokensFn,
    token_budget: int,
    overlap_sentences: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    posicion = posicion_inicial
    current: list[str] = []
    current_tokens = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = count_tokens(sent)

        if current and current_tokens + sent_tokens > token_budget:
            texto_chunk = " ".join(current)
            chunks.append(
                Chunk(
                    texto=texto_chunk,
                    posicion=posicion,
                    num_tokens=count_tokens(texto_chunk),
                    titulo_seccion=heading,
                )
            )
            posicion += 1
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_tokens = count_tokens(" ".join(current)) if current else 0
            continue  # reintenta la misma oracion contra el chunk reiniciado

        current.append(sent)
        current_tokens += sent_tokens
        i += 1

    if current:
        texto_chunk = " ".join(current)
        chunks.append(
            Chunk(
                texto=texto_chunk,
                posicion=posicion,
                num_tokens=count_tokens(texto_chunk),
                titulo_seccion=heading,
            )
        )

    return chunks


def _chunk_tabular_rows(texto_crudo: str, count_tokens: CountTokensFn) -> list[Chunk]:
    bloques = [b.strip() for b in texto_crudo.split("\n\n") if b.strip()]
    return [
        Chunk(texto=b, posicion=i, num_tokens=count_tokens(b), titulo_seccion=None)
        for i, b in enumerate(bloques)
    ]


def chunk_document(
    texto_crudo: str,
    formato: str,
    lang: str | None = None,
    count_tokens: CountTokensFn | None = None,
    token_budget: int = CHUNK_TOKEN_BUDGET,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[Chunk]:
    if not texto_crudo or not texto_crudo.strip():
        return []

    count_tokens = count_tokens or _default_count_tokens

    if formato in ("csv", "xlsx"):
        return _chunk_tabular_rows(texto_crudo, count_tokens)

    chunks: list[Chunk] = []
    posicion = 0
    for heading, content in split_into_sections(texto_crudo):
        sentences = _section_sentences(content, lang)
        if not sentences:
            continue
        section_chunks = _pack_sentences(
            sentences, heading, posicion, count_tokens, token_budget, overlap_sentences
        )
        chunks.extend(section_chunks)
        posicion += len(section_chunks)

    return chunks
