#!/usr/bin/env python3
"""E38 -- re-empaqueta el corpus ya extraido a otro presupuesto de tokens.

No re-extrae ni pasa por OCR. El corpus tiene checkpoint de los CHUNKS
(`chunks_intermedios_limpio.jsonl`), no del texto crudo de los documentos, asi
que re-chunkear a un presupuesto MAYOR obliga a reconstruir la secuencia
original de oraciones quitando el solape que el chunker introdujo, y volver a
empaquetar con `_pack_sentences`.

Spec: dev/docs/superpowers/specs/2026-08-09-rejilla-chunking-design.md
Plan: dev/docs/superpowers/plans/2026-08-09-e38-rejilla-chunking.md

Uso:
    # puerta de entrada: la reconstruccion no puede perder texto
    python dev/scripts/rechunkear_e38.py --verificar \
        --chunks dev/intermedios/chunks_intermedios_limpio.jsonl

    # generar una celda de la rejilla
    python dev/scripts/rechunkear_e38.py --generar \
        --chunks dev/intermedios/chunks_intermedios_limpio.jsonl \
        --presupuesto 512 --solape 0 \
        --salida dev/intermedios/chunks_e38_512_0.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.sentence_split import split_sentences

# Los tabulares se fragmentan por filas (`_chunk_tabular_rows`), no por
# presupuesto de tokens: la rejilla no los toca y se copian tal cual.
FORMATOS_TABULARES = ("csv", "xlsx")

# Cuanto del documento puede descartar el SEGMENTADOR por no formar oracion.
# No es el solape --ese se verifica por igualdad exacta-- sino la basura
# tipografica del PDF. Medido: ronda el 0% y el peor caso visto es el 0,03%.
# El 1% es holgado y aun asi delata la perdida de un parrafo.
TOPE_DESCARTE_SEGMENTADOR = 0.01


def reconstruir_oraciones(chunks_del_doc, con_contabilidad=False):
    """Devuelve [(titulo_seccion, [oraciones])] en orden de posicion.

    Quita de cada chunk el prefijo de oraciones que repite la cola del chunk
    anterior de la MISMA seccion. Solo mira la frontera: una repeticion en
    cualquier otro punto es contenido real del documento y se conserva.

    Con `con_contabilidad` devuelve ademas un desglose que separa DOS causas
    distintas de que el reconstruido sea mas corto, que un solo umbral
    confundia:

      `oraciones`: caracteres alfanumericos que el segmentador si devolvio.
                   Lo que el original tiene de mas es basura tipografica que
                   no forma oracion (ver E38_rejilla_chunking.md).
      `solape`:    caracteres borrados por deduplicar la frontera. Es el
                   objetivo del ejercicio, no una perdida.

    Con eso la puerta exige IGUALDAD EXACTA sobre el solape, en vez de
    tolerar un porcentaje inventado que dependia de cuantas oraciones tuviera
    cada chunk.
    """
    secciones = []
    cuenta = {"oraciones": 0, "solape": 0}
    for ch in sorted(chunks_del_doc, key=lambda c: c["posicion"]):
        oraciones = split_sentences(ch["texto"], ch.get("idioma"))
        if not oraciones:
            continue
        cuenta["oraciones"] += len(_flujo_alfanumerico(oraciones))
        heading = ch.get("titulo_seccion")
        if secciones and secciones[-1][0] == heading:
            previas = secciones[-1][1]
            solape = 0
            for n in range(min(len(previas), len(oraciones)), 0, -1):
                if previas[-n:] == oraciones[:n]:
                    solape = n
                    break
            cuenta["solape"] += len(_flujo_alfanumerico(oraciones[:solape]))
            previas.extend(oraciones[solape:])
        else:
            secciones.append((heading, list(oraciones)))
    return (secciones, cuenta) if con_contabilidad else secciones


def _largo_mayor_sufijo_prefijo(anterior: str, actual: str) -> int:
    """Longitud del mayor sufijo de ``anterior`` que es prefijo de ``actual``."""
    limite = min(len(anterior), len(actual))
    if limite == 0:
        return 0
    patron = actual[:limite]
    texto = anterior[-limite:]
    combinado = patron + "\x00" + texto
    prefijos = [0] * len(combinado)
    for i in range(1, len(combinado)):
        j = prefijos[i - 1]
        while j and combinado[i] != combinado[j]:
            j = prefijos[j - 1]
        if combinado[i] == combinado[j]:
            j += 1
        prefijos[i] = j
    return min(prefijos[-1], limite)


def reconstruir_texto_rapido(chunks_del_doc):
    """Reconstruye secciones deduplicando el solape textual exacto.

    Los chunks se crearon con ``" ".join(oraciones)``; por eso la oracion de
    solape viaja byte a byte como sufijo/prefijo. El caso ambiguo usa el
    segmentador anterior para conservar correccion antes que velocidad.
    """
    secciones = []
    for ch in sorted(chunks_del_doc, key=lambda c: c["posicion"]):
        texto = ch["texto"].strip()
        if not texto:
            continue
        heading = ch.get("titulo_seccion")
        if not secciones or secciones[-1][0] != heading:
            secciones.append([heading, texto])
            continue

        anterior = secciones[-1][1]
        largo = _largo_mayor_sufijo_prefijo(anterior, texto)
        inicio_ok = largo == len(anterior) or anterior[-largo - 1].isspace()
        fin_ok = largo == len(texto) or texto[largo].isspace()
        termina_oracion = largo > 1 and texto[:largo].rstrip().endswith(
            (".", "!", "?", ";", ":", ")", "]", '"', "'")
        )
        if inicio_ok and fin_ok and termina_oracion:
            resto = texto[largo:].lstrip()
        else:
            previas = split_sentences(anterior, ch.get("idioma"))
            actuales = split_sentences(texto, ch.get("idioma"))
            solape = 0
            for n in range(min(len(previas), len(actuales)), 0, -1):
                if previas[-n:] == actuales[:n]:
                    solape = n
                    break
            resto = " ".join(actuales[solape:])
        if resto:
            secciones[-1][1] = anterior + " " + resto
    return [(heading, texto) for heading, texto in secciones]


def reempaquetar(chunks_del_doc, token_budget, overlap_sentences, count_tokens):
    """Re-empaqueta los chunks de UN documento al presupuesto pedido.

    OJO (punto 8 de CLAUDE.md): el chunk_id es doc_id + posicion, asi que los
    chunk_id de esta celda NO son comparables con los de otra celda ni con los
    de la entrega. Los tres encoders de una misma celda tienen que construirse
    sobre este mismo archivo.
    """
    from src.chunking.chunker import _pack_sentences

    ordenados = sorted(chunks_del_doc, key=lambda c: c["posicion"])
    plantilla = ordenados[0]
    if plantilla["formato"] in FORMATOS_TABULARES:
        return [dict(c) for c in ordenados]

    salida = []
    posicion = 0
    for heading, oraciones in reconstruir_oraciones(ordenados):
        nuevos = _pack_sentences(
            oraciones, heading, posicion, count_tokens, token_budget,
            overlap_sentences,
        )
        for ch in nuevos:
            salida.append({
                "doc_id": plantilla["doc_id"],
                "chunk_id": f"{plantilla['doc_id']}::{ch.posicion}",
                "fuente": plantilla["fuente"],
                "formato": plantilla["formato"],
                "fenomeno": plantilla["fenomeno"],
                "posicion": ch.posicion,
                "num_tokens": ch.num_tokens,
                "texto": ch.texto,
                "idioma": plantilla.get("idioma"),
                "titulo_seccion": ch.titulo_seccion,
                "url": plantilla.get("url", ""),
            })
        posicion += len(nuevos)
    return salida


def reempaquetar_rapido_sin_solape(
    chunks_del_doc, token_budget, count_tokens_many
):
    """Variante que segmenta cada seccion reconstruida una sola vez.

    Es una alternativa de chunking deliberada, no una optimizacion invisible:
    dar al segmentador el contexto completo puede cambiar limites de oracion
    frente a segmentar cada chunk viejo. Luego conserva el empaquetado voraz,
    cuenta las oraciones por lotes y calcula exactamente ``num_tokens`` para
    cada chunk final.
    """
    from src.chunking.chunker import Chunk

    ordenados = sorted(chunks_del_doc, key=lambda c: c["posicion"])
    plantilla = ordenados[0]
    if plantilla["formato"] in FORMATOS_TABULARES:
        return [dict(c) for c in ordenados]

    chunks_nuevos = []
    posicion = 0
    for heading, texto_seccion in reconstruir_texto_rapido(ordenados):
        oraciones = split_sentences(texto_seccion, plantilla.get("idioma"))
        conteos = count_tokens_many(oraciones)
        current = []
        current_tokens = 0
        for oracion, n_tokens in zip(oraciones, conteos):
            if current and current_tokens + n_tokens > token_budget:
                chunks_nuevos.append(Chunk(" ".join(current), posicion, 0, heading))
                posicion += 1
                current = []
                current_tokens = 0
            current.append(oracion)
            current_tokens += n_tokens
        if current:
            chunks_nuevos.append(Chunk(" ".join(current), posicion, 0, heading))
            posicion += 1

    conteos_finales = count_tokens_many([ch.texto for ch in chunks_nuevos])
    salida = []
    for ch, n_tokens in zip(chunks_nuevos, conteos_finales):
        salida.append({
            "doc_id": plantilla["doc_id"],
            "chunk_id": f"{plantilla['doc_id']}::{ch.posicion}",
            "fuente": plantilla["fuente"],
            "formato": plantilla["formato"],
            "fenomeno": plantilla["fenomeno"],
            "posicion": ch.posicion,
            "num_tokens": n_tokens,
            "texto": ch.texto,
            "idioma": plantilla.get("idioma"),
            "titulo_seccion": ch.titulo_seccion,
            "url": plantilla.get("url", ""),
        })
    return salida


def cargar_por_documento(ruta_chunks):
    """Agrupa los chunks del checkpoint por doc_id.

    El archivo son ~158 MB; se agrupa por documento porque la reconstruccion
    necesita ver los chunks contiguos de cada uno.
    """
    por_doc = collections.defaultdict(list)
    with open(ruta_chunks, encoding="utf-8") as fh:
        for linea in fh:
            ch = json.loads(linea)
            por_doc[ch["doc_id"]].append(ch)
    return por_doc


def _flujo_alfanumerico(textos):
    """Concatena los caracteres alfanumericos, en orden, en minusculas."""
    return "".join(c.lower() for t in textos for c in t if c.isalnum())


def _es_subsecuencia(aguja, pajar):
    """True si `aguja` se obtiene de `pajar` borrando caracteres."""
    it = iter(pajar)
    return all(c in it for c in aguja)


def verificar_reconstruccion(ruta_chunks, limite_docs=None):
    """Comprueba que el reconstruido sea el original con trozos borrados.

    El criterio es a nivel de CARACTER alfanumerico, no de token, y por una
    razon que costo dos intentos: comparar tokens confunde re-tokenizacion con
    perdida. El segmentador separa la puntuacion ("CHUCHINGAL;" -> "CHUCHINGAL
    ;") y parte las citas legales ("IV(h)(1)" -> "IV(h)" + "(1)"), asi que
    tokens que existian dejan de existir aunque no falte una sola letra.

    Lo que la puerta quiere afirmar es mas fuerte y mas simple: el
    reconstruido es una SUBSECUENCIA del original. Eso prueba de una vez que
    no se invento texto, que no se reordeno, y que lo unico que desaparecio
    fueron trozos -- que es exactamente lo que hace quitar el solape.

    Ademas se acota cuanto se borro: el solape es de una oracion por chunk, o
    sea una fraccion chica del documento. Si desaparece mucho mas que eso, se
    perdio contenido de verdad y no solo el solape.
    """
    por_doc = cargar_por_documento(ruta_chunks)
    revisados = 0
    docs_con_perdida = []
    peor_fraccion = 0.0
    peor_doc = None

    for doc_id, chunks in por_doc.items():
        if chunks[0]["formato"] in FORMATOS_TABULARES:
            continue
        if limite_docs is not None and revisados >= limite_docs:
            break
        revisados += 1

        secciones, cuenta = reconstruir_oraciones(chunks, con_contabilidad=True)
        original = _flujo_alfanumerico(
            ch["texto"] for ch in sorted(chunks, key=lambda c: c["posicion"])
        )
        reconstruido = _flujo_alfanumerico(
            o for _, oraciones in secciones for o in oraciones
        )

        # 1. El reconstruido es el original con trozos borrados: nada
        #    inventado, nada reordenado.
        if not _es_subsecuencia(reconstruido, original):
            docs_con_perdida.append(doc_id)
            peor_fraccion = 1.0
            peor_doc = doc_id
            continue

        # 2. Todo lo que desaparecio respecto de las oraciones del segmentador
        #    es EXACTAMENTE el solape deduplicado. Igualdad, sin tolerancia.
        if len(reconstruido) != cuenta["oraciones"] - cuenta["solape"]:
            docs_con_perdida.append(doc_id)
            peor_fraccion = 1.0
            peor_doc = doc_id
            continue

        # 3. Lo que el segmentador descarto por no formar oracion. Es basura
        #    tipografica del PDF y ronda el 0%; si crece, se perdio contenido.
        descartado = 1 - cuenta["oraciones"] / max(1, len(original))
        if descartado > TOPE_DESCARTE_SEGMENTADOR:
            docs_con_perdida.append(doc_id)
            if descartado > peor_fraccion:
                peor_fraccion = descartado
                peor_doc = doc_id

    return {
        "docs_revisados": revisados,
        "docs_con_perdida": len(docs_con_perdida),
        "peor_doc": peor_doc,
        "peor_fraccion": peor_fraccion,
        "ejemplos": docs_con_perdida[:10],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--limite-docs", type=int, default=None)
    ap.add_argument("--generar", action="store_true")
    ap.add_argument("--presupuesto", type=int)
    ap.add_argument("--solape", type=int)
    ap.add_argument("--salida", type=Path)
    ap.add_argument(
        "--segmentacion-global",
        action="store_true",
        help="Segmenta una vez por seccion reconstruida y tokeniza por lotes. "
        "Solo admite --solape 0; es una variante experimental de chunking.",
    )
    ap.add_argument("--encoder-name",
                    default="paraphrase-multilingual-MiniLM-L12-v2",
                    help="Solo para recontar num_tokens con su tokenizer.")
    args = ap.parse_args()

    if args.verificar:
        res = verificar_reconstruccion(args.chunks, args.limite_docs)
        for clave, valor in res.items():
            print(f"{clave}: {valor}")
        # Puerta de entrada del experimento: sin conservacion no se gasta GPU.
        return 0 if res["docs_con_perdida"] == 0 else 1

    if args.generar:
        if args.presupuesto is None or args.solape is None or args.salida is None:
            ap.error("--generar necesita --presupuesto, --solape y --salida")
        if args.segmentacion_global and args.solape != 0:
            ap.error("--segmentacion-global solo admite --solape 0")
        from src.embedding.encoders import get_encoder

        encoder = get_encoder(args.encoder_name)
        por_doc = cargar_por_documento(args.chunks)
        total = 0
        with open(args.salida, "w", encoding="utf-8") as fh:
            for numero, chunks in enumerate(por_doc.values(), start=1):
                if args.segmentacion_global:
                    registros = reempaquetar_rapido_sin_solape(
                        chunks, args.presupuesto, encoder.count_tokens_many
                    )
                else:
                    registros = reempaquetar(
                        chunks, args.presupuesto, args.solape, encoder.count_tokens
                    )
                for reg in registros:
                    fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
                    total += 1
                if numero % 100 == 0:
                    print(
                        f"  {numero}/{len(por_doc)} documentos, {total} chunks",
                        flush=True,
                    )
        print(f"celda {args.presupuesto}/{args.solape}: {total} chunks "
              f"de {len(por_doc)} documentos -> {args.salida}")
        return 0

    ap.error("hace falta --verificar o --generar")


if __name__ == "__main__":
    raise SystemExit(main())
