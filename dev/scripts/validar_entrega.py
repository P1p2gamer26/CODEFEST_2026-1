#!/usr/bin/env python3
"""Valida la carpeta Entrega/ contra TODO lo que exige la especificacion.

Correr SIEMPRE antes de mandar la entrega. La especificacion es explicita en
que una estructura o un formato incorrectos llevan a penalizacion severa o a
quedar fuera de la evaluacion, asi que estas comprobaciones son el ultimo
filtro antes de entregar.

Cubre:
  1. Estructura de carpetas y archivos exigidos (sec. 1.4).
  2. Esquema estricto de resultados.jsonl (sec. 9.3): 3 documentos, 10
     fragmentos, <=250 palabras, campos y rangos de rank.
  3. Trazabilidad: cada chunk_id/doc_id de resultados.jsonl existe en la
     metadata, y la metadata esta alineada con el indice FAISS (sec. 5.3).
  4. Campos obligatorios de metadata por fragmento (Tabla 1).
  5. Informe tecnico presente y dentro del limite de 8 paginas.

Uso:
    python scripts/validar_entrega.py
    python scripts/validar_entrega.py --esperar-50   # exige exactamente 50 consultas
"""

import argparse
import json
import re
import sys
from pathlib import Path

# doc_id de respaldo generado por src/ingestion/doc_id.py::compute_doc_id().
_ES_HASH_CONTENIDO = re.compile(r"[0-9a-f]{16}")

ROOT = Path(__file__).resolve().parents[2]  # dev/scripts/ -> dev/ -> raiz del repo
ENTREGA = ROOT / "Entrega"

CAMPOS_METADATA_OBLIGATORIOS = {
    "doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto",
}
TIPOS_TABLA_1 = {
    "doc_id": str, "chunk_id": str, "fuente": str, "formato": str,
    "fenomeno": int, "posicion": int, "num_tokens": int, "texto": str,
}
# Cierres validos de oracion, incluidos los de cita y parentesis.
FIN_DE_ORACION = (".", "?", "!", '"', "'", ")", "]", "»", "…", ":")
# Cierre de oracion + numero de nota al pie, que el extractor de PDF pega al
# final del parrafo. Es texto completo, no un corte a media oracion.
_LLAMADA_DE_NOTA = re.compile(r"[.?!\"»)\]]\s*\d{1,3}$")
MANIFEST = ROOT / "dev" / "intermedios" / "doc_id_manifest.csv"
MAX_PALABRAS = 250
N_DOCS = 3
N_FRAGS = 10

problemas: list[str] = []
avisos: list[str] = []


def fallo(msg: str) -> None:
    problemas.append(msg)


def aviso(msg: str) -> None:
    avisos.append(msg)


def validar_estructura() -> list[Path]:
    """Sec. 1.4: archivos y carpetas exigidos. Devuelve las carpetas de encoder."""
    for nombre in ("resultados.jsonl", "generador.py", "informe_tecnico.pdf"):
        if not (ENTREGA / nombre).is_file():
            fallo(f"falta Entrega/{nombre} (exigido por la sec. 1.4)")

    base = ENTREGA / "base_vectorial"
    if not base.is_dir():
        fallo("falta Entrega/base_vectorial/")
        return []

    encoder_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("encoder_"))
    if not encoder_dirs:
        fallo("no hay ninguna carpeta base_vectorial/encoder_<nombre>/")

    for d in encoder_dirs:
        for archivo in ("index.faiss", "metadata.jsonl"):
            if not (d / archivo).is_file():
                fallo(f"falta {d.relative_to(ENTREGA)}/{archivo}")

    if not (base / "grafo" / "grafo.graphml").is_file():
        aviso("sin grafo/grafo.graphml: se pierde el puntaje del bonus (sec. 7)")

    # Archivos que NO deberian estar. La sec. 1.4 dibuja una estructura exacta
    # y avisa que incumplirla es penalizacion severa o exclusion. Encontrados
    # a mano el 2 ago 2026: un Entrega/__pycache__/ que dejo una corrida, y
    # los .gz que se habian generado para publicar el Release y quedaron
    # dentro de la carpeta de entrega. Ninguno lo detectaba este validador.
    ESPERADOS_RAIZ = {"resultados.jsonl", "generador.py", "informe_tecnico.pdf", "base_vectorial"}
    for hijo in ENTREGA.iterdir():
        if hijo.name not in ESPERADOS_RAIZ:
            fallo(f"sobra Entrega/{hijo.name}: la sec. 1.4 fija la estructura exacta")

    for d in encoder_dirs:
        for hijo in d.iterdir():
            if hijo.name not in ("index.faiss", "metadata.jsonl"):
                fallo(f"sobra {d.relative_to(ENTREGA)}/{hijo.name}")
    grafo_dir = base / "grafo"
    if grafo_dir.is_dir():
        for hijo in grafo_dir.iterdir():
            if hijo.name != "grafo.graphml":
                fallo(f"sobra grafo/{hijo.name} (los .gz van al Release, no a la entrega)")
    for hijo in base.iterdir():
        if hijo.name != "grafo" and not hijo.name.startswith("encoder_"):
            fallo(f"sobra base_vectorial/{hijo.name}")

    return encoder_dirs


def validar_grafo(doc_ids: set[str]) -> None:
    """El grafo tiene que hablar del MISMO corpus que el indice.

    Que el archivo exista no basta: un grafo.graphml sobrante de otra corrida
    referencia documentos que no estan indexados, y como bonus cuenta en
    contra en vez de a favor. Ademas, con --use-graph inyectaria en la fusion
    chunk_id inexistentes. Este chequeo existe porque justamente eso paso: la
    entrega llevaba el grafo del corpus sintetico viejo y la validacion la
    daba por buena."""
    grafo = ENTREGA / "base_vectorial" / "grafo" / "grafo.graphml"
    if not grafo.is_file():
        return

    try:
        import networkx as nx

        g = nx.read_graphml(grafo)
    except Exception as exc:  # noqa: BLE001 -- cualquier fallo de lectura invalida el bonus
        fallo(f"grafo/grafo.graphml no se puede leer ({exc})")
        return

    citados = {d["doc_id"] for _, _, d in g.edges(data=True) if "doc_id" in d}
    if not citados:
        aviso("el grafo no cita ningun doc_id: no es trazable al corpus (sec. 7.3)")
        return

    ajenos = citados - doc_ids
    if ajenos:
        fallo(
            f"el grafo cita {len(ajenos)} de {len(citados)} doc_id que NO estan indexados "
            f"(ej. {sorted(ajenos)[:3]}): quedo de otra corrida, hay que reconstruirlo "
            f"con 'build_corpus_index.py --solo-grafo' o sacarlo de la entrega"
        )
    else:
        print(
            f"  grafo de conocimiento (bonus): {g.number_of_nodes()} nodos, "
            f"{g.number_of_edges()} aristas, {len(citados)} doc_id todos indexados"
        )


def cargar_metadata(encoder_dir: Path) -> list[dict]:
    filas = []
    with (encoder_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError as e:
                fallo(f"{encoder_dir.name}/metadata.jsonl:{n}: JSON invalido ({e})")
    return filas


def validar_metadata(encoder_dir: Path, filas: list[dict]) -> None:
    """Tabla 1 + sec. 5.3 (alineacion con el indice FAISS)."""
    for i, fila in enumerate(filas[:], start=1):
        faltantes = CAMPOS_METADATA_OBLIGATORIOS - set(fila)
        if faltantes:
            fallo(f"{encoder_dir.name}/metadata.jsonl:{i}: faltan campos {sorted(faltantes)}")
            break  # un solo reporte por archivo, no 50k lineas

    ids = [f.get("chunk_id") for f in filas]
    if len(set(ids)) != len(ids):
        fallo(f"{encoder_dir.name}/metadata.jsonl: hay chunk_id duplicados")

    # Cobertura del manifest de ADL: un doc_id con forma de hash de contenido
    # significa que ese documento NO estaba en el manifest y por tanto no puede
    # emparejar con el ground truth (F1@3 = 0 para ese documento).
    docs = {f.get("doc_id") for f in filas}
    hashes = {d for d in docs if d and _ES_HASH_CONTENIDO.fullmatch(d)}
    if hashes:
        fallo(
            f"{encoder_dir.name}: {len(hashes)} de {len(docs)} doc_id son hashes de "
            f"contenido, no doc_id de ADL (p. ej. {sorted(hashes)[0]}). Falta pasar "
            f"--doc-id-manifest o el manifest no cubre esos archivos; no puntuan en F1@3."
        )
    else:
        print(f"  {encoder_dir.name}: {len(docs)} documentos, todos con doc_id de ADL")

    try:
        import faiss

        index = faiss.read_index(str(encoder_dir / "index.faiss"))
        # Filas `en_indice: false` son documentos sin texto (imagenes del
        # manifest): no tienen vector y van al final. La alineacion se exige
        # contra las filas con vector, y solo si son las primeras.
        con_vector = [f for f in filas if f.get("en_indice") is not False]
        desalineadas = index.ntotal != len(con_vector) or filas[: index.ntotal] != con_vector
        if desalineadas:
            fallo(
                f"{encoder_dir.name}: indice y metadata DESALINEADOS "
                f"(index.ntotal={index.ntotal} vs {len(filas)} lineas). "
                f"Rompe la trazabilidad exigida en la sec. 5.3."
            )
        else:
            print(f"  {encoder_dir.name}: {index.ntotal} vectores alineados con metadata")
    except ImportError:
        aviso("faiss no instalado: no se pudo verificar la alineacion indice/metadata")


def validar_tipos_metadata(filas: list[dict]) -> list[str]:
    """Tipos de la Tabla 1 (sec. 3.4).

    `validar_metadata` solo mira que los campos ESTEN. Un `fenomeno` que
    viaje como cadena, o un `posicion` como texto, cumple la presencia y
    rompe cualquier consumidor que espere el tipo declarado en la tabla.
    """
    errores: list[str] = []
    for n, fila in enumerate(filas, start=1):
        for campo, tipo in TIPOS_TABLA_1.items():
            valor = fila.get(campo)
            # bool es subclase de int en Python: hay que excluirlo a mano.
            if not isinstance(valor, tipo) or isinstance(valor, bool):
                errores.append(
                    f"linea {n}: {campo} es {type(valor).__name__}, "
                    f"se esperaba {tipo.__name__}"
                )
        if fila.get("fenomeno") not in (1, 2, 3):
            errores.append(f"linea {n}: fenomeno fuera de {{1, 2, 3}}")
        if len(errores) > 20:  # no volcar 128.526 lineas
            errores.append("... (mas errores, truncado)")
            return errores
    return errores


def validar_completitud_linguistica(objetos: list[dict]) -> list[str]:
    """Sec. 3.3: ningun fragmento con oraciones cortadas.

    Es AVISO y no fallo a proposito: el texto de OCR llega sin puntuacion y
    `truncate.py` corta por palabras como ultimo recurso, asi que un final
    sin punto no siempre es un corte a media oracion.
    """
    avisos_: list[str] = []
    for obj in objetos:
        for frag in obj.get("fragments", []):
            texto = (frag.get("text") or "").rstrip()
            if not texto or texto.endswith(FIN_DE_ORACION):
                continue
            # "...del tratado.65" es una oracion COMPLETA con la llamada a una
            # nota al pie pegada por el extractor de PDF, no un corte.
            if _LLAMADA_DE_NOTA.search(texto):
                continue
            avisos_.append(
                f"{obj.get('query_id')} frag#{frag.get('rank')}: "
                f"termina en '...{texto[-30:]}'"
            )
    return avisos_


def validar_python39(ruta_generador: Path) -> list[str]:
    """Q&A final: ADL evalua con Python >= 3.9.5.

    Una anotacion `X | None` sin `from __future__ import annotations` se
    evalua al definir la funcion y revienta en el import, antes de leer una
    consulta; la sec. 1.4 excluye de la evaluacion lo que no se reproduce.
    El venv local corre 3.13 y no lo detectaria nunca.
    """
    import ast

    if not ruta_generador.is_file():
        return []
    fuente = ruta_generador.read_text(encoding="utf-8")
    try:
        arbol = ast.parse(fuente, feature_version=(3, 9))
    except SyntaxError as exc:
        return [f"{ruta_generador.name}: sintaxis posterior a Python 3.9 ({exc})"]

    # `X | None` es gramatica valida en 3.9 -- ast.parse no lo rechaza. Lo que
    # falla es EVALUARLO, y las anotaciones se evaluan al definir la funcion
    # salvo que el modulo traiga `from __future__ import annotations`.
    pospuestas = any(
        isinstance(n, ast.ImportFrom) and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in arbol.body
    )
    if pospuestas:
        return []

    anotaciones = [
        n.annotation
        for n in ast.walk(arbol)
        if isinstance(n, (ast.arg, ast.AnnAssign)) and n.annotation is not None
    ] + [
        n.returns for n in ast.walk(arbol)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns
    ]
    uniones = [
        a for a in anotaciones
        for sub in ast.walk(a)
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr)
    ]
    if uniones:
        return [
            f"{ruta_generador.name}: {len(uniones)} anotacion(es) usan `X | Y` "
            f"(PEP 604, 3.10+) sin `from __future__ import annotations`; en "
            f"Python 3.9 revientan en el import (linea {uniones[0].lineno})"
        ]
    return []


def validar_cobertura_corpus(doc_ids: set[str], ruta_manifest: Path) -> list[str]:
    """Cuantos doc_id del manifest de ADL quedaron fuera del indice.

    Aviso, no fallo: hay documentos del manifest sin texto extraible
    (imagenes de SWF, un JSON vacio, dos "PDF" que son HTML de error).
    """
    if not ruta_manifest.is_file():
        return []
    import csv

    with ruta_manifest.open(encoding="utf-8", newline="") as f:
        del_manifest = {fila["doc_id"] for fila in csv.DictReader(f) if fila.get("doc_id")}
    faltan = sorted(del_manifest - doc_ids)
    if faltan:
        return [
            f"{len(faltan)} de {len(del_manifest)} doc_id del manifest no estan "
            f"indexados: {', '.join(faltan[:10])}"
            + (" ..." if len(faltan) > 10 else "")
        ]
    return []


def validar_resultados(chunk_ids: set[str], doc_ids: set[str], esperar_50: bool) -> None:
    """Sec. 9.3: esquema estricto + trazabilidad contra la metadata."""
    path = ENTREGA / "resultados.jsonl"
    if not path.is_file():
        return

    objetos = []
    with path.open(encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                objetos.append((n, json.loads(linea)))
            except json.JSONDecodeError as e:
                fallo(f"resultados.jsonl:{n}: JSON invalido ({e})")

    if esperar_50 and len(objetos) != 50:
        fallo(f"resultados.jsonl tiene {len(objetos)} lineas; la sec. 10.3 exige exactamente 50")
    elif len(objetos) != 50:
        aviso(
            f"resultados.jsonl tiene {len(objetos)} lineas (la entrega final debe tener 50, "
            f"q001-q050; ahora son las consultas de prueba)"
        )

    vistos = set()
    for n, obj in objetos:
        qid = obj.get("query_id")
        if not qid:
            fallo(f"resultados.jsonl:{n}: falta query_id")
            continue
        if qid in vistos:
            fallo(f"resultados.jsonl:{n}: query_id duplicado ({qid})")
        vistos.add(qid)

        docs = obj.get("documents")
        if not isinstance(docs, list) or len(docs) != N_DOCS:
            fallo(f"resultados.jsonl:{n} ({qid}): se exigen exactamente {N_DOCS} documentos")
        else:
            if [d.get("rank") for d in docs] != list(range(1, N_DOCS + 1)):
                fallo(f"resultados.jsonl:{n} ({qid}): los rank de documents deben ser 1..{N_DOCS}")
            for d in docs:
                if doc_ids and d.get("doc_id") not in doc_ids:
                    fallo(
                        f"resultados.jsonl:{n} ({qid}): doc_id {d.get('doc_id')!r} no existe en "
                        f"la metadata -- rompe la trazabilidad de la evaluacion"
                    )

        frags = obj.get("fragments")
        if not isinstance(frags, list) or len(frags) != N_FRAGS:
            fallo(f"resultados.jsonl:{n} ({qid}): se exigen exactamente {N_FRAGS} fragmentos")
            continue
        if [f.get("rank") for f in frags] != list(range(1, N_FRAGS + 1)):
            fallo(f"resultados.jsonl:{n} ({qid}): los rank de fragments deben ser 1..{N_FRAGS}")
        for f in frags:
            for campo in ("rank", "chunk_id", "doc_id", "text"):
                if campo not in f:
                    fallo(f"resultados.jsonl:{n} ({qid}): fragmento sin campo '{campo}'")
            texto = f.get("text", "")
            n_palabras = len(texto.split())
            if n_palabras > MAX_PALABRAS:
                fallo(
                    f"resultados.jsonl:{n} ({qid}): fragmento de {n_palabras} palabras, "
                    f"excede el limite de {MAX_PALABRAS} (sec. 9.2)"
                )
            if chunk_ids and f.get("chunk_id") not in chunk_ids:
                fallo(
                    f"resultados.jsonl:{n} ({qid}): chunk_id {f.get('chunk_id')!r} no existe "
                    f"en la metadata -- rompe la trazabilidad"
                )

    # La sec. 10.3 exige los ids q001..q050, en ese orden. Mientras se usen las
    # consultas de prueba provisionales esto es solo un aviso.
    esperados = [f"q{i:03d}" for i in range(1, 51)]
    reales = [obj.get("query_id") for _, obj in objetos]
    if reales and reales != esperados:
        mensaje = (
            f"los query_id no son q001..q050 en orden (primero: {reales[0]!r}). "
            f"La sec. 10.3 lo exige para la entrega final."
        )
        (fallo if esperar_50 else aviso)(mensaje)


def validar_informe() -> None:
    path = ENTREGA / "informe_tecnico.pdf"
    if not path.is_file():
        return
    try:
        import pypdfium2

        paginas = len(pypdfium2.PdfDocument(path))
        if paginas > 8:
            fallo(f"informe_tecnico.pdf tiene {paginas} paginas; el maximo es 8 (sec. 1.4)")
        else:
            print(f"  informe_tecnico.pdf: {paginas} paginas (limite 8)")
    except ImportError:
        aviso("pypdfium2 no instalado: no se pudo contar las paginas del informe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--esperar-50",
        action="store_true",
        help="Exige exactamente 50 consultas (usar cuando ya se corrio con las oficiales q001-q050).",
    )
    args = parser.parse_args()

    print("Validando Entrega/ contra la especificacion...\n")

    encoder_dirs = validar_estructura()

    chunk_ids: set[str] = set()
    doc_ids: set[str] = set()
    for d in encoder_dirs:
        filas = cargar_metadata(d)
        validar_metadata(d, filas)
        for e in validar_tipos_metadata(filas):
            fallo(f"{d.name}/metadata.jsonl: {e}")
        if not chunk_ids:  # trazabilidad contra el primer encoder
            chunk_ids = {f.get("chunk_id") for f in filas}
            doc_ids = {f.get("doc_id") for f in filas}

    validar_resultados(chunk_ids, doc_ids, args.esperar_50)
    validar_grafo(doc_ids)
    validar_informe()

    # Compatibilidad con el interprete del evaluador (Q&A: Python >= 3.9.5).
    for p_ in validar_python39(ENTREGA / "generador.py"):
        fallo(p_)

    # Los dos ultimos son AVISO: hay documentos del manifest sin texto
    # extraible, y el OCR entrega fragmentos sin puntuacion final.
    for a in validar_cobertura_corpus(doc_ids, MANIFEST):
        aviso(a)
    ruta_resultados = ENTREGA / "resultados.jsonl"
    if ruta_resultados.is_file():
        objetos = [json.loads(l) for l in ruta_resultados.open(encoding="utf-8") if l.strip()]
        cortados = validar_completitud_linguistica(objetos)
        if cortados:
            aviso(
                f"{len(cortados)} de {sum(len(o.get('fragments', [])) for o in objetos)} "
                f"fragmentos no terminan en cierre de oracion (sec. 3.3); "
                f"p. ej. {cortados[0]}"
            )

    print()
    for a in avisos:
        print(f"  AVISO: {a}")
    if problemas:
        print(f"\n{len(problemas)} PROBLEMA(S) que pueden costar la evaluacion:\n")
        for p_ in problemas:
            print(f"  - {p_}")
        sys.exit(1)

    print("\nEntrega valida: estructura, esquema y trazabilidad correctos.")


if __name__ == "__main__":
    main()
