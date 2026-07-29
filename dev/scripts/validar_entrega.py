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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # dev/scripts/ -> dev/ -> raiz del repo
ENTREGA = ROOT / "Entrega"

CAMPOS_METADATA_OBLIGATORIOS = {
    "doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto",
}
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

    grafo = base / "grafo" / "grafo.graphml"
    if grafo.is_file():
        print("  grafo de conocimiento (bonus) presente")
    else:
        aviso("sin grafo/grafo.graphml: se pierde el puntaje del bonus (sec. 7)")

    return encoder_dirs


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

    try:
        import faiss

        index = faiss.read_index(str(encoder_dir / "index.faiss"))
        if index.ntotal != len(filas):
            fallo(
                f"{encoder_dir.name}: indice y metadata DESALINEADOS "
                f"(index.ntotal={index.ntotal} vs {len(filas)} lineas). "
                f"Rompe la trazabilidad exigida en la sec. 5.3."
            )
        else:
            print(f"  {encoder_dir.name}: {index.ntotal} vectores alineados con metadata")
    except ImportError:
        aviso("faiss no instalado: no se pudo verificar la alineacion indice/metadata")


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
        if not chunk_ids:  # trazabilidad contra el primer encoder
            chunk_ids = {f.get("chunk_id") for f in filas}
            doc_ids = {f.get("doc_id") for f in filas}

    validar_resultados(chunk_ids, doc_ids, args.esperar_50)
    validar_informe()

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
