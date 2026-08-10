#!/usr/bin/env python3
"""Mapa entidad -> doc_id, extraido del grafo entregado por STREAMING (E30).

POR QUE STREAMING. `grafo.graphml` pesa 183 MB con 224.101 nodos y 754.876
aristas; `networkx.read_graphml` lo materializa entero en varios GB y esta
maquina tiene 8. Aca solo hace falta, por cada arista, sus dos extremos (que
son el TEXTO de la entidad, ver build_graph.py: `add_node(subject,
label=subject)`) y su `doc_id` (clave d2). `iterparse` con `elem.clear()` lo
resuelve con memoria acotada.

El resultado se cachea en `dev/intermedios/entidades_por_doc.json` para no
re-parsear en cada corrida del barrido.

    .venv/Scripts/python.exe dev/scripts/mapa_entidades_grafo.py
"""

import argparse
import json
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEV_DIR, ROOT_DIR  # noqa: E402

GRAFO = ROOT_DIR / "Entrega" / "base_vectorial" / "grafo" / "grafo.graphml"
CACHE = DEV_DIR / "intermedios" / "entidades_por_doc.json"

NS = "{http://graphml.graphdrawing.org/xmlns}"

# Entidades de menos de 3 caracteres no discriminan nada (siglas de una letra,
# restos de OCR). Fijado antes de medir.
MIN_LARGO = 3


def normalizar(texto: str) -> str:
    """Minusculas sin acentos, espacios colapsados. El grafo mezcla ES/EN/PT y
    la consulta llega en espanol; comparar sin normalizar perderia 'OTAN' vs
    'Otan' y 'Colombia' vs 'COLOMBIA'."""
    t = unicodedata.normalize("NFKD", texto.strip().lower())
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def construir(grafo: Path) -> dict[str, list[str]]:
    """Devuelve entidad normalizada -> lista de doc_id que la mencionan."""
    por_entidad: dict[str, set] = defaultdict(set)
    aristas = 0
    for _, elem in ET.iterparse(str(grafo), events=("end",)):
        if elem.tag != NS + "edge":
            if elem.tag == NS + "node":
                elem.clear()
            continue
        aristas += 1
        doc_id = ""
        for data in elem:
            if data.get("key") == "d2":
                doc_id = data.text or ""
                break
        if doc_id:
            for extremo in (elem.get("source"), elem.get("target")):
                ent = normalizar(extremo or "")
                if len(ent) >= MIN_LARGO:
                    por_entidad[ent].add(doc_id)
        elem.clear()
        if aristas % 200000 == 0:
            print(f"  {aristas:,} aristas, {len(por_entidad):,} entidades", flush=True)
    return {k: sorted(v) for k, v in por_entidad.items()}


def cargar(cache: Path = CACHE, grafo: Path = GRAFO) -> dict[str, set]:
    """Mapa entidad -> set(doc_id), construyendo el cache la primera vez."""
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(construir(grafo)), encoding="utf-8")
    return {k: set(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}


def _autocomprobacion() -> None:
    """El unico check que hace falta: que el streaming lea las mismas aristas
    que un parseo normal sobre un GraphML minimo con la forma del entregado."""
    import tempfile

    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        '<key id="d2" for="edge" attr.name="doc_id" attr.type="string"/>'
        '<graph edgedefault="directed">'
        '<node id="OTÁN"><data key="d0">OTÁN</data></node>'
        '<node id="Colombia"><data key="d0">Colombia</data></node>'
        '<node id="x"><data key="d0">x</data></node>'
        '<edge source="OTÁN" target="Colombia">'
        '<data key="d1">firmar</data><data key="d2">F1-X-001</data></edge>'
        '<edge source="Colombia" target="x">'
        '<data key="d1">r</data><data key="d2">F1-X-002</data></edge>'
        "</graph></graphml>"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.graphml"
        p.write_text(xml, encoding="utf-8")
        m = construir(p)
    assert m == {"otan": ["F1-X-001"], "colombia": ["F1-X-001", "F1-X-002"]}, m
    print("autocomprobacion ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grafo", type=Path, default=GRAFO)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--autocomprobacion", action="store_true")
    args = ap.parse_args()

    if args.autocomprobacion:
        _autocomprobacion()
        return
    if args.rehacer and args.cache.is_file():
        args.cache.unlink()
    m = cargar(args.cache, args.grafo)
    docs = {d for v in m.values() for d in v}
    print(f"{len(m):,} entidades, {len(docs):,} documentos -> {args.cache}")
    print(f"  {args.cache.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
