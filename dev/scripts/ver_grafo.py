#!/usr/bin/env python3
"""Inspecciona y dibuja el grafo de conocimiento (bonus, sec. 7).

Por defecto imprime un resumen legible en consola (entidades mas conectadas,
tipos de relacion, ejemplos con su evidencia textual). Con --plot lo dibuja.

Al dibujar se muestran solo las N entidades mas conectadas: con el corpus real
el grafo tendra miles de nodos y dibujarlos todos produce una maranha ilegible.

Uso:
    python dev/scripts/ver_grafo.py                    # resumen en consola
    python dev/scripts/ver_grafo.py --plot             # abre una ventana
    python dev/scripts/ver_grafo.py --plot --out g.png # guarda a PNG
    python dev/scripts/ver_grafo.py --plot --top 60    # mas entidades
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx  # noqa: E402

from src.config import GRAFO_PATH  # noqa: E402


def resumen(g: nx.Graph) -> None:
    print(f"\nGrafo: {g.number_of_nodes()} entidades, {g.number_of_edges()} relaciones")
    if g.number_of_nodes() == 0:
        return

    componentes = list(nx.connected_components(g.to_undirected()))
    print(f"Componentes conexas: {len(componentes)} (la mayor tiene {max(map(len, componentes))})")

    relaciones = Counter(d.get("relation", "?") for _, _, d in g.edges(data=True))
    print("\nTipos de relacion:")
    for rel, n in relaciones.most_common():
        print(f"  {n:5d}  {rel}")

    grados = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    print("\nEntidades mas conectadas:")
    for nodo, grado in grados[:15]:
        etiqueta = g.nodes[nodo].get("label", nodo)
        print(f"  {grado:3d}  {etiqueta}")

    print("\nEjemplos de relaciones (con su chunk de origen, para trazabilidad):")
    for u, v, d in list(g.edges(data=True))[:8]:
        lu = g.nodes[u].get("label", u)
        lv = g.nodes[v].get("label", v)
        print(f"  {lu} --[{d.get('relation', '?')}]--> {lv}   (chunk {d.get('chunk_id', '?')})")
    print()


def dibujar(g: nx.Graph, top: int, out: Path | None) -> None:
    import matplotlib.pyplot as plt

    # Solo las `top` entidades mas conectadas: con miles de nodos, dibujar
    # todo no comunica nada.
    if g.number_of_nodes() > top:
        mantener = [n for n, _ in sorted(g.degree(), key=lambda x: x[1], reverse=True)[:top]]
        g = g.subgraph(mantener).copy()
        print(f"mostrando las {top} entidades mas conectadas de las {g.number_of_nodes()}")

    grados = dict(g.degree())
    etiquetas = {n: g.nodes[n].get("label", n) for n in g.nodes}

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, k=0.6, seed=42)  # seed fija -> dibujo reproducible
    nx.draw_networkx_edges(g, pos, alpha=0.3, edge_color="#888888")
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=[120 + 90 * grados[n] for n in g.nodes],
        node_color=[grados[n] for n in g.nodes],
        cmap="viridis",
        alpha=0.85,
    )
    nx.draw_networkx_labels(g, pos, labels=etiquetas, font_size=8)
    plt.title(f"Grafo de conocimiento ({g.number_of_nodes()} entidades, {g.number_of_edges()} relaciones)")
    plt.axis("off")
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"guardado en: {out}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grafo", type=Path, default=GRAFO_PATH)
    parser.add_argument("--plot", action="store_true", help="Dibuja el grafo ademas del resumen.")
    parser.add_argument("--top", type=int, default=40, help="Entidades a dibujar (por grado).")
    parser.add_argument("--out", type=Path, default=None, help="Guarda el dibujo a PNG.")
    args = parser.parse_args()

    if not args.grafo.is_file():
        print(f"no existe {args.grafo}; construirlo con: "
              f"python dev/scripts/build_corpus_index.py --with-graph")
        sys.exit(1)

    g = nx.read_graphml(args.grafo)
    resumen(g)
    if args.plot:
        dibujar(g, args.top, args.out)


if __name__ == "__main__":
    main()
