#!/usr/bin/env python3
"""Vuelca a disco el pool de candidatos de las 50 consultas, ya re-puntuado.

POR QUE EXISTE. Todo experimento de REORDENAMIENTO (E11, E12, E15, E22, E23...)
reordena candidatos que ya estan recuperados: no necesita vectores ni FAISS
para nada. Pero el arnes los reconstruia desde el indice en cada corrida, o sea
que cada barrido pagaba ~987 MB de RAM -- MiniLM 197 + gte 395 + e5 395 -- mas
los tres `metadata.jsonl` parseados en Python, para producir un reordenamiento
que no usa un solo vector. Con 8 GB eso deja fuera cualquier trabajo en
paralelo, y ya obligo a partir en dos fases el arnes de E17.

Volcando el pool una vez, esos experimentos corren con RAM cero.

QUE SE VUELCA: los `k_pool` candidatos por consulta que sobreviven al camino
ENTREGADO, con todos los campos que un reordenamiento puede necesitar. Es
decir, la salida de la cascada, no el pool crudo del primario: si se volcara
el crudo, cada experimento tendria que re-aplicar el re-rank y volveriamos a
necesitar los indices.

OJO AL REGENERARLO: el volcado congela la configuracion entregada. Si cambia
el primario, el peso, la profundidad, el glosario o `k_pool`, hay que
regenerarlo o los experimentos de orden estaran reordenando el pool de otro
sistema. Por eso se escribe la configuracion dentro del propio archivo y
`cargar_pools()` la devuelve para que quien lo use pueda verificarla.

    .venv/Scripts/python.exe dev/scripts/volcar_pools.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))

from src.config import DEV_DIR  # noqa: E402
from src.embedding.build_index import load_index  # noqa: E402
from src.embedding.encoders import get_encoder  # noqa: E402
from src.retrieval.glosario import expandir_consulta  # noqa: E402
from src.retrieval.search import search  # noqa: E402

CONSULTAS = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
SALIDA = DEV_DIR / "intermedios" / "pools_entregados.json"

MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
SECUNDARIOS = ("gte-multilingual-base", "multilingual-e5-base")
PESO = 0.60
PROF = 200
K_POOL = 100

CAMPOS = ("chunk_id", "doc_id", "fuente", "texto", "formato", "fenomeno", "idioma", "fila")


def volcar(consultas, salida):
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = load_index(MINILM)

    pools = {}
    for c in consultas:
        hits = search(expandir_consulta(c["text"]), enc_p, idx_p, metadata, k=PROF)[:PROF]
        pools[c["query_id"]] = hits
    del idx_p, metadata

    for sec in SECUNDARIOS:
        idx_s, _ = load_index(sec)
        enc_s = get_encoder(name=sec)
        for c in consultas:
            qv = enc_s.encode_query(expandir_consulta(c["text"]))
            for h in pools[c["query_id"]]:
                h.score += PESO * float(np.dot(qv, idx_s.reconstruct(h.fila)))
        del idx_s

    out = {
        "config": {
            "primario": MINILM, "secundarios": list(SECUNDARIOS),
            "peso": PESO, "profundidad": PROF, "k_pool": K_POOL, "glosario": True,
        },
        "pools": {},
    }
    for c in consultas:
        hits = sorted(pools[c["query_id"]], key=lambda h: -h.score)[:K_POOL]
        out["pools"][c["query_id"]] = [
            {**{k: getattr(h, k) for k in CAMPOS}, "score": h.score, "rank": i}
            for i, h in enumerate(hits, 1)
        ]

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in out["pools"].values())
    print(f"{len(out['pools'])} consultas, {n} candidatos -> {salida}")
    print(f"  {salida.stat().st_size / 1e6:.1f} MB")


def cargar_pools(path=SALIDA):
    """Devuelve (config, pools). Usar esto en vez de reconstruir desde FAISS."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["config"], d["pools"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path, default=SALIDA)
    args = ap.parse_args()

    consultas = [json.loads(l) for l in CONSULTAS.read_text(encoding="utf-8").splitlines() if l.strip()]
    volcar(consultas, args.salida)


if __name__ == "__main__":
    main()
