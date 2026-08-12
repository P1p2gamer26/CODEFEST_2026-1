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

Con `--con-similitudes` se vuelca ADEMAS, para cada candidato, **el coseno de
cada encoder por separado** sobre los `PROF` candidatos crudos del primario.
Eso descompone el score entregado

    score = cos_minilm + 0.60*cos_gte + 0.60*cos_e5

en sus sumandos, y con ellos **cualquier variante de arquitectura de la
cascada se puede recalcular en memoria, sin FAISS y sin GPU**: pesos
asimetricos, recortes entre etapas, disparos condicionales. Son 50 x 200 x 3
numeros, unos pocos MB. Sin esto, cada variante cuesta ~1 GB de RAM y compite
con las corridas largas de la GPU.

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

# Encoders que NO estan en la entrega y cuyo indice vive en otro sitio. Sus
# cosenos se vuelcan igual con --con-similitudes, para poder medir variantes
# de cascada que los incluyan sin volver a cargar FAISS. Su peso al volcar es
# CERO: no participan en el score entregado, solo se registran.
EXTRA_DIRS = {"bge-m3": DEV_DIR / "intermedios" / "bge" / "encoder_bge-m3"}
PESO = 0.60
PROF = 200
K_POOL = 100

CAMPOS = ("chunk_id", "doc_id", "fuente", "texto", "formato", "fenomeno", "idioma", "fila")


def volcar(consultas, salida, con_similitudes=False):
    enc_p = get_encoder(name=MINILM)
    idx_p, metadata = load_index(MINILM)

    pools = {}
    for c in consultas:
        hits = search(expandir_consulta(c["text"]), enc_p, idx_p, metadata, k=PROF)[:PROF]
        pools[c["query_id"]] = hits
    del idx_p, metadata

    # El coseno del primario ES el score con el que salen de `search`, antes de
    # que ningun secundario lo toque. Hay que guardarlo AHORA.
    sims = {c["query_id"]: {MINILM: [h.score for h in pools[c["query_id"]]]} for c in consultas}

    for sec in SECUNDARIOS:
        idx_s, _ = load_index(sec)
        enc_s = get_encoder(name=sec)
        for c in consultas:
            qv = enc_s.encode_query(expandir_consulta(c["text"]))
            propios = []
            for h in pools[c["query_id"]]:
                cos = float(np.dot(qv, idx_s.reconstruct(h.fila)))
                propios.append(cos)
                h.score += PESO * cos
            sims[c["query_id"]][sec] = propios
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

    if con_similitudes:
        # Encoders de fuera de la entrega: se registran sus cosenos con peso 0,
        # o sea sin tocar el score entregado.
        for nombre, ruta in EXTRA_DIRS.items():
            if not (ruta / "index.faiss").is_file():
                print(f"  (sin indice para {nombre}, se omite)")
                continue
            idx_x, _ = load_index(nombre, index_dir=ruta)
            enc_x = get_encoder(name=nombre)
            for c in consultas:
                qv = enc_x.encode_query(expandir_consulta(c["text"]))
                sims[c["query_id"]][nombre] = [
                    float(np.dot(qv, idx_x.reconstruct(h.fila))) for h in pools[c["query_id"]]
                ]
            del idx_x
            print(f"  similitudes de {nombre} volcadas")

        # Los PROF candidatos CRUDOS del primario, en su orden original, con el
        # coseno de cada encoder. Es lo que permite recalcular cualquier
        # arquitectura de cascada sin volver a tocar FAISS.
        out["crudos"] = {
            c["query_id"]: [
                {"fila": h.fila, "chunk_id": h.chunk_id, "doc_id": h.doc_id}
                for h in pools[c["query_id"]]
            ]
            for c in consultas
        }
        out["similitudes"] = sims

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
    ap.add_argument(
        "--con-similitudes",
        action="store_true",
        help="Vuelca ademas el coseno de cada encoder por candidato, lo que permite "
        "recalcular cualquier arquitectura de cascada sin cargar FAISS.",
    )
    args = ap.parse_args()

    consultas = [json.loads(l) for l in CONSULTAS.read_text(encoding="utf-8").splitlines() if l.strip()]
    volcar(consultas, args.salida, con_similitudes=args.con_similitudes)


if __name__ == "__main__":
    main()
