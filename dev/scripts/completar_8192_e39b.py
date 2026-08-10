#!/usr/bin/env python3
"""E39b: completar la corrida a max_length=8192 SIN re-encodear 10.000 pares.

La corrida completa que lanzo el usuario murio en q041 con
CUBLAS_STATUS_EXECUTION_FAILED (m 1024 n 24240 k 4096: un batch de 24 pares
padded a ~1010 tokens, TDR del driver de la GTX 1650). Pero era innecesaria:
los pares <=512 tokens se tokenizan igual a 512 y a 8192, asi que sus scores
son identicos por construccion. Solo importan los pares que superan 512 tokens
(medido: 19 de 10.000, maximo 1010). Este script:

  1. identifica esos 19 pares con el tokenizer del cross-encoder,
  2. los re-escorea a max_length=8192 (batch chico, evita el TDR),
  3. corre una SONDA de determinismo: re-escorea hasta 3 pares cortos por
     consulta a 8192 y los compara contra el X512 -- debe dar ~1e-5,
  4. escribe X8192.npy = X512 con los 19 scores corregidos, y orden8192.json,
     que es lo que habria producido la corrida completa si hubiera terminado.

Luego fase 2 --max-length 8192 consume ese archivo como si fuera el run real.

Uso (GPU): .venv-cuda/Scripts/python.exe dev/scripts/completar_8192_e39b.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Entrega"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer  # noqa: E402

from generador import expandir_consulta  # noqa: E402
from barrido_cross_encoder import (  # noqa: E402
    DUMP,
    METADATA,
    CONSULTAS,
    MODEL_ID,
    SALIDA,
    cargar_candidatos,
    leer_metadata_necesaria,
)

MAX_LEN = 512
LIMITE = 8192


def main() -> None:
    dump = json.load(open(DUMP, encoding="utf-8"))
    metadata = leer_metadata_necesaria(dump)
    cands_por_q = cargar_candidatos(dump, metadata)

    consultas = []
    for line in open(CONSULTAS, encoding="utf-8"):
        consultas.append(json.loads(line))
    consultas_por_id = {c["query_id"]: c for c in consultas}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    X512 = np.load(SALIDA / "X.npy")
    orden = json.loads((SALIDA / "orden.json").read_text(encoding="utf-8"))
    assert X512.shape == (len(orden), len(dump["crudos"][orden[0]])), X512.shape

    # 1) pares largos + indices de sonda (pares cortos)
    largos: list[tuple[int, int, str]] = []   # (qi, columna, qid)
    sondas: list[tuple[int, int, str]] = []
    for qi, qid in enumerate(orden):
        exp = expandir_consulta(consultas_por_id[qid]["text"])
        n_short = 0
        for i, c in enumerate(cands_por_q[qid]):
            L = len(tok(exp, c["texto"], truncation=False)["input_ids"])
            if L > MAX_LEN:
                largos.append((qi, i, qid))
            elif n_short < 3:
                sondas.append((qi, i, qid))
                n_short += 1
    print(f"pares >512: {len(largos)} | sonda cortos: {len(sondas)}", flush=True)

    # 2) re-escorear en GPU a 8192
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    model = CrossEncoder(MODEL_ID, max_length=LIMITE, device="cuda")
    t0 = time.time()

    def reescorear(items):
        pares = [(expandir_consulta(consultas_por_id[qid]["text"]),
                  cands_por_q[qid][i]["texto"]) for qi, i, qid in items]
        x = model.predict(pares, apply_sigmoid=True, batch_size=4)
        return np.asarray(x, dtype="float32")

    X8 = X512.copy()
    reporte = []
    for k in range(0, len(largos), 24):
        bloque = largos[k:k + 24]
        vals = reescorear(bloque)
        for (qi, i, qid), v in zip(bloque, vals):
            reporte.append((qid, i, float(X512[qi, i]), float(v)))
            X8[qi, i] = v
    probe_vals = reescorear(sondas)
    print(f"re-score en {time.time() - t0:.1f} s", flush=True)

    # 3) guardar el X8192 "completo" + orden
    np.save(SALIDA / "X8192.npy", X8)
    (SALIDA / "orden8192.json").write_text(json.dumps(orden), encoding="utf-8")

    # 4) reporte
    with open(SALIDA / "e39b_reporte.txt", "w", encoding="utf-8") as f:
        f.write("pares largos re-escoreados a 8192 (qid, col, x512, x8192, delta):\n")
        for qid, i, a, b in sorted(reporte):
            f.write(f"  {qid} [{i:3d}] 512={a:.6f} 8192={b:.6f} d={b - a:+.6f}\n")
        max_probe = 0.0
        n_big = 0
        for (qi, i, qid), v in zip(sondas, probe_vals):
            d = abs(float(v) - float(X512[qi, i]))
            max_probe = max(max_probe, d)
            if d > 1e-4:
                n_big += 1
        f.write(f"\nSONDA determinismo (pares <=512, 8192 vs 512): max|d|={max_probe:.6f} "
                f"con |d|>1e-4: {n_big}/{len(sondas)}\n")
    print("X8192.npy y orden8192.json escritos. Reporte ->", SALIDA / "e39b_reporte.txt", flush=True)


if __name__ == "__main__":
    main()
