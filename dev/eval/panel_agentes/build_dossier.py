"""Dossier de candidatos para el panel de agentes anotadores.

A diferencia de candidatos.md (extracto de 300 chars del mejor chunk), aca
cada documento va con varios chunks: el panel juzga documentos y con 300
chars se marca de menos (limitacion ya documentada en dev/eval/README.md).

Mezcla consultas SIN anotar con consultas YA anotadas a mano, sin decir
cuales son cuales. Sin eso el panel no se puede calibrar y sus etiquetas no
valen como ground truth (ver dev/docs/PLAN_MAESTRO.md: la primera ronda asistida dio F1 0.28
contra el humano).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "dev")

from src.config import DEV_DIR
from src.embedding.build_index import load_index
from src.embedding.encoders import get_encoder
from src.retrieval.aggregate import aggregate_documents
from src.retrieval.bm25 import BM25
from src.retrieval.search import search

SALIDA = Path(sys.argv[1])
ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"
K_POOL = 200
N_DENSO = 10
N_LEXICO = 5
CHUNKS_POR_DOC = 3
CHARS_POR_CHUNK = 700

# Terminos para el rescate lexico. REEMPLAZAN la consulta (medido: una
# consulta larga en espanol ahoga el termino raro, ver dev/docs/PLAN_MAESTRO.md).
TERMINOS = {
    "q001": "CBRN chemical biological radiological nuclear",
    "q028": "on-orbit servicing rendezvous proximity operations Shijian",
    "q015": "semiconductor export controls chips",
    "q007": "international humanitarian law autonomous weapons accountability",
    "q008": "drone swarm targeting ISR",
    "q011": "barriers adoption military artificial intelligence",
    "q012": "training data operational lessons learned",
}

SIN_ANOTAR = ["q001", "q007", "q008", "q011", "q012", "q015", "q028", "q038", "q048"]
# Calibracion: anotadas a mano. Se toman las INDEPENDIENTES (sin campo pool),
# que son las unicas cuyo ground truth no lo propuso el propio recuperador.
CALIBRACION = ["q005", "q017", "q020", "q033", "q040", "q044"]


def cargar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    consultas = {c["query_id"]: c["text"] for c in cargar_jsonl(DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl")}
    encoder = get_encoder(name=ENCODER)
    index, metadata = load_index(ENCODER)
    bm25 = BM25(metadata)
    print(f"indice {index.ntotal} vectores, BM25 {len(bm25.postings)} terminos", file=sys.stderr)

    por_doc = {}
    for m in metadata:
        por_doc.setdefault(m["doc_id"], []).append(m)

    qids = SIN_ANOTAR + CALIBRACION
    random.Random(20260802).shuffle(qids)  # el panel no debe saber cual es cual

    lineas = [
        "# Dossier de anotacion",
        "",
        "Para cada consulta, los documentos candidatos con varios fragmentos de cada uno.",
        "Marca los RELEVANTES. Algunas consultas pueden no tener ninguno.",
        "",
    ]
    clave = {}
    for qid in qids:
        texto = consultas[qid]
        hits = search(texto, encoder, index, metadata, k=K_POOL)
        docs = [d.doc_id for d in aggregate_documents(hits, top_n=N_DENSO, strategy="sum")]
        lexicos = bm25.search(TERMINOS.get(qid, texto), k=K_POOL)
        for d in aggregate_documents(lexicos, top_n=N_DENSO + N_LEXICO, strategy="sum"):
            if d.doc_id not in docs and len(docs) < N_DENSO + N_LEXICO:
                docs.append(d.doc_id)

        mejor = {}
        for h in hits + lexicos:
            mejor.setdefault(h.doc_id, h.chunk_id)

        clave[qid] = docs
        lineas += [f"## {qid}", "", f"**Consulta:** {texto}", ""]
        for doc_id in docs:
            chunks = por_doc.get(doc_id, [])
            # el mejor chunk primero, luego sus vecinos, para dar contexto
            idx = next((i for i, c in enumerate(chunks) if c["chunk_id"] == mejor.get(doc_id)), 0)
            ventana = chunks[max(0, idx - 1) : max(0, idx - 1) + CHUNKS_POR_DOC]
            lineas += [f"### {doc_id} — {chunks[0]['fuente'] if chunks else '?'}"]
            for c in ventana:
                lineas.append("> " + " ".join(c["texto"].split())[:CHARS_POR_CHUNK] + "...")
                lineas.append("")
        lineas.append("")

    SALIDA.write_text("\n".join(lineas), encoding="utf-8")
    SALIDA.with_suffix(".clave.json").write_text(json.dumps(clave, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"escrito {SALIDA} ({len(qids)} consultas)", file=sys.stderr)


if __name__ == "__main__":
    main()
