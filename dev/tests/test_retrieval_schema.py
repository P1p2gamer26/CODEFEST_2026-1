"""Prueba de integracion end-to-end de Entrega/generador.py: construye un
indice sintetico minimo, corre el script real (via subprocess, tal como lo
correria un evaluador externo) y valida que resultados.jsonl cumpla
ESTRICTAMENTE el esquema de la seccion 9.3 -- esto es exactamente lo que
haria que la entrega real fuera penalizada o descartada si fallara
(sec. 9.3.2)."""

import json
import os
import subprocess
import sys
from pathlib import Path

DEV_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEV_DIR.parent
GENERADOR_PATH = REPO_ROOT / "Entrega" / "generador.py"

SAMPLE_TEXTS_BY_DOC = {
    "docA": [
        "Los satelites en orbita baja enfrentan riesgos crecientes de colision. "
        "La cooperacion internacional es clave para mitigar la congestion orbital.",
        "Las directrices de desintegracion post-mision buscan reducir la basura espacial.",
    ],
    "docB": [
        "La inteligencia artificial se aplica cada vez mas en logistica militar. "
        "Los ministerios de defensa han creado unidades especializadas en ciberseguridad.",
    ],
    "docC": [
        "El desarrollo territorial en America Latina muestra brechas persistentes entre regiones.",
    ],
    "docD": [
        "Space debris mitigation requires sustained international cooperation among operators.",
    ],
}


def _build_tiny_index(index_dir: Path) -> None:
    sys.path.insert(0, str(DEV_DIR))
    from src.embedding.build_index import build_and_persist
    from src.embedding.encoders import HashingFakeEncoder
    from src.ingestion.pipeline import ChunkRecord

    encoder = HashingFakeEncoder(name="schema-test-encoder")
    records = []
    for doc_id, textos in SAMPLE_TEXTS_BY_DOC.items():
        for pos, texto in enumerate(textos):
            records.append(
                ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-c{pos:04d}",
                    fuente=f"{doc_id}.pdf",
                    formato="pdf",
                    fenomeno=1,
                    posicion=pos,
                    num_tokens=len(texto.split()),
                    texto=texto,
                )
            )
    build_and_persist(records, encoder, out_dir=index_dir)


def test_generador_produces_schema_exact_output(tmp_path):
    index_dir = tmp_path / "encoder_schema-test-encoder"
    _build_tiny_index(index_dir)

    consultas_path = tmp_path / "consultas.jsonl"
    consultas_path.write_text(
        "\n".join(
            json.dumps(obj)
            for obj in [
                {"query_id": "q001", "text": "riesgos de colision en orbita baja"},
                {"query_id": "q002", "text": "inteligencia artificial en defensa"},
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "resultados.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(GENERADOR_PATH),
            "--consultas", str(consultas_path),
            "--use-fake-encoder",
            "--encoder-name", "schema-test-encoder",
            "--index-dir", str(index_dir),
            "--out", str(out_path),
        ],
        # cwd fuera del repo y PYTHONPATH vacio: si generador.py dependiera de
        # `src/` esto fallaria con ModuleNotFoundError, que es justamente lo que
        # le pasaria al evaluador al recibir solo la carpeta Entrega/.
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"generador.py fallo:\n{result.stdout}\n{result.stderr}"

    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, "una linea por consulta"

    seen_query_ids = []
    for line in lines:
        obj = json.loads(line)  # cada linea debe ser JSON valido por si sola

        assert set(obj.keys()) == {"query_id", "documents", "fragments"}
        seen_query_ids.append(obj["query_id"])

        assert len(obj["documents"]) == 3, "exactamente 3 documentos (sec. 9.2)"
        for i, doc in enumerate(obj["documents"], start=1):
            assert set(doc.keys()) == {"rank", "doc_id"}
            assert doc["rank"] == i

        assert len(obj["fragments"]) <= 10, "a lo sumo 10 fragmentos"
        assert len(obj["fragments"]) > 0
        for i, frag in enumerate(obj["fragments"], start=1):
            assert set(frag.keys()) == {"rank", "chunk_id", "doc_id", "text"}
            assert frag["rank"] == i
            assert len(frag["text"].split()) <= 250, "limite de 250 palabras (sec. 9.2)"

    assert seen_query_ids == ["q001", "q002"], "orden de las consultas preservado"


def test_generador_es_autocontenido():
    """`Entrega/` se entrega SOLA: generador.py no puede importar nada de
    `dev/src/` ni parchear sys.path para alcanzarlo. Es una copia aplanada del
    camino online y debe mantenerse asi."""
    codigo = GENERADOR_PATH.read_text(encoding="utf-8")
    for prohibido in ("from src.", "import src", "sys.path"):
        assert prohibido not in codigo, (
            f"generador.py contiene {prohibido!r}: dejo de ser autocontenido"
        )
