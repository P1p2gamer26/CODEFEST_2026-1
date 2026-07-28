"""Historial de corridas del pipeline (offline/online), persistido como JSON
Lines local -- vive en `intermedios/` (gitignorado, igual que el resto de
artefactos intermedios) porque es un registro de uso de la maquina de cada
integrante del equipo, no un entregable del reto.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..config import INTERMEDIOS_DIR
from .runner import RunSummary

HISTORY_PATH = INTERMEDIOS_DIR / "historial_ejecuciones.jsonl"


def append_run(summary: RunSummary, history_path: Path = HISTORY_PATH) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **asdict(summary)}
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_history(history_path: Path = HISTORY_PATH) -> list[dict]:
    if not history_path.exists():
        return []
    entries = []
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries
