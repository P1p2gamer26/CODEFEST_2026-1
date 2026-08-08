#!/usr/bin/env python3
"""Interfaz grafica de escritorio tipo chat (Tkinter, stdlib -- sin
dependencias nuevas): escribes cualquier consulta en una caja de texto y el
sistema responde con los 3 documentos + 10 fragmentos mas relevantes (sec. 9
de la especificacion). Un panel aparte, siempre visible, muestra la actividad
en vivo (que se esta cargando, cuanto va tardando, cuantos tokens procesa el
encoder). No hay ningun modelo generativo detras -- lo que "responde" es la
recuperacion vectorial + FAISS + grafo, formateada para leerse comodo (sec.
8.3: nada de LLM en ninguna etapa).

No reemplaza `scripts/build_corpus_index.py` ni `Entrega/generador.py`: llama
a las mismas funciones reusables de `src/gui/runner.py`. Los comandos de CLI
en README.md siguen funcionando igual.

Uso:
    python scripts/gui_app.py
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

# Modo offline: evita que cada arranque consulte huggingface.co para revalidar
# un modelo que ya esta en disco (son varios segundos y falla sin red). Tiene
# que quedar antes de importar cualquier cosa que arrastre transformers, o sea
# antes de importar `runner` mas abajo -- no es un import fuera de sitio.
# Se activa solo si el cache existe: forzarlo en una maquina limpia convertia
# la primera descarga, que es legitima, en un error criptico de "modelo no
# encontrado en modo offline".
_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.path.insert(0, str(Path(__file__).resolve().parent))  # para importar eval_mini

from eval_mini import cargar_jsonl, f1, ndcg  # noqa: E402
from src.cleaning.lang_detect import detect_language  # noqa: E402
from src.config import CORPUS_DIR, ENCODER_PRIMARY_NAME, GRAFO_PATH  # noqa: E402
from src.gui.history import append_run, read_history  # noqa: E402
from src.gui.runner import RunSummary, Session, run_offline  # noqa: E402

DEV_DIR = Path(__file__).resolve().parent.parent
CONSULTAS_50_PATH = DEV_DIR / "consultas_prueba" / "consultas_50_oficiales.jsonl"
GROUND_TRUTH_PATH = DEV_DIR / "eval" / "ground_truth_mini.jsonl"

MIN_SCORE_CONFIABLE = 0.35  # similitud coseno; por debajo de esto, la consulta probablemente
# no tiene relacion con el corpus -- umbral elegido a ojo con paraphrase-multilingual-MiniLM,
# no es un valor oficial del reto (el reto no exige umbral de confianza, sec. 8.7 lo permite
# como post-filtro opcional). Ajustar aqui si se cambia de encoder.


def _formatear_respuesta(
    resultado: dict, best_score: float, idioma_query: str | None, idioma_fragmento: str | None
) -> str:
    """Texto legible de un resultado (sec. 9) para mostrar como burbuja de
    'respuesta' en el chat: destaca el fragmento #1 como LA respuesta, y deja
    el resto (9 fragmentos + 3 documentos) como evidencia de apoyo, plegada.

    Esto es solo una decision de presentacion del chat -- no cambia el
    esquema oficial de `Entrega/resultados.jsonl` (Sec. 9.2 del PDF exige
    devolver los 10 fragmentos y 3 documentos completos, porque eso es lo que
    evalua NDCG@10/F1@3; aqui solo se resalta cual es el mas relevante de
    esos 10 para que se lea como una respuesta, no como un volcado de datos)."""
    lineas = []
    if best_score < MIN_SCORE_CONFIABLE:
        lineas.append(
            f"[Aviso: similitud baja ({best_score:.2f}). Esto no es un chatbot general -- solo "
            "busca en el corpus de los 3 fenomenos (IA+defensa, espacio/LEO, dinamicas territoriales "
            "LatAm). Tu consulta probablemente no tiene relacion con esos temas; lo de abajo es el "
            "fragmento MENOS distinto que encontro, no una respuesta confiable.]"
        )
        lineas.append("")

    top_fragmento = resultado["fragments"][0]
    if idioma_fragmento and idioma_query and idioma_fragmento != idioma_query:
        lineas.append(
            f"[Nota: tu consulta parece estar en '{idioma_query}' pero el fragmento mas parecido "
            f"esta en '{idioma_fragmento}' -- el encoder es multilingue a proposito (busca por "
            "significado, no por idioma exacto, sec. 4.3 del PDF), pero si la similitud no es alta "
            "esto puede ser una coincidencia debil por palabras sueltas mas que un match real.]"
        )
        lineas.append("")
    lineas.append(f"Respuesta (del documento {top_fragmento['doc_id']}, idioma: {idioma_fragmento or '?'}):")
    lineas.append(top_fragmento["text"].strip())
    lineas.append("")

    otros_docs = [d["doc_id"] for d in resultado["documents"] if d["doc_id"] != top_fragmento["doc_id"]]
    if otros_docs:
        lineas.append(f"Tambien relevante: {', '.join(otros_docs)}")

    lineas.append(
        f"(mostrando el fragmento mas relevante de 10, y {len(resultado['documents'])} documentos "
        "en total -- ver Entrega/resultados.jsonl para la lista completa que exige la entrega)"
    )
    return "\n".join(lineas)


class ActivityPanel(ttk.LabelFrame):
    """Panel APARTE (no modal) que muestra que esta ocurriendo en todo
    momento: carga del modelo/indice, avance de una corrida en lote, tiempo y
    tokens. Se alimenta via una Queue thread-safe drenada con `after()`."""

    def __init__(self, master):
        super().__init__(master, text="Actividad", padding=8)
        self._queue: queue.Queue = queue.Queue()
        self._start_time = time.time()

        top = ttk.Frame(self)
        top.pack(fill="x")
        self.lbl_estado = ttk.Label(top, text="Iniciando...")
        self.lbl_estado.pack(side="left")
        self.lbl_tiempo = ttk.Label(top, text="")
        self.lbl_tiempo.pack(side="right")

        self.log = ScrolledText(self, state="disabled", wrap="word", height=10)
        self.log.pack(fill="both", expand=True, pady=(6, 0))

        self._tick()

    def push(self, texto: str, estado: str | None = None) -> None:
        """Llamado desde CUALQUIER hilo -- solo encola, nunca toca widgets."""
        self._queue.put((texto, estado))

    def _tick(self) -> None:
        try:
            while True:
                texto, estado = self._queue.get_nowait()
                self._append(texto)
                if estado is not None:
                    self.lbl_estado.configure(text=estado)
        except queue.Empty:
            pass
        self.after(200, self._tick)

    def _append(self, texto: str) -> None:
        self.log.configure(state="normal")
        marca = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{marca}] {texto}\n")
        self.log.see("end")
        self.log.configure(state="disabled")


class MetricsPanel(ttk.LabelFrame):
    """F1@3 y NDCG@10 en vivo contra el mini ground truth de `dev/eval/`, con
    las mismas funciones que usa `scripts/eval_mini.py` (no se reimplementan:
    dos copias de una metrica terminan divergiendo).

    Dos avisos que van en pantalla y no solo aca: el NDCG@10 hereda la
    relevancia del documento (aproximacion, no estima la nota de ADL), y las 9
    consultas etiquetadas por el panel de agentes se cuentan APARTE porque no
    son comparables con las humanas."""

    def __init__(self, master):
        super().__init__(master, text="Metricas (mini ground truth)", padding=8)
        self.gt: dict[str, tuple[set, str]] = {}
        if GROUND_TRUTH_PATH.exists():
            for fila in cargar_jsonl(GROUND_TRUTH_PATH):
                # docs_relevantes vacio = se miro y no habia nada relevante en
                # el pool; eval_mini la excluye del promedio y aqui igual.
                if fila["docs_relevantes"]:
                    origen = "agente" if fila.get("anotador") else "humano"
                    self.gt[fila["query_id"]] = (set(fila["docs_relevantes"]), origen)

        self.acumulado: dict[str, list[tuple[float, float]]] = {}

        self.lbl_ultima = ttk.Label(self, text="ultima consulta: --")
        self.lbl_ultima.pack(anchor="w")
        self.lbl_humano = ttk.Label(self, text="humanas: --", font=("TkDefaultFont", 10, "bold"))
        self.lbl_humano.pack(anchor="w", pady=(6, 0))
        self.lbl_agente = ttk.Label(self, text="agente: --")
        self.lbl_agente.pack(anchor="w")
        ttk.Label(
            self,
            text="NDCG@10 aproximado: la relevancia del fragmento se hereda de su\n"
            "documento. Sirve para comparar configuraciones, no para estimar la nota.",
            foreground="#8a8a8a",
            font=("TkDefaultFont", 8),
        ).pack(anchor="w", pady=(6, 0))

        if not self.gt:
            self.lbl_ultima.configure(text=f"sin ground truth en {GROUND_TRUTH_PATH.name}")

    def reiniciar(self) -> None:
        self.acumulado.clear()
        self.lbl_ultima.configure(text="ultima consulta: --")
        self._refrescar()

    def registrar(self, resultado: dict) -> None:
        """Solo puntua las consultas que estan en el ground truth; las del chat
        libre no tienen respuesta correcta conocida."""
        entrada = self.gt.get(resultado["query_id"])
        if entrada is None:
            self.lbl_ultima.configure(text=f"ultima consulta: {resultado['query_id']} (sin anotar)")
            return
        relevantes, origen = entrada
        valor_f1 = f1([d["doc_id"] for d in resultado["documents"][:3]], relevantes)[2]
        valor_ndcg = ndcg(resultado.get("fragments", []), relevantes)
        self.acumulado.setdefault(origen, []).append((valor_f1, valor_ndcg))
        self.lbl_ultima.configure(
            text=f"ultima consulta: {resultado['query_id']}  F1@3 {valor_f1:.2f}  NDCG@10 {valor_ndcg:.2f}"
        )
        self._refrescar()

    def _refrescar(self) -> None:
        for origen, label in (("humano", self.lbl_humano), ("agente", self.lbl_agente)):
            vs = self.acumulado.get(origen, [])
            if not vs:
                label.configure(text=f"{origen}: --")
                continue
            label.configure(
                text=f"{origen}: {len(vs):2d} consultas   "
                f"F1@3 {sum(v[0] for v in vs) / len(vs):.3f}   "
                f"NDCG@10 {sum(v[1] for v in vs) / len(vs):.3f}"
            )


class ChatPanel(ttk.Frame):
    def __init__(self, master, on_send, on_run_50):
        super().__init__(master, padding=8)

        self.log = ScrolledText(self, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("user", foreground="#1c5cab", spacing1=6, spacing3=2)
        self.log.tag_configure("sistema", foreground="#2b2b2b", spacing1=2, spacing3=10)
        self.log.tag_configure("meta", foreground="#8a8a8a", font=("TkDefaultFont", 8))

        entrada = ttk.Frame(self)
        entrada.pack(fill="x", pady=(8, 0))
        self.entry = ttk.Entry(entrada)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self._enviar())
        ttk.Button(entrada, text="Enviar", command=self._enviar).pack(side="left", padx=(6, 0))

        botones = ttk.Frame(self)
        botones.pack(fill="x", pady=(6, 0))
        ttk.Button(
            botones, text="Correr las 50 consultas oficiales", command=on_run_50
        ).pack(side="left")

        self._on_send = on_send

    def _enviar(self):
        texto = self.entry.get().strip()
        if not texto:
            return
        self.entry.delete(0, "end")
        self.add_user_message(texto)
        self._on_send(texto)

    def add_user_message(self, texto: str) -> None:
        self._append(f"Tu: {texto}", "user")

    def add_system_message(self, texto: str, meta: str | None = None) -> None:
        self._append(f"Sistema:\n{texto}", "sistema")
        if meta:
            self._append(meta, "meta")

    def _append(self, texto: str, tag: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", texto + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CODEFEST Ad Astra 2026 -- Chat de recuperacion")
        self.geometry("1080x640")

        self.session: Session | None = None

        header = ttk.Frame(self, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(header, text=f"Corpus: {CORPUS_DIR}").pack(side="left")
        self.lbl_encoders = ttk.Label(header, text=f"   |   Encoders: {ENCODER_PRIMARY_NAME} (cargando...)")
        self.lbl_encoders.pack(side="left")
        ttk.Button(header, text="Reconstruir indice (offline)", command=self._on_offline).pack(
            side="right", padx=(6, 0)
        )
        ttk.Button(header, text="Ver historial", command=self._on_historial).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.chat = ChatPanel(body, on_send=self._on_send, on_run_50=self._on_run_50)
        lateral = ttk.Frame(body)
        self.metrics = MetricsPanel(lateral)
        self.metrics.pack(fill="x")
        self.activity = ActivityPanel(lateral)
        self.activity.pack(fill="both", expand=True, pady=(8, 0))
        body.add(self.chat, weight=3)
        body.add(lateral, weight=2)

        self.chat.entry.configure(state="disabled")
        self._cargar_sesion_en_hilo()

    # -- carga perezosa del encoder + indice + grafo (una sola vez) ----------

    def _cargar_sesion_en_hilo(self):
        self.activity.push("Cargando encoder e indice (puede tardar unos segundos)...", "Cargando modelo...")

        def worker():
            try:
                usa_grafo = GRAFO_PATH.exists()
                session = Session(
                    use_graph=usa_grafo,
                    progress_cb=lambda e: self.activity.push(_formatear_evento_offline(e)),
                )
                self.after(0, self._sesion_lista, session, usa_grafo)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self.activity.push, f"ERROR cargando la sesion: {exc}", "Error")

        threading.Thread(target=worker, daemon=True).start()

    def _sesion_lista(self, session: Session, usa_grafo: bool):
        self.session = session
        cascada = " -> ".join([session.encoder.name] + [e.name for e, _ in session.reranks])
        self.activity.push(
            f"Listo. Indice con {session.index.ntotal} vectores, cascada {cascada}"
            + (", grafo cargado" if usa_grafo else " (sin grafo)"),
            "Listo",
        )
        self.lbl_encoders.configure(text=f"   |   Encoders: {cascada}")
        self.chat.entry.configure(state="normal")

    # -- chat interactivo -----------------------------------------------------

    def _on_send(self, texto: str):
        if self.session is None:
            self.chat.add_system_message("Todavia se esta cargando el indice, espera un momento...")
            return

        def worker():
            try:
                resultado, tokens, elapsed, best_score = self.session.ask(texto)
                idioma_query = detect_language(texto)
                idioma_fragmento = self.session.metadata_by_chunk_id.get(
                    resultado["fragments"][0]["chunk_id"], {}
                ).get("idioma")
                respuesta = _formatear_respuesta(resultado, best_score, idioma_query, idioma_fragmento)
                meta = f"({tokens} tokens, {elapsed:.2f}s, similitud {best_score:.2f}, encoder {self.session.encoder.name})"
                self.after(0, self.metrics.registrar, resultado)
                self.after(0, lambda: self.chat.add_system_message(respuesta, meta))
                self.activity.push(f"Consulta respondida: \"{texto[:60]}\" -- {tokens} tokens, {elapsed:.2f}s")
                append_run(
                    RunSummary(
                        tipo="chat",
                        encoder=self.session.encoder.name,
                        detalle=texto,
                        n_items=1,
                        tokens_totales=tokens,
                        duracion_seg=elapsed,
                        estado="ok",
                        out_path="",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self.chat.add_system_message(f"ERROR: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    # -- correr las 50 consultas de prueba, una por una, en el mismo chat ----

    def _on_run_50(self):
        if self.session is None:
            self.chat.add_system_message("Todavia se esta cargando el indice, espera un momento...")
            return

        def worker():
            import json

            with CONSULTAS_50_PATH.open(encoding="utf-8") as f:
                consultas = [json.loads(line) for line in f if line.strip()]

            self.activity.push(f"Corriendo {len(consultas)} consultas de prueba...", "Corriendo lote de 50...")
            self.after(0, self.metrics.reiniciar)
            start = time.time()
            tokens_totales = 0
            for consulta in consultas:
                resultado, tokens, elapsed, best_score = self.session.ask(
                    consulta["text"], query_id=consulta["query_id"]
                )
                tokens_totales += tokens
                idioma_query = detect_language(consulta["text"])
                idioma_fragmento = self.session.metadata_by_chunk_id.get(
                    resultado["fragments"][0]["chunk_id"], {}
                ).get("idioma")
                respuesta = _formatear_respuesta(resultado, best_score, idioma_query, idioma_fragmento)
                meta = f"({consulta['query_id']}, {tokens} tokens, {elapsed:.2f}s, similitud {best_score:.2f})"
                self.after(0, lambda t=consulta["text"], r=respuesta, m=meta: (
                    self.chat.add_user_message(t),
                    self.chat.add_system_message(r, m),
                ))
                self.after(0, self.metrics.registrar, resultado)
                self.activity.push(f"[{consulta['query_id']}] respondida -- {tokens} tokens")

            duracion = time.time() - start
            append_run(
                RunSummary(
                    tipo="chat_lote_50",
                    encoder=self.session.encoder.name,
                    detalle=str(CONSULTAS_50_PATH),
                    n_items=len(consultas),
                    tokens_totales=tokens_totales,
                    duracion_seg=duracion,
                    estado="ok",
                    out_path="",
                )
            )
            self.activity.push(f"Lote de 50 terminado en {duracion:.1f}s, {tokens_totales} tokens", "Listo")

        threading.Thread(target=worker, daemon=True).start()

    # -- reconstruir el indice offline (ventana de progreso reusa Activity) --

    def _on_offline(self):
        self.chat.entry.configure(state="disabled")
        self.activity.push("Reconstruyendo la base de conocimiento (offline)...", "Construyendo indice...")

        def worker():
            try:
                summary = run_offline(
                    corpus_dir=CORPUS_DIR,
                    encoder_name=ENCODER_PRIMARY_NAME,
                    with_graph=True,
                    progress_cb=lambda e: self.activity.push(_formatear_evento_offline(e)),
                )
                append_run(summary)
                self.activity.push(
                    f"Indice reconstruido: {summary.n_items} documentos, {summary.tokens_totales} tokens, "
                    f"{summary.duracion_seg:.1f}s",
                    "Listo",
                )
            except Exception as exc:  # noqa: BLE001
                self.activity.push(f"ERROR reconstruyendo el indice: {exc}", "Error")
            finally:
                # el indice cambio -- se recarga la sesion de chat
                self.session = None
                self._cargar_sesion_en_hilo()

        threading.Thread(target=worker, daemon=True).start()

    def _on_historial(self):
        win = tk.Toplevel(self)
        win.title("Historial de corridas")
        win.geometry("860x320")

        columnas = ("timestamp", "tipo", "encoder", "detalle", "n_items", "tokens_totales", "duracion_seg")
        tree = ttk.Treeview(win, columns=columnas, show="headings")
        encabezados = {
            "timestamp": "Fecha",
            "tipo": "Tipo",
            "encoder": "Encoder",
            "detalle": "Detalle",
            "n_items": "Items",
            "tokens_totales": "Tokens",
            "duracion_seg": "Duracion (s)",
        }
        anchos = {"detalle": 260}
        for col in columnas:
            tree.heading(col, text=encabezados[col])
            tree.column(col, width=anchos.get(col, 100), anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        for entry in read_history():
            detalle = entry.get("detalle", "")
            if len(detalle) > 60:
                detalle = detalle[:60] + "..."
            tree.insert(
                "",
                "end",
                values=(
                    entry["timestamp"][:19].replace("T", " "),
                    entry["tipo"],
                    entry["encoder"],
                    detalle,
                    entry["n_items"],
                    entry["tokens_totales"],
                    f"{entry['duracion_seg']:.1f}",
                ),
            )


def _formatear_evento_offline(evento: dict) -> str:
    tipo = evento.get("tipo")
    if tipo == "documento":
        return f"[doc] {evento['nombre']} -> {evento['n_chunks']} chunks (tokens: {evento['tokens_acumulados']})"
    if tipo == "error_documento":
        return f"[ERROR] {evento['nombre']}: {evento['mensaje']}"
    if tipo == "encoder_listo":
        return f"[encoder] {evento['nombre']} cargado como re-puntuador de la cascada"
    if tipo == "encoder_omitido":
        return f"[encoder] SIN {evento['nombre']} -- la cascada corre degradada: {evento['mensaje']}"
    if tipo == "grafo":
        return f"[grafo] {evento['nodos']} nodos, {evento['aristas']} aristas"
    if tipo == "inicio":
        return f"Iniciando fase {evento['fase']} con encoder {evento['encoder']}"
    return str(evento)


if __name__ == "__main__":
    App().mainloop()
