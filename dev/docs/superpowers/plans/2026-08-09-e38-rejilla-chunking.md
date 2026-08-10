# E38 — Rejilla de chunking: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir si un chunk más grande (384 o 512 tokens) y/o sin solape mejora el ranking de documentos frente al chunking entregado de 280 tokens con solape 1, y adoptar la celda ganadora solo si pasa el criterio y los tres vetos.

**Architecture:** Se reconstruye la secuencia de oraciones por documento a partir de `chunks_intermedios_limpio.jsonl` deduplicando el solape, se re-empaqueta con `_pack_sentences` a cada presupuesto, se construye un índice de MiniLM por celda en GPU, y se evalúa cada celda con el arnés existente. Solo la celda ganadora paga el build de los tres encoders.

**Tech Stack:** Python 3.13 en `.venv` (CPU, entrega) y `.venv-cuda` (GPU, torch 2.6+cu124), FAISS `IndexFlatIP`, sentence-transformers, pytest.

## Global Constraints

- **Nunca agregar `Co-Authored-By: Claude` ni atribución de IA a los commits.**
- Trabajar sobre la rama `Julian_Africano`, no sobre `main`.
- Comentarios y docstrings en español, **ASCII sin acentos** (convención de `dev/src/`). El texto de `docs/` sí lleva acentos.
- **Ningún modelo generativo en ningún punto** (sec. 8.3 del PDF).
- `Entrega/` no se toca hasta que una celda pase el criterio completo. Hoy reproduce byte a byte en frío, sha256 del archivo en disco `987293ac3cc4769bf859168b1279c3c8e1ea3fbdcbfb3acf3a3d7dccf154621b`.
- Todos los índices de prueba van a `dev/intermedios/`, **nunca** a `Entrega/base_vectorial/`. Usar `--out-base`.
- **Lanzar la GPU siempre con `.venv-cuda\Scripts\python.exe` y con `Start-Process` desacoplado.** Las corridas largas lanzadas desde una herramienta de fondo del agente mueren sin dejar traceback.
- **Nunca instalar `requirements.txt` completo en `.venv-cuda`** (fija `torch==2.13.0` y destruiría el cu124). Dependencias sueltas únicamente.
- Base a batir, verificada antes de mirar ninguna celda: 50 consultas **0.455 / 0.516 / 0.499**, 41 humanas **0.486 / 0.537 / 0.520**, 10 independientes **0.433 / 0.474 / 0.467**, **11 consultas con F1@3 = 0**, **0 fragmentos ilegibles**.

---

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `dev/scripts/rechunkear_e38.py` | Reconstruye oraciones desde los chunks y re-empaqueta a un presupuesto/solape dado. Una celda por corrida. |
| `dev/tests/test_rechunk_e38.py` | Cubre la reconstrucción del solape y la invariancia de la celda base. |
| `dev/scripts/barrido_e38.py` | Evalúa cada índice de celda y emite la tabla de nueve lecturas con IC al 90%. |
| `dev/experimentos/E38_rejilla_chunking.md` | Bitácora del experimento: pre-registro, tablas, veredicto. |

`dev/src/chunking/chunker.py` **no se modifica**: `chunk_document` y `_pack_sentences` ya reciben `token_budget` y `overlap_sentences` como parámetros.

---

### Task 1: Reconstruir la secuencia de oraciones deduplicando el solape

Esta es la tarea que hace viable todo el experimento. No hay checkpoint del texto crudo de los documentos, solo de los chunks ya fragmentados; re-extraer serían horas más OCR. Pero el solape es exactamente 1 oración, así que la secuencia original se recupera quitando de cada chunk las oraciones que repiten la cola del anterior.

**Files:**
- Create: `dev/scripts/rechunkear_e38.py`
- Test: `dev/tests/test_rechunk_e38.py`

**Interfaces:**
- Consumes: `src.chunking.sentence_split.split_sentences(text, lang)`, `src.chunking.chunker._pack_sentences(sentences, heading, posicion_inicial, count_tokens, token_budget, overlap_sentences)`
- Produces: `reconstruir_oraciones(chunks_del_doc: list[dict]) -> list[tuple[str | None, list[str]]]`, que devuelve una lista de `(titulo_seccion, oraciones)` en orden de `posicion`.

- [ ] **Step 1: Write the failing test**

```python
# dev/tests/test_rechunk_e38.py
from scripts.rechunkear_e38 import reconstruir_oraciones


def test_quita_la_oracion_de_solape_entre_chunks_contiguos():
    # Dos chunks de la misma seccion: el segundo repite la ultima oracion
    # del primero, que es exactamente lo que hace CHUNK_OVERLAP_SENTENCES=1.
    chunks = [
        {"posicion": 0, "titulo_seccion": "S1", "idioma": "es",
         "texto": "Alfa uno. Beta dos. Gamma tres."},
        {"posicion": 1, "titulo_seccion": "S1", "idioma": "es",
         "texto": "Gamma tres. Delta cuatro."},
    ]
    secciones = reconstruir_oraciones(chunks)
    assert len(secciones) == 1
    heading, oraciones = secciones[0]
    assert heading == "S1"
    assert oraciones == ["Alfa uno.", "Beta dos.", "Gamma tres.", "Delta cuatro."]


def test_no_deduplica_una_repeticion_legitima_no_contigua():
    # La misma oracion dos veces, pero NO en la frontera: es contenido real
    # del documento, no solape, y tiene que conservarse.
    chunks = [
        {"posicion": 0, "titulo_seccion": None, "idioma": "es",
         "texto": "Alfa uno. Beta dos."},
        {"posicion": 1, "titulo_seccion": None, "idioma": "es",
         "texto": "Beta dos. Alfa uno. Delta cuatro."},
    ]
    _, oraciones = reconstruir_oraciones(chunks)[0]
    assert oraciones == ["Alfa uno.", "Beta dos.", "Alfa uno.", "Delta cuatro."]


def test_separa_por_seccion():
    chunks = [
        {"posicion": 0, "titulo_seccion": "A", "idioma": "es", "texto": "Uno."},
        {"posicion": 1, "titulo_seccion": "B", "idioma": "es", "texto": "Dos."},
    ]
    secciones = reconstruir_oraciones(chunks)
    assert [h for h, _ in secciones] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest dev/tests/test_rechunk_e38.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'scripts.rechunkear_e38'`

- [ ] **Step 3: Write minimal implementation**

```python
# dev/scripts/rechunkear_e38.py  (fragmento: solo lo que el test exige)
"""E38 -- re-empaqueta el corpus ya extraido a otro presupuesto de tokens.

No re-extrae ni pasa OCR: reconstruye la secuencia de oraciones desde
`chunks_intermedios_limpio.jsonl` quitando el solape, y vuelve a empaquetar
con `_pack_sentences`. Ver el spec en
dev/docs/superpowers/specs/2026-08-09-rejilla-chunking-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.sentence_split import split_sentences


def reconstruir_oraciones(chunks_del_doc):
    """Devuelve [(titulo_seccion, [oraciones])] en orden de posicion.

    Quita de cada chunk el prefijo de oraciones que repite la cola del chunk
    anterior de la MISMA seccion. Solo mira la frontera: una repeticion en
    otro punto del texto es contenido real del documento.
    """
    secciones = []
    for ch in sorted(chunks_del_doc, key=lambda c: c["posicion"]):
        oraciones = split_sentences(ch["texto"], ch.get("idioma"))
        heading = ch.get("titulo_seccion")
        if secciones and secciones[-1][0] == heading:
            previas = secciones[-1][1]
            solape = 0
            tope = min(len(previas), len(oraciones))
            for n in range(tope, 0, -1):
                if previas[-n:] == oraciones[:n]:
                    solape = n
                    break
            previas.extend(oraciones[solape:])
        else:
            secciones.append((heading, list(oraciones)))
    return secciones
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest dev/tests/test_rechunk_e38.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dev/scripts/rechunkear_e38.py dev/tests/test_rechunk_e38.py
git commit -m "E38: reconstruir la secuencia de oraciones sin re-extraer el corpus"
```

---

### Task 2: Verificar la reconstrucción contra el corpus real antes de confiar en ella

La Task 1 asume que el solape es exactamente 1 oración y que `split_sentences` re-parte un chunk en las mismas oraciones con que se armó. **Las dos cosas pueden fallar en el corpus real** (el chunker descarta el solape en el caso patológico de oraciones larguísimas, y el segmentador puede derivar). Si la reconstrucción pierde o duplica texto, toda la rejilla mide sobre un corpus corrupto y no lo sabríamos.

**Files:**
- Modify: `dev/scripts/rechunkear_e38.py` (añadir el subcomando `--verificar`)
- Test: `dev/tests/test_rechunk_e38.py`

**Interfaces:**
- Produces: `verificar_reconstruccion(ruta_chunks: Path, limite_docs: int | None) -> dict` con las claves `docs`, `oraciones_reconstruidas`, `docs_con_perdida`, `peor_doc`.

- [ ] **Step 1: Write the failing test**

```python
def test_reconstruccion_conserva_toda_oracion_no_solapada():
    from scripts.rechunkear_e38 import reconstruir_oraciones
    chunks = [
        {"posicion": 0, "titulo_seccion": "S", "idioma": "es",
         "texto": "Uno uno. Dos dos. Tres tres."},
        {"posicion": 1, "titulo_seccion": "S", "idioma": "es",
         "texto": "Tres tres. Cuatro cuatro."},
    ]
    _, oraciones = reconstruir_oraciones(chunks)[0]
    unidas = " ".join(oraciones)
    for esperada in ["Uno uno.", "Dos dos.", "Tres tres.", "Cuatro cuatro."]:
        assert esperada in unidas
    assert unidas.count("Tres tres.") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest dev/tests/test_rechunk_e38.py::test_reconstruccion_conserva_toda_oracion_no_solapada -v`
Expected: PASS si la Task 1 quedó bien. **Si falla, arreglar `reconstruir_oraciones` antes de seguir.**

- [ ] **Step 3: Añadir el verificador sobre el corpus real**

```python
def verificar_reconstruccion(ruta_chunks, limite_docs=None):
    """Compara el texto reconstruido contra el original, documento a documento.

    El criterio es de CONSERVACION, no de identidad: el re-empaquetado puede
    cambiar donde cae cada corte, pero no puede perder ni inventar palabras.
    """
    import collections
    import json

    por_doc = collections.defaultdict(list)
    for linea in open(ruta_chunks, encoding="utf-8"):
        ch = json.loads(linea)
        if ch["formato"] in ("csv", "xlsx"):
            continue  # tabulares: se chunkean por filas, el presupuesto no aplica
        por_doc[ch["doc_id"]].append(ch)

    docs_con_perdida = []
    peor = (0.0, None)
    for i, (doc_id, chunks) in enumerate(por_doc.items()):
        if limite_docs is not None and i >= limite_docs:
            break
        secciones = reconstruir_oraciones(chunks)
        reconstruido = collections.Counter(
            " ".join(o for _, oraciones in secciones for o in oraciones).split()
        )
        # El original repite las palabras del solape; se cuenta el MAXIMO por
        # palabra entre chunks contiguos seria complejo, asi que el criterio es
        # que ninguna palabra del original falte del reconstruido.
        original = collections.Counter()
        for ch in chunks:
            original.update(ch["texto"].split())
        faltan = {p: n for p, n in original.items() if reconstruido[p] == 0}
        if faltan:
            frac = sum(faltan.values()) / max(1, sum(original.values()))
            docs_con_perdida.append(doc_id)
            if frac > peor[0]:
                peor = (frac, doc_id)

    return {
        "docs": min(len(por_doc), limite_docs or len(por_doc)),
        "docs_con_perdida": len(docs_con_perdida),
        "peor_doc": peor[1],
        "peor_fraccion": peor[0],
    }
```

- [ ] **Step 4: Correr el verificador sobre el corpus completo**

Run:
```bash
.venv/Scripts/python.exe dev/scripts/rechunkear_e38.py --verificar \
  --chunks dev/intermedios/chunks_intermedios_limpio.jsonl
```
Expected: `docs_con_perdida: 0`.
**Puerta de entrada: si hay documentos con pérdida, el experimento se detiene acá** y se documenta en la bitácora. Re-chunkear sobre un corpus que pierde texto invalidaría las seis celdas.

- [ ] **Step 5: Commit**

```bash
git add dev/scripts/rechunkear_e38.py dev/tests/test_rechunk_e38.py
git commit -m "E38: verificador de conservacion de texto en la reconstruccion"
```

---

### Task 3: Generar las cinco celdas de chunking

**Files:**
- Modify: `dev/scripts/rechunkear_e38.py` (subcomando de generación)
- Test: `dev/tests/test_rechunk_e38.py`

**Interfaces:**
- Produces: `dev/intermedios/chunks_e38_<presupuesto>_<solape>.jsonl` con el mismo esquema de `chunks_intermedios_limpio.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
def test_la_celda_base_reproduce_el_conteo_original(tmp_path):
    """280/1 sobre el corpus reconstruido debe dar casi los mismos chunks.

    Es el CONTROL del experimento: si la celda base no se parece al corpus
    entregado, la reconstruccion cambia algo y las otras cinco celdas no son
    comparables con la entrega.
    """
    from scripts.rechunkear_e38 import reempaquetar
    chunks = [
        {"doc_id": "D1", "posicion": 0, "titulo_seccion": None, "idioma": "es",
         "formato": "pdf", "fuente": "a.pdf", "fenomeno": 1, "url": "",
         "texto": "Uno uno uno. Dos dos dos. Tres tres tres."},
    ]
    salida = reempaquetar(chunks, token_budget=280, overlap_sentences=1,
                          count_tokens=lambda t: len(t.split()))
    assert len(salida) == 1
    assert salida[0]["doc_id"] == "D1"
    assert salida[0]["posicion"] == 0
    assert salida[0]["chunk_id"] == "D1::0"
    assert "Tres tres tres." in salida[0]["texto"]


def test_presupuesto_menor_produce_mas_chunks():
    from scripts.rechunkear_e38 import reempaquetar
    chunks = [
        {"doc_id": "D1", "posicion": 0, "titulo_seccion": None, "idioma": "es",
         "formato": "pdf", "fuente": "a.pdf", "fenomeno": 1, "url": "",
         "texto": "Uno uno uno. Dos dos dos. Tres tres tres. Cuatro cuatro."},
    ]
    ct = lambda t: len(t.split())
    grandes = reempaquetar(chunks, 280, 1, ct)
    chicos = reempaquetar(chunks, 4, 0, ct)
    assert len(chicos) > len(grandes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest dev/tests/test_rechunk_e38.py -v`
Expected: FAIL con `ImportError: cannot import name 'reempaquetar'`

- [ ] **Step 3: Implementar `reempaquetar`**

```python
def reempaquetar(chunks_del_doc, token_budget, overlap_sentences, count_tokens):
    """Re-empaqueta los chunks de UN documento al presupuesto pedido.

    El chunk_id se reconstruye como doc_id::posicion, que es la convencion
    vigente. OJO (punto 8 de CLAUDE.md): los chunk_id de esta celda NO son
    comparables con los de otra celda ni con los de la entrega.
    """
    from src.chunking.chunker import _pack_sentences

    plantilla = chunks_del_doc[0]
    if plantilla["formato"] in ("csv", "xlsx"):
        return [dict(c) for c in sorted(chunks_del_doc, key=lambda c: c["posicion"])]

    salida = []
    posicion = 0
    for heading, oraciones in reconstruir_oraciones(chunks_del_doc):
        nuevos = _pack_sentences(
            oraciones, heading, posicion, count_tokens, token_budget, overlap_sentences
        )
        for ch in nuevos:
            salida.append({
                "doc_id": plantilla["doc_id"],
                "chunk_id": f"{plantilla['doc_id']}::{ch.posicion}",
                "fuente": plantilla["fuente"],
                "formato": plantilla["formato"],
                "fenomeno": plantilla["fenomeno"],
                "posicion": ch.posicion,
                "num_tokens": ch.num_tokens,
                "texto": ch.texto,
                "idioma": plantilla.get("idioma"),
                "titulo_seccion": ch.titulo_seccion,
                "url": plantilla.get("url", ""),
            })
        posicion += len(nuevos)
    return salida
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest dev/tests/test_rechunk_e38.py -v`
Expected: 5 passed

- [ ] **Step 5: Generar las seis celdas y comparar conteos**

Run:
```bash
for pres in 280 384 512; do for sol in 0 1; do
  .venv/Scripts/python.exe dev/scripts/rechunkear_e38.py --generar \
    --chunks dev/intermedios/chunks_intermedios_limpio.jsonl \
    --presupuesto $pres --solape $sol \
    --encoder-name paraphrase-multilingual-MiniLM-L12-v2 \
    --salida dev/intermedios/chunks_e38_${pres}_${sol}.jsonl
done; done
```

**Control obligatorio:** la celda `280_1` debe quedar cerca de los **128.526** chunks del corpus entregado. Anotar el número exacto en la bitácora. Una desviación mayor al 2% significa que la reconstrucción altera el chunking y hay que entenderla antes de seguir.

- [ ] **Step 6: Commit**

```bash
git add dev/scripts/rechunkear_e38.py dev/tests/test_rechunk_e38.py
git commit -m "E38: generar las seis celdas de la rejilla de chunking"
```

---

### Task 4: Construir un índice de MiniLM por celda en GPU

**Files:**
- Ninguno nuevo. Se usa `dev/scripts/build_corpus_index.py --desde-chunks`.

- [ ] **Step 1: Lanzar la primera celda desacoplada de la sesión**

```powershell
$repo = "C:\Users\Julian\Downloads\CODEFEST_2026-1"
Start-Process -FilePath "$repo\.venv-cuda\Scripts\python.exe" `
  -ArgumentList "$repo\dev\scripts\build_corpus_index.py","--desde-chunks","$repo\dev\intermedios\chunks_e38_512_0.jsonl","--encoder-name","paraphrase-multilingual-MiniLM-L12-v2","--out-base","$repo\dev\intermedios\idx_e38_512_0" `
  -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$repo\dev\intermedios\e38_512_0.log" `
  -RedirectStandardError  "$repo\dev\intermedios\e38_512_0.err" -PassThru
```

Dos cosas al vigilarla: `-PassThru` devuelve el PID del envoltorio del venv, no el del worker (buscar el de CPU/RAM alta con `Get-Process python*`), y HuggingFace escribe su log de red en stderr, así que el `.err` con contenido no significa que falló.

- [ ] **Step 2: Verificar la celda al terminar**

Run:
```bash
.venv/Scripts/python.exe -c "
import faiss, json
ix = faiss.read_index('dev/intermedios/idx_e38_512_0/encoder_paraphrase-multilingual-MiniLM-L12-v2/index.faiss')
n = sum(1 for _ in open('dev/intermedios/idx_e38_512_0/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl', encoding='utf-8'))
print('vectores', ix.ntotal, 'metadata', n, 'ALINEADO' if ix.ntotal == n else 'DESALINEADO')
"
```
Expected: `ALINEADO`. Si no, la celda se descarta y se re-construye; un índice desalineado da resultados sin sentido en silencio.

- [ ] **Step 3: Repetir para las cinco celdas restantes**

Una por vez, en serie: la GTX 1650 tiene 4 GB y dos builds concurrentes se quedan sin VRAM. Celdas: `280_0`, `280_1`, `384_0`, `384_1`, `512_1`.

- [ ] **Step 4: Commit del registro (no de los índices)**

Los índices **no se commitean** (punto 12 de CLAUDE.md: no queda cuota de LFS). Solo la bitácora con los conteos.

```bash
git add dev/experimentos/E38_rejilla_chunking.md
git commit -m "E38: seis indices de MiniLM construidos, conteos de chunks por celda"
```

---

### Task 5: Evaluar la rejilla con las nueve lecturas

**Files:**
- Create: `dev/scripts/barrido_e38.py`
- Create: `dev/experimentos/E38_rejilla_chunking.md`

**Interfaces:**
- Consumes: `dev/scripts/eval_mini.py` (F1@3, NDCG@10, NDCG penalizado y el techo alcanzable), `Entrega/generador.py` con `--rerank-encoder none`.

- [ ] **Step 1: Medir la base de MiniLM solo bajo la configuración actual**

Antes de mirar ninguna celda. La base de la etapa 1 **no es** la cascada de 0.455/0.516: es MiniLM solo con el post-filtrado de E32 activo, sobre el corpus entregado. Se registra en la bitácora antes de construir la tabla.

Run:
```bash
.venv/Scripts/python.exe Entrega/generador.py \
  --consultas dev/consultas_prueba/consultas_50_oficiales.jsonl \
  --rerank-encoder none --out dev/intermedios/res_e38_base.jsonl
.venv/Scripts/python.exe dev/scripts/eval_mini.py --resultados dev/intermedios/res_e38_base.jsonl
```

- [ ] **Step 2: Medir cada celda con `k_pool` crudo y escalado**

`k_pool` se mide en chunks, no en volumen de texto. Valores escalados por `round(100 * 280 / presupuesto)`: **100** para 280, **73** para 384, **55** para 512. Cada celda se corre con los dos.

Run (una celda, los dos valores):
```bash
for kp in 100 55; do
  .venv/Scripts/python.exe Entrega/generador.py \
    --consultas dev/consultas_prueba/consultas_50_oficiales.jsonl \
    --rerank-encoder none --k-pool $kp \
    --base-vectorial dev/intermedios/idx_e38_512_0 \
    --out dev/intermedios/res_e38_512_0_kp${kp}.jsonl
  .venv/Scripts/python.exe dev/scripts/eval_mini.py \
    --resultados dev/intermedios/res_e38_512_0_kp${kp}.jsonl
done
```

Si `generador.py` no expone `--base-vectorial` o `--k-pool`, usar el arnés de barrido existente (`dev/scripts/barrido_pool_topm_e37.py` carga índices y llama a las funciones de `dev/src/`) como plantilla en vez de añadir flags a `generador.py`: **el punto 14 de CLAUDE.md exige que `generador.py` siga siendo autocontenido y reproduzca la entrega sin flags.**

- [ ] **Step 3: Emitir la tabla con IC al 90% del delta pareado**

Nueve lecturas por celda: F1@3, NDCG@10 y NDCG penalizado sobre las 50, las 41 humanas y las 10 independientes. Más las dos columnas de veto: **consultas con F1@3 = 0** y **fragmentos ilegibles**.

- [ ] **Step 4: Aplicar el criterio y los tres vetos**

Una celda se adopta solo si cumple **todo**:
1. IC al 90% del delta pareado excluye una pérdida de 0.02.
2. Confirma en las 41 humanas, no solo en las 50.
3. Consultas con F1@3 = 0 **≤ 11**.
4. Fragmentos ilegibles **= 0**.
5. La ganancia no se concentra en las 9 consultas de panel.

- [ ] **Step 5: Escribir el veredicto y commitear**

```bash
git add dev/scripts/barrido_e38.py dev/experimentos/E38_rejilla_chunking.md
git commit -m "E38: la rejilla de chunking, seis celdas y nueve lecturas"
```

---

### Task 6: Fase de cierre (SOLO si una celda pasa el criterio)

Si ninguna pasa, el experimento se cierra como negativo documentado y **`Entrega/` no se toca**. Esta tarea no se ejecuta.

- [ ] **Step 1: Construir los índices de gte y e5 sobre el chunkeo ganador**

Los tres índices de una celda deben salir del **mismo** chunkeo: distintos chunkeos hacen que el mismo `chunk_id` apunte a textos distintos y la cascada mezclaría fragmentos que no son el mismo (punto 8 de CLAUDE.md). `generador.py` valida los `chunk_id` al cargar y aborta con exit 2 si no coinciden.

gte son ~7,3 h de GPU. Lanzar con `Start-Process` desacoplado.

- [ ] **Step 2: Verificar alineación de los tres índices**

Run: `.venv/Scripts/python.exe dev/scripts/verificar_alineacion.py`
Expected: mismo `ntotal` en los tres y `chunk_id` idénticos entre ellos.

- [ ] **Step 3: Regenerar la entrega y validarla**

```bash
.venv/Scripts/python.exe Entrega/generador.py --consultas dev/consultas_prueba/consultas_50_oficiales.jsonl
.venv/Scripts/python.exe dev/scripts/validar_entrega.py
.venv/Scripts/python.exe -m pytest dev/tests -q
```
Expected: validador limpio y **143 tests o más** en verde.

- [ ] **Step 4: Corrida en frío desde fuera del repo**

```bash
cd /tmp/frio && PYTHONPATH= /c/Users/Julian/Downloads/CODEFEST_2026-1/.venv/Scripts/python.exe \
  /c/Users/Julian/Downloads/CODEFEST_2026-1/Entrega/generador.py \
  --consultas /c/Users/Julian/Downloads/CODEFEST_2026-1/dev/consultas_prueba/consultas_50_oficiales.jsonl \
  --out ./frio.jsonl
diff ./frio.jsonl /c/Users/Julian/Downloads/CODEFEST_2026-1/Entrega/resultados.jsonl && echo IDENTICO
```
Expected: `IDENTICO`. Es lo único que el evaluador ejecuta.

- [ ] **Step 5: Borrar `Entrega/__pycache__` y subir los índices al Release**

`__pycache__` se regenera cada vez que los tests importan `generador.py`, y el validador lo marca como estructura sobrante. Es el **último** paso, después de los tests.

```bash
rm -rf Entrega/__pycache__
gh release upload indices-v2 --clobber \
  Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/index.faiss#minilm-index.faiss
```

- [ ] **Step 6: Actualizar CLAUDE.md y commitear**

CLAUDE.md está desactualizado (dice 0.440/0.506 cuando E32 ya lo dejó en 0.455/0.516). Actualizarlo con el resultado de E38 y el sha256 nuevo de `resultados.jsonl`.

---

## Self-Review

**Cobertura del spec:** sec. 3.1 rejilla → Task 3; sec. 3.2 criba en dos etapas → Tasks 4 y 6; sec. 3.3 corrección de `k_pool` → Task 5 step 2; sec. 4.1 base verificada → Task 5 step 1; sec. 4.2 criterio → Task 5 step 4; sec. 4.3 vetos → Task 5 step 4; sec. 5 riesgos → Global Constraints y Task 2 (puerta de entrada) y Task 6 step 1 (invariante de `chunk_id`); sec. 6 cierre → Task 6 completa; sec. 7 fuera de alcance → no genera tareas, correcto.

**Riesgo no cubierto por el spec y añadido acá:** el spec no anticipaba que no hay checkpoint del texto crudo. Tasks 1 y 2 existen por eso, y la Task 2 es una **puerta de entrada** que puede matar el experimento antes de gastar GPU.

**Consistencia de tipos:** `reconstruir_oraciones` devuelve `list[tuple[str | None, list[str]]]` y `reempaquetar` la consume con ese desempaquetado en las tres tareas. `count_tokens` es `Callable[[str], int]` en las dos.
