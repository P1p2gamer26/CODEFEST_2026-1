# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial

Implementación completa de la Etapa 1 del reto: extracción, limpieza,
chunking, embeddings, índice FAISS, recuperación (con fusión multi-encoder y
grafo de conocimiento bonus) y generación de `resultados.jsonl` a partir de
consultas en lenguaje natural. Especificación completa en
`Material de apoyo/CODEFEST_2026-1.pdf`.

> Sin modelos generativos en ningún punto del pipeline (prohibido por la
> sec. 8.3 de la especificación). Todo el sistema es recuperación pura sobre
> vectores, FAISS y metadata.

## Índice

1. [Qué hay en este repo](#1-qué-hay-en-este-repo)
2. [Requisitos previos](#2-requisitos-previos)
3. [Instalación (Windows / macOS / Linux)](#3-instalación-windows--macos--linux)
4. [Cómo correr todo desde cero](#4-cómo-correr-todo-desde-cero)
5. [Interfaz gráfica (GUI)](#5-interfaz-gráfica-gui)
6. [Qué es provisional vs. infraestructura reusable](#6-qué-es-provisional-vs-infraestructura-reusable)
7. [Checklist de entregables frente a la especificación](#7-checklist-de-entregables-frente-a-la-especificación)
8. [Estructura del proyecto](#8-estructura-del-proyecto)
9. [Solución de problemas](#9-solución-de-problemas)

## 1. Qué hay en este repo

| Carpeta | Contenido |
|---|---|
| `src/` | Código reusable del pipeline (extracción, limpieza, chunking, embeddings, retrieval, grafo, GUI). No depende del corpus concreto. |
| `scripts/` | CLIs que orquestan `src/`: construir el índice, generar el corpus sintético, inspeccionar resultados, lanzar la GUI. |
| `Entrega/` | Estructura oficial de entrega (Sección 1.4 de la especificación): `generador.py`, `resultados.jsonl`, `informe_tecnico.pdf`, `base_vectorial/`. |
| `tests/` | Suite de pytest (31 pruebas) que valida la mecánica de cada etapa sin depender de red. |
| `corpus_ejemplo/` | Corpus **sintético** provisional (ver sección 6). |
| `consultas_prueba/` | Consultas de prueba **provisionales** (ver sección 6), no las oficiales q001–q050. |
| `docs/` | Documentación adicional (arquitectura de encoders). |
| `intermedios/` | Artefactos locales no versionados (historial de la GUI, índices de prueba). |

Ver la [sección 8](#8-estructura-del-proyecto) para el detalle módulo por módulo.

## 2. Requisitos previos

- **Python 3.11 o superior** (probado con 3.13). Verificar con:
  ```bash
  python --version   # o: python3 --version
  ```
- **Git** para clonar el repo.
- Acceso normal a internet la primera vez que se instalan dependencias y se
  descargan los pesos del encoder/modelos de spaCy (PyPI, HuggingFace). Un
  proxy corporativo restringido puede bloquear estas descargas — ver
  sección 9.
- Unos ~3 GB libres en disco (torch + transformers + modelos de spaCy).

No se necesita GPU: todo corre en CPU (`faiss-cpu`, `torch` CPU).

## 3. Instalación (Windows / macOS / Linux)

### 3.1 Clonar y entrar al proyecto

```bash
git clone <url-del-repo>
cd CODEFEST_2026-1
```

### 3.2 Crear el entorno virtual (una sola vez)

<details open>
<summary><b>Windows — PowerShell</b></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la ejecución de scripts (`no se puede cargar el
archivo ... porque la ejecución de scripts está deshabilitada`), correr una
vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

</details>

<details>
<summary><b>Windows — cmd.exe</b></summary>

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

</details>

<details>
<summary><b>Windows — Git Bash</b></summary>

```bash
python -m venv .venv
source .venv/Scripts/activate
```

</details>

<details>
<summary><b>macOS / Linux — bash / zsh</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```

</details>

**Importante:** repetir la activación en *cada* terminal nueva que abras. Si
el prompt no muestra `(.venv)` al inicio, no estás en el entorno correcto y
comandos como `pytest` o `python scripts/...` van a fallar con errores de
"módulo no encontrado" aunque ya hayas instalado todo.

### 3.3 Instalar dependencias

Mismo comando en las tres plataformas (con el venv activado):

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download pt_core_news_sm
```

Tarda varios minutos (torch, faiss, transformers pesan). Requiere acceso
normal a internet (PyPI); no funciona en entornos con proxy restringido.

### 3.4 Verificar que todo quedó bien instalado

```bash
pytest tests/ -v
```

Debería dar **`31 passed`**. Estos tests usan un encoder falso y determinista
(`HashingFakeEncoder`) solo para validar la mecánica del pipeline sin
depender de red ni de calidad semántica real — es normal y esperado que
corran sin conexión.

## 4. Cómo correr todo desde cero

El pipeline tiene dos fases (ver diagrama de flujo, Sección 6 de la
especificación): una **OFFLINE** (se corre una sola vez por corpus, es la
pesada) y una **ONLINE** (se corre por cada tanda de consultas, es rápida
porque reutiliza el índice ya construido).

### Paso 1 — Construir el índice (fase OFFLINE)

```bash
python scripts/build_corpus_index.py --with-graph
```

Toma `corpus_ejemplo/` (o `--corpus-dir` apuntando al corpus real), extrae
texto, limpia, fragmenta, codifica cada fragmento con el encoder real de
HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
se descarga solo la primera vez) y escribe:

- `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/index.faiss`
- `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl`
- `Entrega/base_vectorial/grafo/grafo.graphml` (bonus, por `--with-graph`)

Se corre una sola vez por corpus. Volver a correrlo sobrescribe el índice.

### Paso 2 — Generar resultados (fase ONLINE)

```bash
python Entrega/generador.py --consultas consultas_prueba/consultas_prueba.jsonl --use-graph
```

Lee el índice ya construido, busca cada consulta, agrega a nivel documento,
fusiona con el grafo si se pasó `--use-graph`, y escribe
`Entrega/resultados.jsonl` con el esquema oficial (Sección 9 de la
especificación: 3 documentos + 10 fragmentos ≤250 palabras por consulta). Es
rápido porque ya no recalcula embeddings del corpus, solo busca en el
índice ya construido.

### Paso 3 — Ver los resultados

Inspección cualitativa legible en consola (qué documentos/fragmentos trajo
cada consulta):

```bash
python scripts/inspect_results.py \
  --consultas consultas_prueba/consultas_prueba.jsonl \
  --resultados Entrega/resultados.jsonl \
  --index-dir Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2
```

Validar el esquema por línea de comandos (mismo comando en cualquier
plataforma porque es Python puro):

```bash
python -c "
import json
with open('Entrega/resultados.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        assert len(obj['documents']) == 3
        assert len(obj['fragments']) == 10
        assert all(len(fr['text'].split()) <= 250 for fr in obj['fragments'])
print('esquema OK')
"
```

### Paso 4 — Cuando cambien corpus/consultas (de sintéticos a los oficiales de ADL)

Repetir solo los pasos 1 y 2, apuntando a los archivos reales — nada de
`src/` necesita cambiar:

```bash
python scripts/build_corpus_index.py --with-graph --corpus-dir <corpus_real_de_ADL>
python Entrega/generador.py --consultas <consultas_oficiales_q001-q050.jsonl> --use-graph
```

### Demo alternativa sin red (solo para depurar la mecánica, no usar para la entrega)

```bash
python scripts/build_corpus_index.py --use-fake-encoder --with-graph \
  --out-dir intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --graph-out-path intermedios/demo_index_fake_encoder/grafo.graphml
python Entrega/generador.py --consultas consultas_prueba/consultas_prueba.jsonl \
  --use-fake-encoder \
  --index-dir intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --out intermedios/demo_index_fake_encoder/resultados_demo.jsonl
```

## 5. Interfaz gráfica (GUI)

Alternativa a los comandos de arriba para quien prefiera no usar la
terminal. Es una capa opcional sobre el mismo pipeline: llama exactamente a
las mismas funciones de `src/` que usan `scripts/build_corpus_index.py` y
`Entrega/generador.py` (ver `src/gui/runner.py`) — los comandos de CLI
documentados arriba siguen funcionando igual, uno no reemplaza al otro.

Requiere el venv activado (secciones 3.2–3.3). Sin dependencias nuevas: usa
`tkinter`, que viene incluido con la instalación estándar de Python en
Windows y macOS. En Linux puede requerir instalarlo aparte:

```bash
# Debian/Ubuntu
sudo apt install python3-tk
# Fedora
sudo dnf install python3-tkinter
```

Lanzar la GUI (igual en las tres plataformas):

```bash
python scripts/gui_app.py
```

Es una interfaz tipo chat, no un panel de botones con ventanas emergentes:
escribes cualquier consulta y el sistema responde con los 3 documentos + 10
fragmentos más relevantes, formateados como una burbuja de respuesta. No hay
ningún modelo generativo detrás (prohibido por la sec. 8.3): lo que
"responde" es la recuperación vectorial + FAISS + grafo de siempre, solo que
presentada de forma legible.

Al abrir la ventana, carga el índice y el encoder una sola vez en segundo
plano (la caja de texto queda deshabilitada mientras tanto) y después cada
consulta que escribas es casi instantánea, porque ya no recarga el modelo.

La ventana está dividida en dos partes:

- **Chat (izquierda)**: caja de texto + botones "Enviar" y "Correr las 50
  consultas de prueba" (usa `consultas_prueba/consultas_50.jsonl`, cada una
  aparece en el chat como si fuera una conversación). Cada respuesta muestra
  cuántos tokens procesó el encoder para esa consulta y cuánto tardó (no es
  costo de API/LLM — es el conteo de tokens de entrada del encoder,
  `Encoder.count_tokens()` en `src/embedding/encoders.py`).
- **Actividad (derecha)**: panel aparte, siempre visible (no una ventana que
  hay que abrir), que muestra en vivo qué está pasando: carga del modelo,
  cada consulta respondida, y el progreso documento por documento cuando se
  reconstruye el índice.

Arriba también hay un botón **"Reconstruir índice (offline)"** (recorre el
corpus, reconstruye `index.faiss`/`metadata.jsonl`/el grafo, y recarga la
sesión de chat al terminar) y **"Ver historial"**, con cada interacción
anterior: preguntas sueltas del chat, corridas del lote de 50, y
reconstrucciones del índice — fecha, encoder, detalle, tokens y duración.

El historial se guarda en `intermedios/historial_ejecuciones.jsonl` (JSON
Lines, gitignorado — es un registro local de cada máquina, no un entregable
del reto). Para reiniciarlo, simplemente borrar ese archivo.

## 6. Qué es provisional vs. infraestructura reusable

**Provisional (se reemplaza cuando ADL entregue lo oficial, sin tocar `src/`):**

- `corpus_ejemplo/`: 15 documentos **sintéticos** (PDF/HTML, ES/EN/PT),
  redactados a mano con `scripts/gen_synthetic_corpus.py` porque el proxy de
  salida de este entorno de desarrollo bloquea la descarga de documentos
  reales (ver `informe_tecnico.pdf`, sección de limitaciones, y
  `corpus_ejemplo/fuentes.md`). **No son documentos reales, no citar como
  tales.**
- `consultas_prueba/consultas_prueba.jsonl`: 15 consultas de prueba ES/EN/PT
  escritas a mano, NO son las 50 consultas oficiales q001–q050 (ADL aún no
  las entrega). El formato exacto del archivo oficial de consultas tampoco
  se conoce todavía; el adaptador está aislado en `load_consultas()` dentro
  de `Entrega/generador.py`.
- Los `doc_id` los suministra ADL junto con el corpus, y son la clave real
  de emparejamiento con el ground truth (aclarado en la Q&A final; la
  sec. 10.2.1 del PDF decía `fuente`, pero fue un error de versionamiento).
  El pipeline ya lo soporta: pasar `--doc-id-manifest <archivo>` a
  `scripts/build_corpus_index.py` con el mapeo que entregue ADL
  (JSON/JSONL/CSV). Sin manifest se usa un hash del contenido, suficiente
  para el corpus de ejemplo pero **no empareja con el ground truth**. Ver
  `src/ingestion/doc_id.py`.
- El campo `fuente` (Tabla 1, obligatorio) se deriva del nombre de archivo o
  la URL detectada (`derive_fuente()` en `src/ingestion/pipeline.py`) y se
  conserva como trazabilidad — ajustar ahí si ADL usa otra convención.

**Reusable sin cambios:** todo `src/` (extraction, cleaning, chunking,
embedding, retrieval, graph, ingestion), y la lógica central de
`Entrega/generador.py`. Para usar el corpus real: colocarlo donde hoy está
`corpus_ejemplo/` (o apuntar `--corpus-dir` a otra carpeta) y volver a
correr `scripts/build_corpus_index.py`.

### Limitación de red de este entorno de desarrollo

El proxy de salida de este sandbox bloqueaba (403) tanto los dominios de las
fuentes documentales reales (sipri.org, esa.int, cepal.org, etc.) como
`huggingface.co`. Esto significó que:

1. No se pudo descargar un corpus real → se generó uno sintético (ver
   arriba).
2. No se pudieron descargar los pesos del encoder real ni el modelo de NER
   de HuggingFace originalmente considerado para el grafo.

Mitigación:

- El encoder está detrás de una interfaz intercambiable
  (`src/embedding/encoders.py`): `SentenceTransformerEncoder` (real,
  producción) y `HashingFakeEncoder` (determinista, sin red, solo para
  pruebas de mecánica — NO produce embeddings semánticamente válidos). El
  índice y `resultados.jsonl` que están hoy en `Entrega/` **sí fueron
  generados con el encoder real** (confirmado: la carpeta del índice lleva
  el nombre del encoder real y su dimensión es 384, consistente con
  MiniLM).
- El grafo de conocimiento usa el NER ya incluido en los modelos spaCy
  `es/en/pt_core_news_sm` (se instalan como paquetes de pip, no requieren
  `huggingface.co`) en vez del modelo HF originalmente propuesto.

**Antes de la entrega final**, si cambia el corpus, correr en un entorno con
acceso normal a internet:

```bash
python scripts/build_corpus_index.py --with-graph --corpus-dir <corpus_real_de_ADL>
python Entrega/generador.py --consultas <archivo_oficial_de_ADL> --use-graph
```

Esto sobrescribe `Entrega/base_vectorial/` y `Entrega/resultados.jsonl` con
el índice y los resultados reales (encoder real, corpus real).

## 7. Checklist de entregables frente a la especificación

Mapeo directo a la Sección 1.4 ("Entregables") de
`Material de apoyo/CODEFEST_2026-1.pdf`:

| # | Entregable exigido | Dónde está | Estado |
|---|---|---|---|
| 1 | Base vectorial: `index.faiss` + `metadata.jsonl` por encoder, en `base_vectorial/encoder_<nombre>/` | `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/` | ✅ generado con el encoder real, `index.faiss` serializado con `faiss.write_index()` |
| 1b | Grafo de conocimiento (bonus) en `base_vectorial/grafo/grafo.graphml` | `Entrega/base_vectorial/grafo/grafo.graphml` | ✅ (bonus implementado) |
| 2 | `resultados.jsonl`, 50 líneas, consultas q001–q050 | `Entrega/resultados.jsonl` | ⚠️ 50 líneas presentes, pero generadas con las consultas de prueba provisionales, no las oficiales — regenerar con q001–q050 cuando ADL las entregue (paso 4 de la sección 6) |
| 3 | Documento técnico en PDF (máx. 8 páginas): chunking, encoder(s), tipo de índice FAISS, grafo | `Entrega/informe_tecnico.pdf` | ✅ |
| 4 | Script `generador.py` que reproduce `resultados.jsonl` desde el índice | `Entrega/generador.py` | ✅ |

Campos de metadata obligatorios por fragmento (Tabla 1 de la especificación:
`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`,
`num_tokens`, `texto`) están todos presentes en `metadata.jsonl` — ver un
registro de ejemplo corriendo:

```bash
python -c "import json; print(json.loads(open('Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl', encoding='utf-8').readline()))"
```

Esquema del resultado por consulta (Sección 9 de la especificación: 3
documentos con `rank`/`doc_id`, 10 fragmentos con `rank`/`chunk_id`/`doc_id`/
`text` ≤250 palabras) se valida con el comando de la sección 4, paso 3.

**Pendiente real (no técnico, de datos):** reemplazar corpus y consultas
sintéticos por los oficiales de ADL en cuanto estén disponibles, y
regenerar `Entrega/` con ellos antes de la entrega final (ver sección 6).

## 8. Estructura del proyecto

```
src/                    codigo reusable del pipeline
  extraction/           un extractor por formato de origen (pdf/html/json/csv-xlsx/imagen/pbf)
  cleaning/              limpieza de texto + deteccion de idioma ES/EN/PT
  chunking/              chunking hibrido: estructural -> por oracion -> por presupuesto de tokens (sec. 3)
  embedding/             interfaz Encoder (real y fake) + construccion del indice FAISS
  retrieval/             busqueda en FAISS, agregacion a documento, fusion multi-encoder/grafo, truncado a 250 palabras
  graph/                 NER (spaCy) + relaciones + construccion y consulta del grafo (bonus, sec. 7)
  gui/                   orquestacion de la GUI (misma logica de src/, sin llamadas nuevas)
  ingestion/              pipeline por documento (extraccion+limpieza+chunking) y doc_id/fuente
  config.py               constantes centrales (encoder por defecto, modelos spaCy, limites)
corpus_ejemplo/         corpus de ejemplo SINTETICO (provisional, ver sec. 6)
consultas_prueba/       consultas de prueba (provisional, ver sec. 6)
scripts/
  gen_synthetic_corpus.py   genera corpus_ejemplo/ (dev only, no es parte del pipeline)
  build_corpus_index.py     OFFLINE: corpus -> indice FAISS + metadata + grafo
  inspect_results.py        inspeccion cualitativa manual de resultados.jsonl
  gui_app.py                 lanza la interfaz grafica (Tkinter)
tests/                  pytest (31 pruebas, corren con HashingFakeEncoder, sin red)
Entrega/                estructura oficial de entrega (ver sec. 1.4 de la especificacion)
docs/                   documentacion adicional (arquitectura de encoders)
```

## 9. Solución de problemas

- **`ModuleNotFoundError` al correr `pytest` o cualquier script** → el venv
  no está activado en esa terminal. Repetir el paso 3.2 (el prompt debe
  mostrar `(.venv)` al inicio).
- **`OSError: Can't find model 'es_core_news_sm'`** (u otro idioma) al
  correr los tests de grafo o `--with-graph` → falta instalar los modelos de
  spaCy, repetir el paso 3.3.
- **Descargas de PyPI o HuggingFace fallan con 403 / timeout** → estás en
  una red con proxy restringido (ver sección 6). Los tests (`pytest`) no
  requieren red porque usan `HashingFakeEncoder`; construir el índice real
  (paso 1 de la sección 4) sí necesita acceso normal a internet.
- **PowerShell no deja activar el venv** (`scripts está deshabilitada`) → ver
  la nota en la sección 3.2 (`Set-ExecutionPolicy`).
- **La GUI no abre en Linux** (`No module named tkinter`) → instalar el
  paquete del sistema (`python3-tk` / `python3-tkinter`), ver sección 5.
- **Quiero probar el pipeline sin descargar nada** → usar la "demo
  alternativa sin red" al final de la sección 4 (`--use-fake-encoder`); los
  resultados no tienen calidad semántica real, solo sirven para validar la
  mecánica.
