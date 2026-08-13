# CODEFEST AD ASTRA 2026 — Etapa 1: Base de Conocimiento Vectorial

Implementación completa de la Etapa 1 del reto: extracción, limpieza,
chunking, embeddings, índice FAISS, recuperación (con **cascada de tres
encoders** y grafo de conocimiento bonus) y generación de `resultados.jsonl`
a partir de consultas en lenguaje natural. Especificación completa en
`Material de apoyo/CODEFEST_2026-1.pdf`.

**Cómo recupera:** la consulta se expande con un glosario bilingüe ES→EN,
MiniLM trae 200 candidatos y los re-puntúan `gte-multilingual-base` y
`multilingual-e5-base`; sus cosenos se calibran min-max por consulta y se
combinan con pesos 0,50 y 1,00. El pool se agrega a documento con `top6`
sobre `k_pool=200` y se post-filtra por fenómeno dominante
(umbral 0,8); los 10 fragmentos se ordenan hacia los 3 documentos
entregados. F1@3 **0,499**, NDCG@10 **0,558** y NDCG penalizado **0,539**
sobre las 50 consultas. El
detalle de por qué cada encoder está donde está:
`dev/docs/arquitectura_encoders.md`.

> Sin modelos generativos en ningún punto del pipeline (prohibido por la
> sec. 8.3 de la especificación). Todo el sistema es recuperación pura sobre
> vectores, FAISS y metadata.

## ⚠️ Primero: rehidratar `Entrega/base_vectorial/`

**Los índices no se versionan en el repositorio, y ese es el único paso
manual de toda la entrega.** Los tres `index.faiss`, sus `metadata.jsonl` y
el grafo suman **1,6 GB**: por encima del límite de 100 MB por archivo de
GitHub y por encima de la cuota de 1 GB de Git LFS. Por eso se publican como
**GitHub Release**, que admite 2 GiB por archivo sin cuota de almacenamiento
ni de ancho de banda. Git tampoco versiona carpetas vacías, así que el clon
**no trae** `Entrega/base_vectorial/`: hay que crearla y depositar los cinco
archivos.

**Los cinco assets del Release, y a qué archivo va cada uno:**

| asset del Release | tamaño | destino dentro de `Entrega/base_vectorial/` |
|---|---:|---|
| `minilm-index.faiss` | 197 MB | `encoder_paraphrase-multilingual-MiniLM-L12-v2/index.faiss` |
| `gte-index.faiss` | 395 MB | `encoder_gte-multilingual-base/index.faiss` |
| `e5-index.faiss` | 395 MB | `encoder_multilingual-e5-base/index.faiss` |
| `metadata.jsonl.gz` | 34 MB | descomprimir y **copiar a los tres** `encoder_*/metadata.jsonl` |
| `grafo.graphml.gz` | 16 MB | descomprimir en `grafo/grafo.graphml` |

Se publica **un solo** `metadata.jsonl` porque los tres son **byte-idénticos
por diseño**: el chunking se hace una única vez y los tres índices comparten
el orden de filas. Ese es justamente el invariante que hace válida la
cascada, y `validar_entrega.py` lo comprueba.

Los `.faiss` **no se comprimen** (son floats casi aleatorios y no bajan de
tamaño); el texto sí, y por eso los otros dos van en `.gz`. **Los `.gz` hay
que descomprimirlos y borrarlos**: la sec. 1.4 exige `metadata.jsonl` en
JSON Lines crudo y un `index.faiss` cargable directamente con
`faiss.read_index()`, así que la carpeta de entrega no puede contener nada
comprimido.

Descargar los assets, con [`gh`](https://cli.github.com/) o sin él:

```bash
gh release download indices-v2 -D idx        # necesita gh autenticado
```

Sin `gh`, se bajan a mano desde
<https://github.com/P1p2gamer26/CODEFEST_2026-1/releases/tag/indices-v2>
a una carpeta `idx/`, o con `curl` sobre la URL de cada asset.

<details open>
<summary><b>Linux / macOS</b> (bash)</summary>

```bash
B=Entrega/base_vectorial
MINILM=$B/encoder_paraphrase-multilingual-MiniLM-L12-v2
GTE=$B/encoder_gte-multilingual-base
E5=$B/encoder_multilingual-e5-base
mkdir -p "$MINILM" "$GTE" "$E5" "$B/grafo"

cp idx/minilm-index.faiss "$MINILM/index.faiss"
cp idx/gte-index.faiss    "$GTE/index.faiss"
cp idx/e5-index.faiss     "$E5/index.faiss"

gunzip -c idx/metadata.jsonl.gz > "$MINILM/metadata.jsonl"
cp "$MINILM/metadata.jsonl" "$GTE/metadata.jsonl"
cp "$MINILM/metadata.jsonl" "$E5/metadata.jsonl"
gunzip -c idx/grafo.graphml.gz  > "$B/grafo/grafo.graphml"
```
</details>

<details>
<summary><b>Windows</b> (PowerShell)</summary>

```powershell
$B      = "Entrega\base_vectorial"
$MINILM = "$B\encoder_paraphrase-multilingual-MiniLM-L12-v2"
$GTE    = "$B\encoder_gte-multilingual-base"
$E5     = "$B\encoder_multilingual-e5-base"
foreach ($d in @($MINILM, $GTE, $E5, "$B\grafo")) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Copy-Item idx\minilm-index.faiss "$MINILM\index.faiss"
Copy-Item idx\gte-index.faiss    "$GTE\index.faiss"
Copy-Item idx\e5-index.faiss     "$E5\index.faiss"

# PowerShell no trae gunzip; se descomprime con .NET, que viene con Windows.
function Expand-Gz($src, $dst) {
    $in  = [IO.File]::OpenRead($src)
    $out = [IO.File]::Create($dst)
    $gz  = New-Object IO.Compression.GZipStream($in, [IO.Compression.CompressionMode]::Decompress)
    $gz.CopyTo($out); $gz.Dispose(); $out.Dispose(); $in.Dispose()
}
Expand-Gz "idx\metadata.jsonl.gz" "$MINILM\metadata.jsonl"
Copy-Item "$MINILM\metadata.jsonl" "$GTE\metadata.jsonl"
Copy-Item "$MINILM\metadata.jsonl" "$E5\metadata.jsonl"
Expand-Gz "idx\grafo.graphml.gz"  "$B\grafo\grafo.graphml"
```
</details>

Verificación después de rehidratar — comprueba la estructura de la sec. 1.4,
que los tres índices estén alineados con su metadata y que no sobre ningún
archivo:

```bash
python dev/scripts/validar_entrega.py --esperar-50
```

## Índice

0. [Estado y resultados](#0-estado-y-resultados)
1. [Qué hay en este repo](#1-qué-hay-en-este-repo)
2. [Requisitos previos](#2-requisitos-previos)
3. [Instalación (Windows / macOS / Linux)](#3-instalación-windows--macos--linux)
4. [Cómo correr todo desde cero](#4-cómo-correr-todo-desde-cero)
5. [Interfaz gráfica (GUI)](#5-interfaz-gráfica-gui)
6. [Qué es provisional vs. infraestructura reusable](#6-qué-es-provisional-vs-infraestructura-reusable)
7. [Checklist de entregables frente a la especificación](#7-checklist-de-entregables-frente-a-la-especificación)
8. [Estructura del proyecto](#8-estructura-del-proyecto)
9. [Solución de problemas](#9-solución-de-problemas)

## 0. Estado y resultados

**La entrega está completa, válida y reproducible.**

| verificación | estado |
|---|---|
| `pytest dev/tests` | 189 passed |
| `python dev/scripts/validar_entrega.py` | en verde |
| `python dev/scripts/pruebas_robustez.py` | todas pasan |
| **corrida en frío** desde fuera del repo | **reproduce byte a byte** |
| `informe_tecnico.pdf` | 8 de 8 páginas |

### Métricas — **medición interna del equipo, no una calificación oficial**

> **Léase esto antes que los números.** ADL **no publica** las etiquetas de
> relevancia de las 50 consultas oficiales, así que no hay forma de conocer la
> nota real antes de la evaluación. Las cifras de abajo salen de un **ground
> truth construido por nosotros** para poder decidir entre configuraciones con
> evidencia en vez de intuición. **No son la calificación de ADL, no la
> predicen, y su valor absoluto significa poco**: sirven para comparar dos
> versiones de nuestro propio sistema entre sí.

Última medición reproducida el **12 de agosto de 2026** con
`Entrega/resultados.jsonl` y `dev/scripts/eval_mini.py`:

| muestra | F1@3 | NDCG@10 | NDCG penalizado |
|---|---:|---:|---:|
| **50 consultas** | **0,499** | **0,558** | **0,539** |
| 41 de anotación humana | 0,518 | 0,573 | 0,554 |
| 10 sin sesgo de pooling | 0,433 | 0,477 | 0,470 |

El F1@3 global equivale al **55% del techo alcanzable de 0,906**, y quedan
**8 de 50 consultas con F1@3 igual a cero** (eran 11 en la configuración
anterior).

**El techo es 0,906 y no 1**: la sec. 10.2.2 fija P@3 = aciertos/3 con los
tres cupos siempre llenos, así que una consulta con un solo documento
relevante topa en 0,50 por construcción. Cualquier F1@3 hay que leerlo contra
ese techo.

#### Cómo se construyó el ground truth

Es **pooling manual**, la técnica estándar de TREC cuando no hay juicios
publicados. Con 1.826 documentos y 50 consultas, anotar todo es imposible
(91.300 pares), así que se anota solo lo que el sistema propone:

1. **Se vuelca el pool de candidatos** de cada consulta a un archivo con
   casillas para marcar (`dev/scripts/anotar_candidatos.py --generar`).
2. **Una persona marca a mano** cuáles documentos responden realmente la
   consulta, leyendo el fragmento. De ahí salen las **41 consultas de
   anotación humana**.
3. **Nueve consultas quedaron con cero marcas**: ninguno de los candidatos
   propuestos era relevante, o sea que el fallo estaba en la construcción del
   pool y no en el ranking. Para no dejarlas fuera del promedio se etiquetaron
   con **anotación asistida** y van **marcadas como tales** en el archivo
   (campo `anotador`), porque son el eslabón débil: medimos que ese
   procedimiento reproduce al anotador humano con F1 0,23. `eval_mini.py`
   desglosa por procedencia y tiene `--solo-humanas` para excluirlas.
4. **Diez consultas se anotaron sin mirar lo que el sistema propone**, para
   tener una muestra libre del sesgo de pooling (ver abajo).

Todo el material está en `dev/eval/` y el procedimiento en `dev/eval/README.md`.

#### Las tres advertencias que van siempre junto a los números

- **Sesgo de pooling.** Un documento que el recuperador nunca propuso no se
  anotó, así que cuenta como irrelevante aunque quizá no lo sea. Eso **infla**
  al sistema que generó el pool, y por eso existen las **10 consultas sin
  sesgo de pooling**: son las únicas válidas para comparar recuperadores
  distintos entre sí. Más de una mejora "clara" del proyecto se descartó
  precisamente porque su ganancia se evaporaba en esas 10.
- **El NDCG@10 es un proxy.** La relevancia de un fragmento se hereda de su
  documento, porque anotarla de verdad exigiría juicio graduado fragmento por
  fragmento. Un fragmento de bibliografía dentro de un documento relevante
  puntúa 1 en nuestra medición y casi seguro 0 con ADL. El **NDCG
  penalizado** acota por abajo cuánto miente ese proxy descontando el aparato
  bibliográfico.
- **El puntaje del concurso es relativo.** Se combina NDCG@10 y F1@3 por
  **Conteo de Borda**, o sea posición frente a los otros equipos, no un valor
  absoluto.

Cómo se decidió qué cambios entraban —contar victorias por consulta y no
promedios, exigir intervalo de confianza, medir en dos muestras, fijar la
regla antes de medir— está en `dev/docs/lecciones_metodologia.md`.

### Qué se probó y no funcionó

Están todas medidas, con sus números, en la sección "Medido y descartado" de
`dev/docs/PLAN_MAESTRO.md`. Las principales: híbrido BM25 + denso (pierde 15-4), fusión RRF
simétrica de encoders (0,268 vs 0,306), invertir la cascada (0,250),
deduplicar documentos, concatenar chunks vecinos, **fusionar** el grafo en la
recuperación vía RRF (pierde 11-0 — el grafo sí entra, pero como desempate
no desplazante), re-chunkear a 128 tokens (0,294 vs 0,375),
`bge-m3` como cuarto encoder, y cribar el ground truth con rondas de anotacion asistida
(F1 0,23 contra el humano).

**Ejes cerrados por medición, no reabrir sin datos nuevos:** otro encoder
(E04, E25, E31), reordenar lo que el pool ya trajo (E27–E29), el ancho del
pool y `topM` (E33, E37), el glosario (E35, ya está completo) y la decisión
de fenómeno (E36).

**Último cambio adoptado: E46.** Calibra por consulta los cosenos de los tres
encoders (min-max sobre el mismo pool), pondera gte con 0,50 y e5 con 1,00,
amplía el pool a 200 candidatos y agrega hasta seis chunks por documento.
Frente a la configuración anterior mejora F1@3 en +0,045, NDCG@10 en +0,042 y
NDCG penalizado en +0,040, y baja de 11 a 8 las consultas con F1@3 = 0. La
salvedad que va siempre al lado: **en las 10 consultas independientes la
ganancia es plana** (F1 +0,000) y la prueba de signos aislada da p = 0,180.

**Lo que queda abierto es la construcción del pool.** El fallo documentado es
entre idiomas: consulta en español, documento en inglés (NBQR/CBRN,
"reabastecimiento en órbita"/*on-orbit servicing*); el glosario bilingüe lo
mitiga parcialmente. El eje del chunking quedó **cerrado por E38**: la
rejilla completa (280/384/512 tokens × 2 solapes) pierde contra el 280 con
solape de una oración que ya se entrega, y pierde monótonamente al agrandar
el chunk. Con E21 (128 tokens) medido hacia abajo, el tamaño de chunk tiene
evidencia a ambos lados.

### Documentación

| documento | para qué |
|---|---|
| `dev/docs/PLAN_MAESTRO.md` | **empezar por acá**: estado, todo lo probado, lo que queda |
| `dev/docs/arquitectura_encoders.md` | cómo funcionan los tres encoders y por qué |
| `dev/experimentos/E46_calibracion_encoders.md` | configuración actual y comparación pareada de métricas |
| `dev/docs/lecciones_metodologia.md` | **cómo se decide si un cambio sirve** — leer antes de proponer mejoras |
| `dev/docs/PROYECTO_EXPLICADO.md` | mapa módulo por módulo |
| `dev/docs/Explicacion_reto_final.md` | Q&A con ADL y reglas |

## 1. Qué hay en este repo

La raíz tiene exactamente tres carpetas:

| Carpeta | Contenido |
|---|---|
| `Entrega/` | **Lo que se entrega, y nada más** (Sección 1.4 de la especificación): `generador.py`, `resultados.jsonl`, `informe_tecnico.pdf`, `base_vectorial/`. Es autocontenida: se puede copiar sola a otra máquina y correr. |
| `dev/` | Todo el trabajo de desarrollo: `src/`, `scripts/`, `tests/`, `docs/`, corpus y consultas de prueba, artefactos intermedios. |
| `Material de apoyo/` | La especificación del reto y el material entregado por ADL. |

Dentro de `dev/`:

| Carpeta | Contenido |
|---|---|
| `dev/src/` | Código reusable del pipeline (extracción, limpieza, chunking, embeddings, retrieval, grafo, GUI). No depende del corpus concreto. |
| `dev/scripts/` | CLIs que orquestan `dev/src/`: construir el índice, generar el corpus sintético, inspeccionar resultados, validar la entrega, lanzar la GUI. |
| `dev/tests/` | Suite de pytest que valida la mecánica de cada etapa sin depender de red. |
| `dev/corpus/` | Corpus real de ADL: 1.837 archivos en tres carpetas por fenómeno. |
| `dev/consultas_prueba/` | Consultas de prueba **provisionales** (ver sección 6), no las oficiales q001–q050. |
| `dev/docs/` | Documentación adicional (arquitectura de encoders, Q&A con ADL). |
| `dev/intermedios/` | Artefactos locales no versionados (historial de la GUI, índices de prueba). |

> **`Entrega/generador.py` es autocontenido a propósito.** El evaluador recibe
> solo la carpeta `Entrega/`, así que el script no importa nada de `dev/src/`:
> es una copia **aplanada** del camino online (config, encoders, búsqueda,
> agregación, truncado, RRF, grafo), con un comentario por sección indicando su
> módulo de origen. Al cambiar cualquiera de esos módulos hay que re-aplanar el
> cambio en `generador.py`; `dev/tests/test_retrieval_schema.py` lo corre desde
> fuera del repo con `PYTHONPATH` vacío y falla si dejó de ser autónomo.

Ver la [sección 8](#8-estructura-del-proyecto) para el detalle módulo por módulo.

### 1.1 Los índices no están en el repo: se bajan de un Release

`index.faiss`, `metadata.jsonl` y `grafo.graphml` suman **~1,5 GB** y **no
viajan con el clon**. Están publicados como assets de un [GitHub
Release](https://github.com/P1p2gamer26/CODEFEST_2026-1/releases), que admite
2 GiB por archivo sin cuota de almacenamiento ni de ancho de banda — al
contrario de Git LFS, que en el plan gratuito da 1 GB total y **no permite
recuperar el espacio** una vez subido algo (los objetos quedan atados al
historial del repositorio).

Los pasos para rehidratarla, en Linux/macOS y en Windows, están al principio
de este README (seccion "Primero: rehidratar `Entrega/base_vectorial/`"), que
es lo primero que hay que hacer despues de clonar. El tag publicado es
**`indices-v2`**.

#### Cómo PUBLICAR el Release (cuando se regeneran los índices)

Requiere [`gh`](https://cli.github.com/) autenticado una sola vez:

```bash
gh auth login          # elegí GitHub.com → HTTPS → autenticar en el navegador
gh auth status         # comprueba que quedó
```

Después, desde la raíz del repo:

```bash
B=Entrega/base_vectorial

# 1. Comprimir SOLO las copias que van al Release. -k conserva el original,
#    que es el que se entrega. Los .faiss NO se comprimen: son floats casi
#    aleatorios y no bajan nada. El texto sí: medido 4,4x (325 MB -> 49 MB).
gzip -6 -k $B/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl
gzip -6 -k $B/grafo/grafo.graphml

# 2. Publicar. El script comprueba que los cinco archivos existan antes de
#    intentar nada, y sube un solo metadata: los tres encoders lo tienen
#    byte-identico porque comparten el chunking unico.
bash dev/scripts/publicar_release.sh indices-v2

# 3. Borrar los .gz de la carpeta de entrega: eran solo para subir.
rm $B/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl.gz \
   $B/grafo/grafo.graphml.gz
python dev/scripts/validar_entrega.py    # tiene que quedar en verde
```

**El paso 3 no es opcional.** La sec. 1.4 exige que `Entrega/` lleve los
archivos **crudos**: un `.jsonl.gz` no cumple. `validar_entrega.py` falla si
sobra alguno, precisamente porque en una publicación anterior quedaron dos
`.gz` olvidados ahí.

Cosas que conviene saber:

- **La sintaxis `archivo#nombre`** renombra el asset al subirlo. Hace falta
  porque el Release es plano: sin eso, los tres `index.faiss` chocarían.
- **Son ~990 MB de subida.** Con conexión doméstica puede tardar bastante;
  `gh` muestra el progreso y se puede reintentar el mismo comando.
- **Si el tag ya existe**, `gh release create` falla. Ahí conviene
  `gh release delete <tag> --yes` y repetir, o subir assets sueltos con
  `gh release upload <tag> <archivo>#<nombre> --clobber`.
- **Autenticación:** `gh auth login` es interactivo y no corre bien desde un
  script. Si el token venció, el síntoma es un 401 en cualquier comando de
  `gh`; se arregla con `gh auth login --web`, que da un código de un solo uso
  para pegar en <https://github.com/login/device>.
- **No usar Git LFS.** El plan gratuito da 1 GB y **no se puede recuperar el
  espacio**: los objetos quedan atados al historial aunque se borre el
  archivo, `git lfs prune` solo limpia el disco local, y las únicas salidas
  reales son reescribir el historial, borrar y recrear el repo, o esperar 30
  días. El Release no tiene cuota ni de almacenamiento ni de descarga.
- **Esto no afecta a la entrega.** La sec. 1.4 pide una *carpeta* con los
  cuatro componentes; el enunciado no menciona GitHub en ninguna parte. El
  repo es infraestructura nuestra y dónde vivan los binarios es decisión
  libre. Lo que sí es obligatorio es que `Entrega/` tenga los archivos
  **crudos**, nunca comprimidos.

Un solo `metadata.jsonl` para los dos encoders porque es **byte-idéntico**:
el chunking se hace una sola vez y ambos índices comparten el orden de filas.

> **La carpeta `Entrega/` nunca lleva archivos comprimidos.** La Sección 1.4
> de la especificación exige `index.faiss` "directamente cargable con
> `faiss.read_index()` sin dependencias adicionales" y `metadata.jsonl` "en
> formato JSON Lines, con exactamente un objeto JSON por línea". El `.gz` es
> solo el formato de transporte del Release; lo que se entrega son los
> archivos crudos, y `generador.py` no sabe —ni debe saber— descomprimir.

## 2. Requisitos previos

- **Python.** Hay dos umbrales distintos y conviene no confundirlos:
  - **`Entrega/generador.py` corre con Python ≥ 3.9.5**, que es la versión con
    la que evalúa ADL (confirmado en la Q&A final). Es autocontenido y no
    importa nada de `dev/`, así que reproducir `resultados.jsonl` solo necesita
    eso. Un test lo vigila: el script no usa sintaxis posterior a 3.9.
  - **El resto del repo** (fase offline, barridos, GUI, tests) se desarrolló
    con **3.11+** y está probado en 3.13.

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
comandos como `pytest` o `python dev/scripts/...` van a fallar con errores de
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
pytest dev/tests -v
```

Debería dar **`189 passed`**. Estos tests usan un encoder falso y determinista
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
python dev/scripts/build_corpus_index.py --with-graph
```

Toma `dev/corpus/` (o `--corpus-dir` apuntando a otra carpeta), extrae
texto, limpia, fragmenta, codifica cada fragmento con el encoder real de
HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
se descarga solo la primera vez) y escribe:

- `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/index.faiss`
- `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl`
- `Entrega/base_vectorial/grafo/grafo.graphml` (bonus, por `--with-graph`)

Se corre una sola vez por corpus. Volver a correrlo sobrescribe el índice.

### Paso 2 — Generar resultados (fase ONLINE)

```bash
python Entrega/generador.py --consultas dev/consultas_prueba/consultas_prueba.jsonl --use-graph
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
python dev/scripts/inspect_results.py \
  --consultas dev/consultas_prueba/consultas_prueba.jsonl \
  --resultados Entrega/resultados.jsonl \
  --index-dir Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2
```

**Antes de entregar, correr siempre el validador completo:**

```bash
python dev/scripts/validar_entrega.py                 # durante el desarrollo
python dev/scripts/validar_entrega.py --esperar-50    # antes de la entrega final
```

Verifica de una sola pasada todo lo que puede costar la evaluación:
estructura de carpetas y archivos (sec. 1.4), esquema estricto de
`resultados.jsonl` (3 documentos + 10 fragmentos, ranks correctos, ≤250
palabras por fragmento), que cada `doc_id`/`chunk_id` reportado exista de
verdad en la metadata (trazabilidad), que el índice FAISS y `metadata.jsonl`
estén alineados (sec. 5.3), los campos obligatorios de la Tabla 1, y que el
informe técnico no pase de 8 páginas. Sale con código 1 si algo falla, así
que sirve en CI.

### Medir la calidad de recuperación

ADL no publica su ground truth, así que para no elegir configuraciones a ojo
hay uno propio en `dev/eval/` que cubre las 50 consultas: **41 anotadas a
mano y 9 por un las rondas de anotacion asistida** (marcadas con
`anotador: "anotacion-asistida"`, no equivalentes a las humanas). Su `README.md`
explica las limitaciones, que conviene leer antes de creerle a los números.

```bash
# F1@3 sobre el ground truth propio
python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl

# solo las 10 consultas anotadas sin pasar por el recuperador:
# OBLIGATORIO al comparar dos encoders entre sí
python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl --sin-pooling

# barrido de tamaño de pool x estrategia de agregación (carga el índice una vez)
python dev/scripts/barrido_retrieval.py

# ¿la diferencia entre dos configuraciones es real o es azar?
python dev/scripts/barrido_retrieval.py --comparar 30 60

# ampliar el ground truth: propone candidatos para marcar a mano
python dev/scripts/anotar_candidatos.py --generar      # -> dev/eval/candidatos.md
python dev/scripts/anotar_candidatos.py --recolectar

# qué documentos del manifest de ADL no quedaron indexados, y por qué
python dev/scripts/cobertura_corpus.py
```

**`--comparar` importa más de lo que parece.** Con ~30 consultas cada una pesa
0,033 en la media, así que dos que cambien de lado por azar mueven el promedio
más que un efecto real. Caso concreto: un pool de 30 parecía superar a 60
(0,309 vs 0,274) de forma consistente, y al contar por consulta el reparto era
5-3 con 18 empates — indistinguible de lanzar una moneda.

Validación mínima del esquema a mano, si se quiere sin el script (mismo
comando en cualquier plataforma porque es Python puro):

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

### Paso 3b — Cascada de tres encoders (esto es lo que se entrega)

La especificación permite construir la base con más de un encoder (sec. 4.4)
y combinar sus rankings sin modelos generativos (sec. 8.4). La entrega usa
**un encoder que busca y dos que corrigen**, y va **activada por defecto**:
correr `generador.py` sin flags reproduce `resultados.jsonl`.

```
consulta
   ├─ glosario bilingüe ES→EN ──────► consulta expandida
   ├─ MiniLM la vectoriza ──► FAISS ──► 200 candidatos      [RECALL]
   ├─ calibración min-max por encoder y consulta
   ├─ gte re-puntúa  ────────────────┤ +0,50 × señal        [PRECISIÓN]
   ├─ e5 re-puntúa   ────────────────┤ +1,00 × señal        [PRECISIÓN]
   ├─ recorte a k_pool=200 ──────────► agregación top6
   ├─ post-filtro por fenómeno ──────► 3 documentos
   └─ fragmentos hacia esos 3 docs ──► 10 fragmentos ≤250 palabras
```

| | papel | par. | ventana | dim | licencia |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | **primario** | 118 M | 128 | 384 | Apache 2.0 |
| `Alibaba-NLP/gte-multilingual-base` | re-puntuador | 305 M | 512 | 768 | Apache 2.0 |
| `intfloat/multilingual-e5-base` | re-puntuador | 278 M | 512 | 768 | MIT |

```bash
# ONLINE: la cascada ya es el comportamiento por defecto
python Entrega/generador.py --consultas <archivo>

# ...un solo re-puntuador, o ninguno
python Entrega/generador.py --consultas <archivo> --rerank-encoder multilingual-e5-base
python Entrega/generador.py --consultas <archivo> --rerank-encoder none
```

**El coste por consulta son tres vectorizaciones de una frase corta.** Los
vectores de los *pasajes* no se recalculan nunca: se leen del índice del
encoder correspondiente con `reconstruct(fila)`. Eso solo es posible porque
los tres índices describen **los mismos chunks en el mismo orden** — el
chunking se hace una sola vez y esos records se indexan con los tres.

**Por qué MiniLM es el primario siendo el más chico.** Trunca en 128 tokens y
el 96% de los chunks son más largos, así que suena a que debería perder. Se
probó lo contrario dos veces: e5 como primario da **0,250** y gte **0,200**
en las 10 consultas sin sesgo de pooling. Eso también cierra la idea de
re-fragmentar el corpus a 128 tokens — si la ventana fuera el cuello de
botella, los de ventana larga habrían ganado.

**Por qué un encoder puede ser buen re-puntuador y mal primario.** Recall y
precisión son trabajos distintos: el primario tiene que *traer* el documento
correcto entre los 200; los re-puntuadores solo *reordenan* lo que ya llegó,
así que no importa que recuperen mal.

Detalle completo, con las cinco estructuras medidas y las trampas de gte:
**`dev/docs/arquitectura_encoders.md`**.

**Por qué cascada y no fusión RRF simétrica.** La fusión simétrica se
implementó, se midió y **se descartó**: los dos encoders comparten apenas el
11,3% de los documentos del top-3, y RRF premia el acuerdo entre listas — con
ese desacuerdo no fusiona, intercala la lista buena con la mala. Lo que el
secundario hace mal es el *recall*; puntuar candidatos que el primario ya
encontró es otro trabajo.

| | F1@3 (10 indep.) | F1@3 (41) | victorias vs. primario solo |
|---|---|---|---|
| MiniLM solo | 0,300 | 0,306 | — |
| e5 solo | 0,133 | 0,182 | pierde 2-5 y 7-17 |
| fusión RRF simétrica | 0,167 | 0,268 | pierde 1-5 y 8-13 |
| **cascada 0,25 @ 200** | **0,300** | **0,352** | **gana 5-0** |

(Esa tabla es la medición **histórica** que eligió la cascada, con un solo
re-puntuador y peso 0,25.)

**E46 reemplaza el peso único 0,60.** El 0,25 se había fijado con `k_pool=60`,
agregación `sum` y sin glosario — las tres cosas cambiaron después. La grilla
0,10/0,25/0,40/0,60/0,75/0,90 (`dev/scripts/barrido_peso.py`) muestra una
**meseta, no una tendencia**, y 0,60 es el único valor que pasa el criterio de
adopción histórico bajo suma cruda. E46 midió que los rangos medianos de los
cosenos eran 0,120 (MiniLM), 0,276 (GTE) y 0,103 (E5): el mismo peso daba a
GTE mucha más autoridad efectiva. La configuración actual calibra cada señal
a [0,1] dentro del pool y usa 0,50 para GTE y 1,00 para E5. El detalle y los
controles están en `dev/experimentos/E46_calibracion_encoders.md`.
Reproducible con `dev/scripts/barrido_dos_encoders.py` y
`dev/scripts/barrido_thomas.py`.

**Detalle importante:** el chunking se hace **una sola vez** (con el tokenizer
del primer encoder) y esos mismos fragmentos se indexan con todos. Si cada
encoder fragmentara por su cuenta, los `chunk_id` colisionarían apuntando a
textos distintos y la fusión mezclaría fragmentos que no son el mismo.
`generador.py` valida esto al cargar y aborta si los índices no coinciden.

Para saber si el segundo encoder aporta algo antes de pagar su costo:

```bash
python dev/scripts/compare_encoders.py
```

Reporta cuánto se solapan los rankings de ambos. Es el diagnóstico que llevó a
descartar la fusión simétrica: con solapamiento muy bajo, RRF no fusiona nada
útil y la cascada es la forma correcta de aprovechar el segundo encoder.

> Al probar con `--use-fake-encoder` y varios encoders, pasar **siempre**
> `--out-base dev/intermedios/<algo>` (y `--index-base` en el generador) para no
> escribir índices de prueba dentro de `Entrega/`.

### Paso 4 — Cuando cambien corpus/consultas (de sintéticos a los oficiales de ADL)

Repetir solo los pasos 1 y 2, apuntando a los archivos reales — nada de
`dev/src/` necesita cambiar:

```bash
python dev/scripts/build_corpus_index.py --with-graph --corpus-dir <corpus_real_de_ADL>
python Entrega/generador.py --consultas <consultas_oficiales_q001-q050.jsonl> --use-graph
```

### Demo alternativa sin red (solo para depurar la mecánica, no usar para la entrega)

```bash
python dev/scripts/build_corpus_index.py --use-fake-encoder --with-graph \
  --out-dir dev/intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --graph-out-path dev/intermedios/demo_index_fake_encoder/grafo.graphml
python Entrega/generador.py --consultas dev/consultas_prueba/consultas_prueba.jsonl \
  --use-fake-encoder \
  --index-dir dev/intermedios/demo_index_fake_encoder/encoder_paraphrase-multilingual-MiniLM-L12-v2 \
  --out dev/intermedios/demo_index_fake_encoder/resultados_demo.jsonl
```

## 5. Interfaz gráfica (GUI)

Alternativa a los comandos de arriba para quien prefiera no usar la
terminal. Es una capa opcional sobre el mismo pipeline E46: llama a las
funciones de `dev/src/` equivalentes a `Entrega/generador.py` (ver
`dev/src/gui/runner.py`). Arranca **sin fusionar el grafo**, igual que
`Entrega/resultados.jsonl`: el grafo se conserva como artefacto bonus, pero su
fusión RRF se midió y empeora la recuperación.

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
python dev/scripts/gui_app.py
```

Es una interfaz tipo chat, no un panel de botones con ventanas emergentes:
escribes cualquier consulta y el sistema responde con los 3 documentos + 10
fragmentos más relevantes, formateados como una burbuja de respuesta. No hay
ningún modelo generativo detrás (prohibido por la sec. 8.3): lo que
"responde" es la recuperación vectorial con los tres encoders y FAISS, solo
que presentada de forma legible.

Al abrir la ventana, carga el índice y el encoder una sola vez en segundo
plano (la caja de texto queda deshabilitada mientras tanto) y después cada
consulta que escribas es casi instantánea, porque ya no recarga el modelo.

La ventana está dividida en dos partes:

- **Chat (izquierda)**: caja de texto + botones "Enviar" y "Correr las 50
  consultas de prueba" (usa
  `dev/consultas_prueba/consultas_50_oficiales.jsonl`; cada una aparece en el
  chat como si fuera una conversación). Cada respuesta muestra
  cuántos tokens procesó el encoder para esa consulta y cuánto tardó (no es
  costo de API/LLM — es el conteo de tokens de entrada del encoder,
  `Encoder.count_tokens()` en `dev/src/embedding/encoders.py`).
- **Actividad (derecha)**: panel aparte, siempre visible (no una ventana que
  hay que abrir), que muestra en vivo qué está pasando: carga del modelo,
  cada consulta respondida, y el progreso documento por documento cuando se
  reconstruye el índice.

El panel de métricas muestra primero el promedio **total** sobre las consultas
procesadas y luego los desgloses `humano` (41) y `agente` (9). Al terminar las
50 oficiales con E46 debe reproducir aproximadamente **F1@3 0,499** y
**NDCG@10 0,558**. Son métricas contra el ground truth local, no la nota de
ADL.

Arriba también hay un botón **"Reconstruir índice (offline)"** (recorre el
corpus, reconstruye `index.faiss`/`metadata.jsonl`/el grafo, y recarga la
sesión de chat al terminar) y **"Ver historial"**, con cada interacción
anterior: preguntas sueltas del chat, corridas del lote de 50, y
reconstrucciones del índice — fecha, encoder, detalle, tokens y duración.

El historial se guarda en `dev/intermedios/historial_ejecuciones.jsonl` (JSON
Lines, gitignorado — es un registro local de cada máquina, no un entregable
del reto). Para reiniciarlo, simplemente borrar ese archivo.

## 6. Qué es provisional vs. infraestructura reusable

> **Esta sección quedó casi vacía: ADL ya entregó todo.** El corpus real
> (1.837 archivos) está en `dev/corpus/`, las 50 consultas oficiales en
> `dev/consultas_prueba/consultas_50_oficiales.jsonl`, y los `doc_id` del
> manifest de ADL están aplicados. El viejo `dev/corpus_ejemplo/` sintético
> ya no existe.

**Lo que sigue siendo provisional:**

- `dev/eval/ground_truth_mini.jsonl`: **ground truth propio**, no el de ADL,
  que no se publica. Cubre las 50 consultas: 41 anotadas a mano y **9 por un
  las rondas de anotacion asistida**, marcadas con `anotador: "anotacion-asistida"`.
  Esas 9 **no son equivalentes** a las humanas —la anotacion asistida reproduce al humano
  con F1 0,23— así que para comparar contra cualquier medición anterior hay
  que usar `eval_mini.py --solo-humanas`. Detalle en
  `dev/eval/anotacion_asistida/README.md`.
- `dev/consultas_prueba/consultas_50.jsonl`: las inventadas p001–p050 del
  período previo. Sirven solo para probar multilingüe y tolerancia a typos;
  **no son las del reto**.
- Los `doc_id` los suministra ADL en `corpus_meta/Indice_Datos_Codefest.xlsx`.
  `dev/scripts/manifest_desde_xlsx.py` lo convierte al CSV que consume
  `--doc-id-manifest`. **La clave del manifest es la RUTA relativa, no el
  nombre de archivo**: 59 nombres aparecen en dos carpetas con `doc_id`
  distintos, e indexar por nombre le asignaría el `doc_id` equivocado a 127
  documentos. Ver `dev/src/ingestion/doc_id.py`.
- El campo `fuente` (Tabla 1, obligatorio) se deriva del nombre de archivo o
  la URL detectada (`derive_fuente()` en `dev/src/ingestion/pipeline.py`) y se
  conserva como trazabilidad — ajustar ahí si ADL usa otra convención.

**Reusable sin cambios:** todo `dev/src/` (extraction, cleaning, chunking,
embedding, retrieval, graph, ingestion), y la lógica central de
`Entrega/generador.py`. Para usar el corpus real: colocarlo donde hoy está
`dev/corpus/` (o apuntar `--corpus-dir` a otra carpeta) y volver a
correr `dev/scripts/build_corpus_index.py`.

### Herencia del entorno de desarrollo inicial

Al comienzo del proyecto el proxy de salida bloqueaba (403) tanto los
dominios documentales (sipri.org, esa.int, cepal.org…) como
`huggingface.co`. **Eso ya no aplica** —el corpus real y los tres encoders
están descargados y los índices de `Entrega/` se generaron con encoders
reales—, pero dejó dos decisiones de diseño que siguen vigentes y conviene
conocer:

- El encoder está detrás de una interfaz intercambiable
  (`dev/src/embedding/encoders.py`): `SentenceTransformerEncoder` (real,
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
python dev/scripts/build_corpus_index.py --with-graph --corpus-dir <corpus_real_de_ADL>
python Entrega/generador.py --consultas <archivo_oficial_de_ADL> --use-graph
```

Esto sobrescribe `Entrega/base_vectorial/` y `Entrega/resultados.jsonl` con
el índice y los resultados reales (encoder real, corpus real).

## 7. Checklist de entregables frente a la especificación

Mapeo directo a la Sección 1.4 ("Entregables") de
`Material de apoyo/CODEFEST_2026-1.pdf`:

| # | Entregable exigido | Dónde está | Estado |
|---|---|---|---|
| 1 | Base vectorial: `index.faiss` + `metadata.jsonl` por encoder, en `base_vectorial/encoder_<nombre>/` | `Entrega/base_vectorial/encoder_*/` (**tres**) | ✅ generados con encoders reales, `index.faiss` serializado con `faiss.write_index()`. La sec. 1.4 permite una subcarpeta por encoder; los tres `metadata.jsonl` son byte-idénticos porque comparten el chunking único |
| 1b | Grafo de conocimiento (bonus) en `base_vectorial/grafo/grafo.graphml` | `Entrega/base_vectorial/grafo/grafo.graphml` | ✅ construido sobre el corpus real: **224.101 nodos, 754.876 aristas** de 1.687 documentos, y `validar_entrega.py` comprueba que todos sus `doc_id` estén indexados (trazabilidad, sec. 7.3). El grafo **entra en la corrida oficial**, como pide la sec. 7 para conceder el bonus, pero **no por fusión RRF** — esa variante está medida y degrada la recuperación (pierde 11-0 sobre las 50). Entra como **desempate no desplazante**: solo reordena candidatos empatados en score, nunca desplaza a uno mejor puntuado. Medido sobre la entrega: cambia **0 de 50 líneas**, o sea integración del artefacto sin costo en métrica. Se apaga con `--sin-grafo` |
| 2 | `resultados.jsonl`, 50 líneas, consultas q001–q050 | `Entrega/resultados.jsonl` | ✅ generado con las **50 consultas oficiales** (`dev/consultas_prueba/consultas_50_oficiales.jsonl`) sobre el índice del corpus real, y verificado reproducible byte a byte desde una carpeta aislada |
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

### Tres requisitos que casi cuestan la exclusión

Ninguno se veía corriendo el camino feliz; los encontró
`dev/scripts/pruebas_robustez.py`, que ejercita `generador.py` **como lo va a
correr el evaluador**: por subprocess, desde fuera del repo, con `PYTHONPATH`
vacío y con entradas que no preparamos nosotros.

- **Python ≥ 3.9.5** (Q&A final de ADL). El script anotaba tipos con la
  sintaxis `X | None` de PEP 604, que es 3.10+, y sin
  `from __future__ import annotations` esas anotaciones **se evalúan al
  importar**: moría con `TypeError` antes de leer una consulta. No se
  detectaba porque el venv local corre 3.13.
- **BOM en el archivo de consultas.** Se leía con `utf-8`, y cualquier
  archivo guardado con Excel, Bloc de notas o PowerShell en Windows lleva
  BOM. El archivo lo entrega ADL y no controlamos cómo lo guardan.
- **Codificación ANSI (cp1252).** Es la misma clase de fallo, y quedó abierta
  al arreglar solo el BOM: `Set-Content` de Windows PowerShell escribe cp1252
  por defecto y las 50 consultas van en español con acentos, así que el script
  moría con `UnicodeDecodeError` antes de leer una consulta. `load_consultas`
  prueba ahora `utf-8-sig` → `cp1252` → `latin-1`, y el último **no puede
  fallar** porque decodifica cualquier byte. Verificado que las cuatro
  variantes —utf-8, BOM, cp1252 y CRLF— devuelven las mismas 50 consultas.

**La lección, que vale más que los dos arreglos:** probá el artefacto que
entregás, en las condiciones en que lo van a usar — no las funciones que
escribiste.

**Los índices ya están publicados** en el Release `indices-v2` (cinco assets,
1,6 GB). Hay que volver a publicarlos solo si se reconstruyen; el
procedimiento está en la sección 1.1.

## 8. Estructura del proyecto

```
README.md  requirements.txt  .gitignore

Entrega/                    ESTRUCTURA OFICIAL DE ENTREGA (sec. 1.4). Autocontenida.
  generador.py              ONLINE: consultas -> resultados.jsonl. Sin imports del repo.
  resultados.jsonl          50 lineas, q001-q050, esquema estricto (sec. 9)
  informe_tecnico.pdf       maximo 8 paginas
  base_vectorial/
    encoder_<nombre>/       index.faiss + metadata.jsonl (uno por encoder)
    grafo/grafo.graphml     bonus (sec. 7)

Material de apoyo/          especificacion del reto y material de ADL

dev/                        todo el desarrollo (NO se entrega)
  src/                      codigo reusable del pipeline
    extraction/             un extractor por formato de origen (pdf/html/json/csv-xlsx/imagen/pbf)
    cleaning/               limpieza de texto + deteccion de idioma ES/EN/PT
    chunking/               chunking hibrido: estructural -> por oracion -> por presupuesto de tokens (sec. 3)
    embedding/              interfaz Encoder (real y fake) + construccion del indice FAISS
    retrieval/              busqueda en FAISS, agregacion a documento, fusion multi-encoder/grafo, truncado a 250 palabras
    graph/                  NER (spaCy) + relaciones + construccion y consulta del grafo (bonus, sec. 7)
    gui/                    orquestacion de la GUI (misma logica de src/, sin llamadas nuevas)
    ingestion/              pipeline por documento (extraccion+limpieza+chunking) y doc_id/fuente
    config.py               constantes centrales (encoder por defecto, modelos spaCy, limites, rutas)
  scripts/
    gen_synthetic_corpus.py   genera un corpus sintetico (dev only, el corpus real ya llego)
    build_corpus_index.py     OFFLINE: corpus -> indice FAISS + metadata + grafo
    inspect_results.py        inspeccion cualitativa manual de resultados.jsonl
    validar_entrega.py        valida Entrega/ contra la especificacion (correr antes de entregar)
    pruebas_robustez.py       corre generador.py como lo correra ADL: fuera del repo, entradas raras
    barrido_estructuras.py    compara estructuras de encoders con los indices ya construidos
    indexar_desde_metadata.py construye el indice de un encoder nuevo, alineado por construccion
    verificar_alineacion.py   OBLIGATORIO antes de usar un indice nuevo en la cascada
    victorias_ndcg.py         cuenta victorias por consulta en NDCG@10 entre dos resultados
    estado_corrida.ps1        informe de una corrida larga: cuanto lleva, cuanto falta, recursos
    compare_encoders.py       diagnostico: cuanto difieren dos encoders entre si
    eval_mini.py              F1@3 + NDCG@10 (binario y penalizado) contra el ground truth propio
    volcar_pools.py           congela los pools entregados: experimentos de orden sin FAISS
    reparar_guiones.py        arregla el guion de fin de linea (U+FFFE) sin re-extraer
    rechunkear_e38.py         re-empaqueta chunks a otro presupuesto, con puerta de conservacion
    correr_e38.py             corre la rejilla de chunking en serie, reanudable
    barrido_*.py              un experimento medido por archivo (E01-E38, ver dev/docs/PLAN_MAESTRO.md)
    gen_informe_tecnico.py    genera Entrega/informe_tecnico.pdf
    gui_app.py                lanza la interfaz grafica (Tkinter)
  tests/                    pytest (corren con HashingFakeEncoder, sin red)
  corpus/                   corpus REAL de ADL: 1.837 archivos, 3 fenomenos
  corpus_real_ejemplo/      muestra del formato del corpus real
  consultas_prueba/         consultas de prueba (provisional, ver sec. 6)
  docs/                     documentacion adicional (arquitectura de encoders, Q&A con ADL)
  intermedios/              artefactos locales no versionados
  pytest.ini
```

Las tres carpetas de la raíz son independientes en un solo sentido: `dev/`
sabe de `Entrega/` (escribe el índice, valida la estructura, genera el
informe), pero `Entrega/` **nunca** sabe de `dev/`.

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
