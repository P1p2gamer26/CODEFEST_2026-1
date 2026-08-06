# Handoff: la VM NBDG20 como banco de cómputo

Máquina de la Javeriana asignada hasta el **27 de noviembre de 2026**.
Este documento es lo que hay que leer para retomar el trabajo remoto sin la
sesión que lo montó. Mismo rol que `handoff_gte.md`.

## Acceso

```bash
ssh estudiante@10.43.97.37          # llave publica ya instalada, sin password
cd ~/CODEFEST_2026-1
```

La contraseña original (`AliceBlue+Bl`) sigue sirviendo desde una terminal
interactiva, pero el acceso por llave es el que usan los scripts.

## Qué tiene, medido el 6 ago 2026

| | |
|---|---|
| CPU | 4 cores, Intel Xeon Gold 6348 @ 2.60 GHz |
| RAM | 15 GB |
| Disco | 15 G en `/home` — **es el recurso escaso** |
| GPU | **no hay** |
| SO | Rocky Linux 9.8, **sin sudo** |
| Red | HuggingFace, GitHub y PyPI alcanzables |

Dos consecuencias que mandan sobre qué se puede intentar:

1. **Sin GPU, `gte-multilingual-base` vuelve a ser infactible de construir**
   (97 h de CPU, medido). Su índice ya construido sí está acá y se usa como
   re-puntuador; lo que no se puede es indexar otro encoder de ese porte.
2. **Sin sudo no hay `dnf`.** Python 3.9 es lo único del sistema y no aguanta
   `requirements.txt` (torch 2.13 pide ≥3.10). Por eso el entorno va con `uv`,
   que se instala en `~/.local` sin root.

## Cómo quedó montado

```bash
export PATH=$HOME/.local/bin:$PATH      # uv
cd ~/CODEFEST_2026-1
.venv/bin/python  --version             # 3.12.13
```

- Repo clonado con `GIT_LFS_SKIP_SMUDGE=1` desde HTTPS: **no consume la cuota
  de LFS** (punto 12 de `las notas del proyecto`) y no necesita credenciales.
- **`torch` es la build `+cpu`.** La instalación por defecto arrastra 3,4 GB de
  CUDA inútiles en una máquina sin GPU; se reemplazó a mano y el venv bajó de
  5,2 G a 1,5 G. **Si alguna vez hay que reinstalar, usar
  `--index-url https://download.pytorch.org/whl/cpu`** o el disco no alcanza.
- Modelos de spaCy instalados por URL directa del wheel: `spacy download`
  falla con `ConnectionReset` contra GitHub Releases desde esta red.

Lo que se transfirió desde la máquina local (no está en git):

| archivo | por qué |
|---|---|
| `dev/intermedios/chunks_intermedios_limpio.jsonl` | **el artefacto irremplazable.** Los 128.526 chunks con guiones reparados. Con esto se indexa cualquier encoder nuevo sin el corpus de 3 GB |
| `dev/intermedios/doc_id_manifest.csv` | emparejamiento con el ground truth |
| los tres `index.faiss` | MiniLM 197 MB, e5 395 MB, gte 395 MB |
| un solo `metadata.jsonl` | los tres son byte-idénticos (chunking único, punto 8); los otros dos son symlinks |

**`las notas del proyecto` también se copió a mano y esa es una trampa permanente:** está
gitignoreado (para que nunca lleve atribución de IA), así que **no viaja por
git en ninguna dirección**. Cada vez que se edite de un lado hay que copiarlo
al otro con `scp`, o las dos máquinas empiezan a trabajar con reglas distintas.

**El corpus crudo NO está y no debe mandarse.** El chunking está congelado por
el invariante del punto 8: re-chunkear rompería los `chunk_id` y con ellos la
fusión entre encoders.

## El chequeo de sanidad (correr siempre antes de creerle a una medición)

```bash
cd ~/CODEFEST_2026-1
.venv/bin/python -m pytest dev/tests -q
.venv/bin/python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl
```

Tiene que dar **136 passed** y **F1@3 0.402 / NDCG@10 0.457**. Verificado el
6 ago 2026. Si esos números no salen, el entorno remoto dejó de ser equivalente
al local y **ninguna medición hecha acá es comparable** con lo de `las notas del proyecto`.

## Cómo se trabaja acá

`dev/experimentos/cola.jsonl` tiene los experimentos con su hipótesis, su
justificación mecánica y su riesgo escritos **antes** de medir.
`dev/experimentos/bitacora.md` es donde va el resultado, gane o pierda.

Reglas, todas heredadas de `docs/lecciones_metodologia.md`:

1. **Un experimento a la vez**, en el orden de la cola.
2. La hipótesis y la justificación mecánica se escriben antes de correr nada.
   Si no hay una razón mecánica previa, el experimento no se corre.
3. Medir en las **dos** muestras (las 50 y las 10 independientes) y reportar
   **victorias por consulta**, no promedios.
4. Adoptar solo si el IC al 90% del delta pareado excluye una pérdida de 0.02.
   Cuando el cambio toca fragmentos, decidir con **NDCG@10**.
5. **Registrar también lo que falla.** La mitad del valor del proyecto está en
   la lista de "medido y descartado" de `las notas del proyecto`.
6. **No se toca `Entrega/`.** Todo va a `dev/intermedios/`. Regenerar la
   entrega es decisión con humano: el punto 14 (autocontención de
   `generador.py`) y el punto 12 (cuota de LFS) hacen que deshacerlo sea caro.
7. Commits en `Julian_Africano`, **sin atribución de IA**, uno por experimento
   cerrado, para que el trabajo sobreviva a que se corte la sesión.
8. **Reporte por Gmail al cerrar cada experimento.** El conector de Gmail de
   esta cuenta solo soporta `create_draft`, no envío directo — no hay
   `send_message`. Crear un draft a africanojulian@gmail.com con asunto
   `[CODEFEST loop] <id> cerrado — <veredicto corto>` y cuerpo con hipótesis,
   la tabla de métricas y el veredicto. El usuario los lee/manda desde la
   app de Gmail en el celular; no depender de que abra la VM.
9. **Cuando `cola.jsonl` se agote** (todo en `cerrado` salvo E05, que es
   humano), no parar: generar experimentos nuevos con el mismo formato
   (id, hipótesis, justificación mecánica, comando, riesgo, **escritos antes
   de medir**) a partir de lo identificado como no probado en
   `docs/plan_encoders.md` y en el estado de `las notas del proyecto`. Mismo criterio de
   adopción, mismas reglas 1-8. El trabajo es de mejora continua, no una
   lista fija.
10. **Cierre de ronda: un draft de Gmail consolidado antes de generar la
    siguiente tanda.** En el mismo momento de la regla 9 (cola agotada, antes
    de escribir los experimentos nuevos), crear un draft aparte —
    asunto `[CODEFEST loop] Ronda cerrada — resumen y plan siguiente` — con:
    (a) tabla de TODO lo cerrado hasta ahora con su F1@3/NDCG@10 en las dos
    muestras y su veredicto (leer `bitacora.md` completa, no solo lo nuevo);
    (b) cuál es la mejor configuración medida hoy contra la entregada, y si
    algo quedó "adoptable" esperando decisión humana para tocar `Entrega/`;
    (c) la lista de experimentos nuevos que se van a correr en la ronda que
    arranca y por qué (una línea de justificación mecánica cada uno, no el
    JSON crudo). Es el punto donde el usuario mira las métricas acumuladas y
    decide si deja seguir la siguiente ronda o interviene — el loop asume
    que sigue si no hay nadie que lo pare, pero el draft tiene que dejarlo
    fácil de frenar (mencionar explícitamente `detener la tarea por su id` como opción).

## Corridas largas

Nunca como hijo de la sesión: si la sesión se corta, se muere el proceso. Es el
mismo problema que el `Start-Process -WindowStyle Hidden` de Windows.

**Esta VM no tiene `tmux` instalado y no hay sudo para instalarlo** (a pesar
de lo que se asumió al principio). Usar `setsid`/`nohup` en su lugar:

```bash
setsid nohup .venv/bin/python dev/scripts/build_corpus_index.py ... > dev/intermedios/log_idx.txt 2>&1 < /dev/null &
disown
```

La codificación es reanudable (`emb_<encoder>.npy` + `.progreso`, se graba cada
4096 chunks), así que una corrida muerta se relanza con el mismo comando.
**Verificar en el log que dice `codificados 4096/...` y no `reanudando`** si se
cambió el texto de los chunks.

**Antes de lanzar cualquier indexación, mirar `df -h /home`.** Con menos de
2 GB libres no arrancar: un encoder de 1024 dim son ~527 MB de índice más otros
527 del `.npy` intermedio.

## Traer el trabajo de vuelta

Desde la máquina local, la VM es un remoto git más:

```bash
git remote add vm ssh://estudiante@10.43.97.37/home/estudiante/CODEFEST_2026-1
git fetch vm && git log --oneline vm/Julian_Africano
```

Así no hace falta ningún credencial de GitHub en la VM.
