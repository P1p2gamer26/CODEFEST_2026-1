# Diseño — E38: rejilla de chunking (tamaño × solape)

Fecha: 9 de agosto de 2026
Rama: `Julian_Africano`
Estado: diseño aprobado, pendiente de plan de implementación

> Se escribe en `dev/docs/` y no en una carpeta `docs/` en la raíz porque la
> raíz del repositorio tiene solo tres carpetas (`Entrega/`,
> `Material de apoyo/`, `dev/`) y esa convención es parte de la estructura de
> entrega.

## 1. Problema

El sistema entregado mide **F1@3 = 0.455** y **NDCG@10 = 0.516** sobre las 50
consultas. El objetivo del usuario es subir ambas métricas, sin límite de
tiempo y aceptando el coste de rehacer la fase offline.

Dos restricciones acotan lo que se puede hacer, y las dos están medidas:

1. **Todos los ejes online están cerrados.** Diecinueve experimentos negativos
   consecutivos cubren reordenamiento (E07–E12, E19–E20, E22–E23, E27–E29,
   E37), encoders alternativos (E04, E25, E31, cerrado tres veces), glosario
   (E35), grafo (E30) y filtrado léxico. Las únicas adopciones recientes son
   E22/E23 (orden de fragmentos) y E32 (post-filtrado por fenómeno).
2. **El NDCG@10 propio es ciego a la calidad del texto de los fragmentos.**
   E26 lo demostró: la relevancia de un fragmento se hereda de su `doc_id`, así
   que cualquier cambio que no altere de qué documento proviene un fragmento
   mide **+0.000 exacto por construcción**. El usuario descartó anotar
   fragmentos a mano, de modo que **la única palanca medible es el ranking de
   documentos.**

De ahí que el único eje con evidencia a favor de seguir mirándolo sea el
**chunking**, y específicamente la variante que E21 no midió.

## 2. Hipótesis

E21 midió chunks de **128 tokens sin solape** y perdió con claridad (F1@3 de
MiniLM-solo 0.375 → 0.294). Su explicación mecánica, coherente con E19 y E20,
es que al multiplicar por 1,92 el número de chunks **todos los documentos
saturan el tope de `top5`**, el conteo deja de discriminar y el orden lo decide
el ruido.

La hipótesis simétrica **nunca se midió**: si achicar el chunk satura la
agregación, **agrandarlo debería separarla**. Con chunks de 384 o 512 tokens
cada documento aporta menos chunks al pool de 100, saturar `top5` cuesta más, y
el conteo —que E20 demostró que es señal y no sesgo— vuelve a discriminar.

Hipótesis secundaria: **E21 cambió dos variables a la vez** (tamaño *y* solape).
El efecto del solape por sí solo nunca se aisló, y ya se sabe que causa daño
colateral: el 32% de los pares de chunks consecutivos comparte texto, que es lo
que hizo inviable la concatenación de fragmentos.

## 3. Diseño del experimento

### 3.1 Rejilla

| presupuesto de tokens | solape 0 oraciones | solape 1 oración |
|---|---|---|
| 280 | celda nueva | **la entregada (base)** |
| 384 | celda nueva | celda nueva |
| 512 | celda nueva | celda nueva |

Cinco chunkings nuevos. La configuración actual es
`CHUNK_TOKEN_BUDGET = 280`, `CHUNK_OVERLAP_SENTENCES = 1`
(`dev/src/config.py`), 128.526 chunks indexados sobre 1.818 documentos.

El presupuesto de 512 iguala la ventana de `gte-multilingual-base` y
`multilingual-e5-base`, que son los dos re-puntuadores con peso 0.60 cada uno y
que hoy reciben chunks de 280 tokens.

### 3.2 Criba en dos etapas

**Etapa 1 — screening con MiniLM solo.** Un índice de
`paraphrase-multilingual-MiniLM-L12-v2` por celda, y se compara MiniLM-solo
contra MiniLM-solo. Es el mismo diseño que usó E21, de modo que los resultados
son directamente comparables con los suyos. Coste estimado: ~1 h de GPU por
celda, menos en las celdas de presupuesto grande porque producen menos chunks.

**Etapa 2 — build completo, solo para la celda ganadora.** Índices de `gte`
(~7,3 h de GPU) y de `e5-base`, verificación de alineación, y regeneración de
`Entrega/`.

Si ninguna celda pasa el criterio de adopción, **la etapa 2 no se corre** y el
experimento se cierra como negativo documentado.

### 3.3 Corrección de `k_pool`, obligatoria

`k_pool` se mide en chunks, no en volumen de texto. Al pasar de 280 a 512
tokens el chunk crece 1,83×, así que un pool de 100 candidatos abarca casi el
doble de texto que hoy. E21 aprendió esto a los golpes y tuvo que añadir una
tercera fila a su tabla.

Cada celda se mide con **dos valores de `k_pool`**: el crudo (100) y el
**escalado por volumen de texto**, `round(100 × 280 / presupuesto)` — o sea 100,
73 y 55 respectivamente. Sin esa corrección la comparación mezcla el efecto del
tamaño de chunk con el del ancho efectivo del pool.

## 4. Instrumento y criterio de decisión

### 4.1 Base, verificada dígito a dígito antes de mirar ninguna celda

| muestra | F1@3 | NDCG@10 | NDCG penalizado |
|---|---|---|---|
| 50 consultas | 0.455 | 0.516 | 0.499 |
| 41 humanas | 0.486 | 0.537 | 0.520 |
| 10 independientes | 0.433 | 0.474 | 0.467 |

Además: **11 consultas con F1@3 = 0** y **0 fragmentos ilegibles**.

El screening de la etapa 1 compara contra la base de **MiniLM solo** bajo la
configuración actual (incluido el post-filtrado por fenómeno de E32), no contra
la cascada completa. Esa base se mide como primer paso del experimento y se
registra antes de construir ninguna celda.

### 4.2 Criterio de adopción

El vigente en `dev/docs/lecciones_metodologia.md`:

- IC al 90% del delta pareado que **excluya una pérdida de 0.02**.
- **Confirmación en las 41 humanas**, no solo en las 50. Nueve de las 50 llevan
  etiquetas de panel de agentes, que el propio proyecto midió con F1 de acuerdo
  0.23 contra el humano.
- Justificación mecánica escrita **antes** de medir.

### 4.3 Vetos pre-registrados

Cualquiera de los tres mata la celda aunque gane en promedio:

1. **Ceros.** Las consultas con F1@3 = 0 no pueden pasar de 11.
2. **Legibilidad.** Los fragmentos ilegibles (ko/ru/ar/zh/de) no pueden pasar de
   0. Hoy están en 0 gracias a E22, y dos experimentos independientes (E11, E12)
   ya mostraron que cualquier criterio que desplace al idioma de su lugar paga
   en fragmentos que el evaluador no puede leer.
3. **Sesgo de pooling.** Si la ganancia se concentra en las 9 consultas de panel
   y se evapora en las 41 humanas, se descarta. Es la firma que ya mató a
   `doc_rrf`, a gte-primario, a E25 y a E31.

## 5. Riesgos declarados antes de medir

1. **El resultado más probable es negativo.** E19 midió que solo el 33.7% de los
   documentos relevantes entra en los 3 cupos, y que con 10 cupos sería el
   68.4%. El ranking dentro del pool ya está exprimido. Si la rejilla gana algo,
   lo plausible es +0.02 a +0.04, no +0.25.
2. **El techo del F1@3 es 0.906, no 1** (sec. 10.2.2: P@3 = aciertos/3 con los 3
   cupos siempre llenos). Todo número de F1 se cita con el techo al lado.
3. **Re-chunkear rompe el invariante de `chunk_id`.** Los `chunk_id` son
   `doc_id` + posición; una fragmentación distinta hace que el mismo `chunk_id`
   apunte a texto distinto. Los tres índices de una celda deben construirse
   sobre **el mismo chunkeo**, y el checkpoint de codificación valida filas +
   `chunk_id` + **hash del texto** justamente para detectar esto.
4. **Elegir el máximo de una rejilla de seis celdas es sobreajuste de manual**
   si se hace por el promedio. Por eso el criterio es el IC del delta pareado
   más la confirmación en dos muestras, y por eso la celda base entra en la
   rejilla como control.
5. **La GPU no puede lanzarse desde una herramienta de fondo del agente.** Las
   corridas largas mueren sin dejar traceback. Se lanzan con `Start-Process`
   desacoplado, con `.venv-cuda\Scripts\python.exe`, y nunca instalando
   `requirements.txt` completo en ese entorno (fija `torch==2.13.0` y destruiría
   el cu124).

## 6. Fase de cierre, si alguna celda gana

No es un apéndice: es lo que separa un experimento ganador de una entrega
excluida. La entrega actual reproduce byte a byte en frío y esa propiedad hay
que reconquistarla, no asumirla.

1. Regenerar `Entrega/resultados.jsonl` con la configuración ganadora.
2. Correr `dev/scripts/validar_entrega.py` hasta que salga limpio.
3. Correr `pytest dev/tests` (143 tests en verde hoy).
4. Re-aplanar en `Entrega/generador.py` cualquier cambio del camino online y
   verificar que no aparezcan `from src` / `import src` / `sys.path`.
5. **Corrida en frío**: `generador.py` desde un directorio fuera del repo, con
   `PYTHONPATH` vacío y sin ningún flag, y `diff` contra
   `Entrega/resultados.jsonl`.
6. Borrar `Entrega/__pycache__` como último paso (se regenera cada vez que los
   tests importan `generador.py` y el validador lo marca como sobrante).
7. Subir los índices nuevos al Release de GitHub, nunca por LFS.

## 7. Qué queda explícitamente fuera

- **Anotar fragmentos o re-anotar las 9 consultas de panel.** El usuario lo
  descartó. La consecuencia queda escrita: el NDCG@10 solo se moverá en la
  medida en que se muevan los documentos.
- **Sub-fragmentos por rank (sec. 9.2.1).** El enunciado los permite y podrían
  subir el NDCG real, pero nuestro proxy los mediría en +0.000 por la misma
  razón que E26. Adoptar a ciegas un cambio que toca las 500 líneas de la
  entrega es el perfil de riesgo que el proyecto rechazó diecinueve veces.
- **Cualquier recalibración de `k_pool`, `topM`, peso o profundidad como
  experimento propio.** E37 los barrió hace horas y no gana nada. Acá se ajusta
  `k_pool` solo como control de confusión, no como variable a optimizar.
