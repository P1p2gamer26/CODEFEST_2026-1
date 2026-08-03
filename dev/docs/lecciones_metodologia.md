# Lecciones de método

Escrito el 2 de agosto de 2026, después de una jornada en la que se probaron
siete hipótesis de mejora y **seis fallaron**. Lo que sigue no es la lista de
lo que se intentó —eso está en la sección "Medido y descartado" de
`las notas del proyecto`— sino **cómo se decidió que fallaban**, que es lo reutilizable.

---

> **CORRECCIÓN IMPORTANTE (2 ago 2026, más tarde el mismo día).** Un análisis
> de poder estadístico posterior mostró que **las lecciones 1 y 3 estaban
> sobrecorregidas** y que el instrumento con el que se descartaron nueve
> hipótesis **no podía detectarlas**. Leer primero la sección "Corrección: el
> instrumento no medía" al final. Las lecciones 2 y 4 a 10 siguen en pie.

## 1. Contar victorias por consulta, nunca promedios

El caso testigo del día es `top5`: promedia **0,354** contra los **0,344** de
la entrega. Parece una mejora del 3%. Mirado por consulta, **empata en 40 de
41 y en 10 de 10**. Toda la diferencia del promedio es **una sola consulta**.

Con 41 consultas cada una pesa 0,024 en la media. Dos que cambien de lado por
azar mueven el promedio más que cualquier efecto real. **Un promedio sobre una
muestra chica no es evidencia; el reparto de victorias sí.**

`eval_mini.py --comparar-con` existe justamente para esto y hace la prueba de
signos sola.

## 2. Medir siempre en dos muestras, una sin sesgo de pooling

Las consultas anotadas mirando lo que el propio recuperador propuso **le
juegan a favor**: un documento excelente que el sistema nunca trajo jamás
pudo marcarse como relevante, así que no cuenta como error.

Por eso toda medición se corre dos veces: sobre las 41 anotadas y sobre las
10 independientes (`--sin-pooling`). **Un cambio que gana en las 41 y pierde
en las 10 está explotando el sesgo, no mejorando.** Así se descartó `doc_rrf`,
que ganaba 13-10 en las 41 y perdía 4-1 en las independientes.

## 3. Fijar la regla de decisión ANTES de medir

La regla vigente —"se adopta solo si no pierde consultas en ninguna de las dos
muestras"— se escribió antes de correr los experimentos. Eso es lo que
permitió aceptar la cascada de dos encoders sin discutir, y lo que impidió
adoptar el máximo de una grilla de 70 celdas, que es sobreajuste de manual.

Corolario incómodo pero necesario: **cuando un resultado cumple la letra de la
regla pero es evidentemente ruido (`top5`, una consulta de diferencia), hay
que decirlo en vez de cobrar la mejora.**

## 4. Un diagnóstico correcto puede llevar a una solución equivocada

El diagnóstico *"la agregación por `sum` no tiene tope y un documento con 30
fragmentos mediocres desplaza a uno con un fragmento excelente"* era
**correcto** y estaba respaldado por `diagnostico_ceros.py`.

Las tres soluciones que salieron de él —`topM`, forzar el documento del
fragmento nº 1, deduplicar— **fallaron las tres**. El diagnóstico seguía
siendo cierto; el remedio no funcionaba.

**Diagnóstico y remedio se validan por separado.** Que la explicación del
problema sea convincente no hace que la solución funcione.

## 5. Auditar las afirmaciones heredadas del propio proyecto

`las notas del proyecto` afirmaba que *"q001 es irrecuperable tanto por vía densa como
léxica"*. Era falso: hay 20 documentos con CBRN en el corpus y ninguno entra
al pool de candidatos. La consulta no es irrecuperable, el recuperador no
llega.

El costo de ese error fue dar por perdida una consulta que no lo estaba, y
cerrar la línea de investigación que sí tenía margen. **Las notas propias
envejecen y se convierten en dogma; hay que releerlas con la misma
desconfianza que a una fuente externa.**

## 6. Cuando el resumen y la fuente primaria discrepan, manda la fuente

Un resumen de terceros decía que los 10 fragmentos debían salir de los 3
documentos entregados. El PDF oficial (sec. 9.2) pide "los 10 fragmentos más
relevantes", sin esa restricción: son dos listas independientes.

Estuvimos a punto de "arreglar" algo que no estaba roto. **Ante cualquier duda
de formato o de regla, se abre el PDF.**

## 7. Un defecto de datos se arregla porque es un defecto

La reparación del guion de fin de línea (U+FFFE) tocó el 30% de los chunks y
dejó el F1@3 prácticamente igual: gana 3 consultas, pierde 3, empata 35. Por
la métrica sola, no se justificaba.

Se hizo igual, y fue lo correcto: pasó **166 fragmentos entregados con
palabras partidas a 0**. El evaluador lee ese texto. **No todo lo que vale la
pena hacer se ve en la métrica que estás mirando** — y hay que resistir la
tentación de venderlo como mejora de métrica cuando no lo es.

## 8. Los checkpoints tienen que validar contenido, no solo forma

El cache de codificación validaba el número de vectores y el `chunk_id` de la
última fila. La reparación de guiones cambió **el texto** de 38.423 chunks sin
cambiar ni la cantidad ni los ids: el cache se habría dado por bueno y el
índice habría quedado con los vectores del texto roto, **en silencio, tras 7
horas de CPU**.

Ahora el `.progreso` guarda un hash del texto. **Si un checkpoint puede
quedar desalineado sin que se note, no es un checkpoint, es una trampa.**

## 9. Los experimentos baratos van primero

Los seis experimentos del día no costaron **ni un minuto de CPU nueva**,
porque los dos índices ya estaban construidos y todos eran cambios de flag.
Descartar seis hipótesis en una hora vale más que perseguir una durante una
noche.

Antes de lanzar cualquier cosa que tarde horas, la pregunta es: **¿qué puedo
medir ya, con lo que hay construido?**

## 10. Anotadores con sesgos distintos, y el desacuerdo como dato

Para anotar las consultas que faltaban se usaron tres anotadores con criterios
deliberadamente opuestos (estricto / temático inclusivo / jurado que predice a
los organizadores) y se tomó el consenso por mayoría. Coincidieron en 7 de 9.

**Las 2 en las que no coincidieron resultaron ser las más informativas:** el
desacuerdo no venía de criterios distintos sino de que **el pool de candidatos
no contenía la respuesta**. El desacuerdo entre anotadores es una señal de
diagnóstico, no un problema a promediar.

Cautela obligatoria: son etiquetas hechas por modelos. Viven en
`dev/eval/ground_truth_agentes.jsonl`, **separadas** del anotado a mano, y no
se funden. Medir contra etiquetas de modelo premia parecerse a un modelo.

---

---

# Corrección: el instrumento no medía

Un análisis de poder estadístico sobre el propio diseño dio vuelta buena parte
de lo anterior. Los números, calculados por enumeración binomial exacta sobre
el ground truth real:

**1. El F1@3 se mueve en escalones enormes.** Con 3 documentos entregados y
`R` relevantes, cada acierto vale `2/(3+R)` — en promedio **0,315**. Una
consulta no tiene valores intermedios: salta de golpe.

**2. La prueba de signos tiene un piso duro.** Hacen falta **≥6 consultas
discordantes** para bajar de p=0,05, gane como gane. La cascada de dos
encoders dio **5-0 con p=0,062**: no fue "casi significativa", **estaba fuera
del alcance del test antes de correrla**.

**3. Con 41 consultas el instrumento es ciego.** Efecto mínimo detectable con
potencia 0,80: **0,059** en el mejor caso imaginable (el cambio acierta
siempre que actúa), e **inalcanzable a cualquier tamaño de efecto** con un
cambio realista que acierte el 75% de las veces. Para detectar ΔF1 = 0,03
harían falta **n = 140 a 455 consultas**, no 50.

**Corolario que hay que tragarse:** nueve "no concluyente" seguidos con
potencia ~0,3 es **exactamente lo que se espera aunque las nueve hipótesis
fueran mejoras reales**. No son evidencia de que no haya efectos; son
evidencia de que el test no los ve.

**4. La regla "no adoptar nada que pierda consultas" estaba
anti-correlacionada con la calidad.** Probabilidad de adoptar, según el efecto
real:

| ΔF1 verdadero | P(adoptar) |
|---|---|
| +0,016 (casi nulo) | **0,275** |
| +0,032 | 0,073 |
| +0,126 (mejora grande) | **0,073** |
| moneda al aire que toca el 5% | 0,27 |

La regla no filtraba por calidad sino por **cuán poco tocaba el sistema**:
premiaba los cambios inocuos y rechazaba los grandes. Y no controlaba el error
tipo I.

## Qué se usa ahora

**Criterio principal: el delta pareado con su intervalo de confianza**, por
bootstrap sobre los deltas por consulta (`eval_mini.py`, funciones
`bootstrap_delta` y `veredicto_bootstrap`). No tiene el piso de la prueba de
signos y se lee aunque contenga el cero: *"el efecto está entre −0,01 y
+0,09"* es accionable; *"no concluyente"* no lo es.

**Umbral asimétrico, porque esto es un torneo y no una publicación.** El coste
de adoptar una mejora que resulta nula es ~0; el de rechazar una real es
perder posiciones. Por eso el criterio no es *probar que mejora* sino
**descartar solo lo que probablemente daña**: se adopta si el IC al 90%
excluye una pérdida de 0,02, **y** el cambio tiene una justificación mecánica
anterior al experimento (esto último es lo que impide adoptar ruido).

**Decidir con NDCG@10 cuando el cambio toca los fragmentos.** F1@3 tiene 3-5
valores por consulta; NDCG@10 sobre 10 fragmentos es casi continuo. Para
detectar Δ=0,03 hacen falta **260 consultas con F1@3 contra 27-43 con
NDCG@10**: un factor de 6 a 9, gratis, sin anotar una etiqueta más.

**Límite encontrado al aplicarlo, que el análisis no preveía:** el Δ de
NDCG@10 es **exactamente 0,000** para cambios de agregación a documento
(`top3`, `top5`), porque la lista de fragmentos no depende de la estrategia de
agregación. **NDCG solo da resolución extra a los experimentos que tocan
fragmentos.** Para los de nivel documento seguimos atados a F1@3 y a su piso.

---

# Lecciones nuevas (2 ago 2026, jornada de gte)

## 11. Un modelo puede estar roto y no fallar

`gte-multilingual-base` se carga con dos buffers apuntando a memoria sin
inicializar. Uno revienta con `IndexError`; el otro **no**: el modelo codifica
sin información posicional y devuelve vectores normalizados, de la forma
correcta y semánticamente basura. Con el modelo así se midió que un pasaje
irrelevante puntuaba por encima de uno relevante, y **se emitió un veredicto
de calidad sobre un modelo averiado**.

Nos salvó que el otro buffer sí reventara. Si solo hubiera estado mal el
silencioso, se habrían gastado 7 horas de GPU para concluir "gte es malo".

**Antes de juzgar un modelo nuevo, comprobá que funciona.** Una prueba de tres
líneas —una consulta, un pasaje relevante, un distractor obvio— cuesta
segundos y detecta esto.

## 12. Medí el coste, no lo extrapoles

`plan_encoders.md` estimaba gte en ~6 h por regla de tres sobre el número de
parámetros. Fueron **97 h**: 97× más lento que MiniLM, no 2,6×. Se multiplican
parámetros, tokens efectivos y el tipo de atención, y la regla de tres solo
veía el primero.

Costó lanzar una corrida presupuestada en 23 minutos que eran 4,7 horas.
**Medir ms/chunk sobre 64 chunks reales cuesta tres minutos.**

## 13. El camino feliz no prueba nada

La entrega pasaba tests, validador y corrida en frío, y aun así tenía **dos
fallos que la habrían excluido**:

- sintaxis de Python 3.10 con un evaluador que usa ≥ 3.9.5 — invisible porque
  el venv local corre 3.13;
- lectura con `utf-8` de un archivo que ADL puede guardar con BOM.

Los dos aparecieron al correr el script **como lo va a correr el evaluador**:
por subprocess, desde fuera del repo, con entradas que no preparamos nosotros.
Eso es `scripts/pruebas_robustez.py`.

**Probá el artefacto que entregás, en las condiciones en que lo van a usar, no
las funciones que escribiste.**

## 14. Un validador que solo busca lo que falta deja pasar lo que sobra

`validar_entrega.py` comprobaba que estuvieran los archivos exigidos. No
miraba si había otros. En la carpeta de entrega vivían un `__pycache__/` y
dos `.gz` sobrantes de publicar el Release — y la sec. 1.4 fija una estructura
exacta y avisa que incumplirla es penalización severa o exclusión.

**Si una especificación dice "exactamente esto", validá ambos lados: que no
falte y que no sobre.**

## 15. El mejor promedio y el mejor sistema no son lo mismo

`gte→MiniLM` daba el mejor F1@3 de las cinco estructuras sobre las 41
consultas (0,385) y **0,200 sobre las 10 independientes**. Adoptarlo mirando
el promedio habría empeorado el sistema de verdad.

El mecanismo es siempre el mismo y conviene tenerlo presente: las 41 se
anotaron sobre candidatos que propuso MiniLM, así que **premian estructuralmente
a quien recupera lo mismo que MiniLM**. Cualquier cambio en el *primario* está
midiéndose contra un patrón que le juega en contra.

**Un cambio de primario necesita su propio pool anotado. Uno de re-puntuador
no**, porque reordena exactamente los mismos candidatos que el pool cubrió.

## Lo que sigue sin tener solución

Que el ground truth llegue a 50 **no arregla nada**: el MDE pasa de 0,059 a
0,049. La salida realista es cribar sobre 200-400 consultas etiquetadas por
agentes y confirmar solo las sobrevivientes contra las anotadas a mano —
midiendo primero el acuerdo (κ) entre agente y humano sobre las 41 que ya
existen, que es lo que decide si el cribado es utilizable.
