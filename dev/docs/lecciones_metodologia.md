# Lecciones de método

Escrito el 2 de agosto de 2026, después de una jornada en la que se probaron
siete hipótesis de mejora y **seis fallaron**. Lo que sigue no es la lista de
lo que se intentó —eso está en la sección "Medido y descartado" de
`las notas del proyecto`— sino **cómo se decidió que fallaban**, que es lo reutilizable.

---

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

## Lo que este método NO puede resolver

Con ~41 consultas anotadas, la mayoría de los efectos reales caen dentro del
ruido de muestreo. Cuando siete experimentos seguidos dan "no concluyente", la
lectura correcta **no** es "no hay nada que mejorar": es **"la muestra no
resuelve diferencias de este tamaño"**.

La única salida es más ground truth. Todo lo demás es refinar la medición de
algo que no se puede medir.
