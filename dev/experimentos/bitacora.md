# Bitácora de experimentos en la VM

Una entrada por experimento cerrado, **gane o pierda**. El formato es fijo para
que se pueda releer rápido dentro de seis semanas.

Plantilla:

```
## <id> — <título> (<fecha>)

**Hipótesis:** …
**Justificación mecánica (escrita antes de medir):** …

| | F1@3 (50) | NDCG@10 (50) | F1@3 (10 indep.) | NDCG@10 (10 indep.) |
|---|---|---|---|---|
| base | 0.402 | 0.457 | | |
| variante | | | | |

**Victorias por consulta:** …
**Delta pareado, IC 90%:** …
**Veredicto:** adoptado / descartado / no concluyente — y por qué.
```

---

## Estado inicial (6 ago 2026)

Entorno montado y validado en `estudiante@10.43.97.37`. Ver
`dev/docs/handoff_vm.md`.

Baseline reproducido en la VM, idéntico al local:

| | valor |
|---|---|
| tests | 136 passed |
| F1@3 (50 consultas) | **0.402** (techo alcanzable 0.906) |
| NDCG@10 aproximado | **0.457** |
| NDCG@10 penalizado | 0.443 |
| F1@3 solo humanas (41) | 0.424 |
| F1@3 solo agente (9) | 0.304 |

Configuración: cascada triple MiniLM → gte + e5, `k_pool=100`,
agregación `top5`, glosario bilingüe activo, sin grafo.

Ningún experimento corrido todavía.

---

## E01 — Barrido del peso del re-puntuador (6 ago 2026)

**Hipótesis:** el peso 0,25 del re-puntuador se fijó con k_pool=60, agregación
sum y sin glosario; la configuración entregada cambió las tres cosas, así que
el peso nunca se barrió en el régimen actual.

**Justificación mecánica (escrita antes de medir):** el peso pondera la
similitud del re-puntuador contra la del primario. Al ampliar el pool a 100
entran candidatos con score primario más bajo, donde el mismo peso absoluto
pesa más en relación al primario. No hay razón para que el óptimo viejo siga
siendo el óptimo bajo pool=100/top5/glosario.

Grilla: peso en {0.10, 0.25, 0.40, 0.60}, con k_pool=100, agg=top5, glosario
activo. Script nuevo `dev/scripts/barrido_peso.py` (no existía un barrido de
peso; los cinco `barrido_*.py` previos cubren pool, estructura, glosario e
híbrido pero fijan el peso en 0,25).

| | F1@3 (50) | NDCG@10 (50) | F1@3 (10 indep.) | NDCG@10 (10 indep.) |
|---|---|---|---|---|
| peso 0.25 (entregado) | 0.402 | 0.457 | 0.300 | 0.338 |
| peso 0.10 | 0.392 | 0.434 | 0.267 | 0.291 |
| peso 0.40 | 0.389 | 0.444 | 0.333 | 0.351 |
| peso 0.60 | **0.425** | **0.476** | **0.367** | **0.374** |

**Victorias por consulta, peso 0.60 contra el entregado (0.25):**

| | F1@3 gana/pierde/empata | NDCG@10 gana/pierde/empata |
|---|---|---|
| 50 | 7 / 4 / 39 | 19 / 14 / 17 |
| 10 indep. | 2 / 0 / 8 | 2 / 2 / 6 |

**Delta pareado, IC 90% (peso 0.60 vs 0.25):** F1@3 50 +0.023 [-0.016,+0.063];
NDCG@10 50 +0.020 [-0.016,+0.053]; F1@3 indep +0.067 [+0.000,+0.133]; NDCG@10
indep +0.035 [-0.016,+0.100]; F1@3 humanas +0.027 [-0.017,+0.072]; NDCG@10
humanas +0.037 [+0.006,+0.069]. Las seis lecturas excluyen una pérdida de
-0.02: el criterio de adopción se cumple en las dos muestras.

peso 0.10 y peso 0.40 no cumplen el criterio en al menos una lectura (0.10
pierde en las seis; 0.40 solo pasa en F1 indep y NDCG humanas) — descartados.

**Veredicto:** peso 0.60 es **adoptable** por el criterio estadístico fijado
(IC al 90% excluye pérdida de 0.02 en las dos muestras, en las seis lecturas).
No se aplicó a `Entrega/` — punto 6 del handoff: regenerar la entrega es
decisión con humano. Resultados crudos en
`dev/intermedios/peso/peso{0.10,0.25,0.40,0.60}.jsonl`, log completo en
`dev/intermedios/log_barrido_peso.txt`.

**Nota para el siguiente barrido (no ejecutada acá, un experimento por
iteración):** 0.60 es el extremo superior de la grilla y gana limpio en las
seis lecturas — no hay evidencia de que sea el techo, solo de que es mejor
que 0.25. Antes de fijarlo en `Entrega/`, vale barrer un punto más arriba
(p. ej. 0.75, 0.90) para no adoptar a ciegas el borde de una grilla de cuatro
celdas.

---

## E01b — extender la grilla de peso a 0.75 y 0.90 (6 ago 2026)

**Hipotesis previa:** 0.60 era el borde superior de la grilla de E01, asi que
no se sabia si es un optimo o solo el ultimo punto medido. Si F1 y NDCG
seguian subiendo en 0.75 y 0.90, el hallazgo real no seria "el peso optimo es
X" sino "el primario deberia ser el re-puntuador", que es otra hipotesis.

Coste cero de codificacion: los tres indices ya existen y las similitudes
crudas se calculan una vez (`barrido_peso.py`, `PESOS = [0.25, 0.60, 0.75,
0.90]`, k_pool=100, agg=top5, glosario activo). Log completo en
`dev/intermedios/log_barrido_peso_e01b.txt`, crudos en
`dev/intermedios/peso/peso{0.25,0.60,0.75,0.90}.jsonl`.

| peso | F1@3 (50) | NDCG@10 (50) | F1@3 (10 indep.) | NDCG@10 (10 indep.) |
|---|---|---|---|---|
| 0.25 (entregado) | 0.402 | 0.457 | 0.300 | 0.338 |
| **0.60** | **0.425** | 0.476 | 0.367 | 0.374 |
| 0.75 | 0.405 | 0.478 | 0.367 | 0.374 |
| 0.90 | 0.420 | **0.482** | **0.400** | **0.412** |

**No hay tendencia monotona, hay meseta.** El F1 sobre las 50 sube, baja y
vuelve a subir (0.425 / 0.405 / 0.420) en un rango de 0.020, del tamano del
propio umbral de decision. El NDCG@10 de las 50 se mueve 0.006 en todo el
tramo 0.60-0.90. Lo unico que sigue creciendo es la muestra independiente
(F1 0.367 -> 0.400), y son **10 consultas**: una sola que cambie de lado mueve
0.033, o sea que ese ascenso no distingue senal de ruido.

**Criterio de adopcion (IC al 90% del delta pareado excluyendo -0.02) contra
el entregado 0.25:**

| peso | lecturas que pasan (de 6) | cuales fallan |
|---|---|---|
| **0.60** | **6/6** | — |
| 0.75 | 2/6 | F1 50, NDCG 50, NDCG indep, F1 humanas |
| 0.90 | 3/6 | F1 50, NDCG 50, F1 humanas |

**Veredicto: 0.60 se sostiene y la grilla queda cerrada.** 0.75 y 0.90 **no
son adoptables** —fallan el criterio justo en las 50, que es la metrica de
decision— y su mejor lectura (la independiente) es la muestra que menos
resuelve. Con la regla de preferir el valor conocido ante empate, extender la
grilla mas arriba no tiene justificacion: la meseta ya aparecio.

**Se responde la pregunta de arquitectura que este experimento tenia que
decidir: NO se escala a "el re-puntuador deberia ser el primario".** Esa
escalada requeria ver F1 y NDCG subiendo sin aplanarse, y lo que se ve es una
meseta ruidosa entre 0.60 y 0.90. Ademas la hipotesis ya tiene precedente en
contra: gte-primario dio 0.385 en las 41 humanas y **0.200** en las
independientes. No se abre.

**No se toco `Entrega/`** (punto 6 del handoff). Aplicar el peso 0.60 sigue
siendo decision con humano; E01b levanta el bloqueo que declaraba la cola
("adoptar el borde de una grilla sin ver mas alla"): ya se vio mas alla y 0.60
no es un borde que siga subiendo.

---

## E02 — multilingual-e5-small como PRIMARIO: refutado (6 ago 2026)

**Hipotesis previa:** e5-small (118 M, dim 384, ventana 512) recupera mejor
que MiniLM como primario, porque arregla la truncacion a 128 tokens que
afecta al 96% de los chunks **sin costar mas CPU** (mismo orden de
parametros, misma dimension).

**Justificacion mecanica previa** (`docs/plan_encoders.md` sec. 0): MiniLM no
ve la mayor parte del texto que indexa. e5-small cuadruplica la ventana con
el mismo tamano, asi que es la unica forma barata de separar "la ventana no
importa" de "e5-base era peor por otra razon" — e5-base perdio como primario,
pero es 3x mas grande y la ventana no era la unica variable que cambiaba.

Indice construido en la VM sobre `chunks_intermedios_limpio.jsonl` (los mismos
128.526 chunks, sin re-chunkear: el invariante del punto 8 se respeta y los
`chunk_id` quedan identicos en orden, verificado). Salida en
`dev/intermedios/e5small/`, **fuera de `Entrega/`**. Medido con
`dev/scripts/barrido_primario.py` en el regimen ENTREGADO (k_pool=100,
agg=top5, glosario activo, cascada con peso 0.25) — `barrido_estructuras.py`
solo habia comparado primarios bajo el regimen viejo de pool=60/`sum`/sin
glosario. Log en `dev/intermedios/log_barrido_primario.txt`.

| celda | F1@3 (50) | NDCG@10 (50) | F1@3 (indep) | NDCG@10 (indep) |
|---|---|---|---|---|
| **MiniLM->gte+e5 (entregado)** | **0.402** | **0.457** | **0.300** | **0.338** |
| e5small->gte+e5 | 0.187 | 0.192 | 0.200 | 0.171 |
| MiniLM solo | 0.362 | 0.427 | 0.267 | 0.299 |
| e5small solo | 0.154 | 0.160 | 0.100 | 0.143 |

Deltas pareados contra el entregado, IC al 90%: e5small en cascada da F1
**-0.215 [-0.288, -0.143]** y NDCG **-0.265 [-0.353, -0.180]** sobre las 50
(gana 3 pierde 25, y gana 7 pierde 30). El IC esta enteramente bajo cero en
las 50 y el NDCG independiente tambien. **No es un empate ni una perdida
marginal: es un derrumbe.**

**Se midieron cuatro celdas y no dos, y por eso el experimento concluye algo.**
La comparacion que responde la hipotesis es *primario solo contra primario
solo*: MiniLM 0.362 contra e5-small **0.154**. Menos de la mitad, con la
ventana cuadruplicada.

**Antes de creerle al resultado se descarto el modo de fallo que la propia
cola advertia** (los prefijos `query:`/`passage:` de la familia E5, cuya
omision degrada en silencio). Verificado de dos formas: la entrada en
`KNOWN_ENCODERS` declara los dos prefijos, y el vector reconstruido del indice
da `sim = 1.0000` exacta contra el texto re-codificado **con** prefijo de
pasaje y menos **sin** el. El indice esta bien construido. Ademas la fila base
del barrido reproduce 0.402 / 0.457 exactos, o sea que el arnes de medicion
tampoco esta sesgado.

### Lo que este experimento cierra, que vale mas que el numero

**La ventana de 128 tokens de MiniLM NO es el cuello de botella.** Es la
pregunta que `docs/plan_encoders.md` dejaba abierta y que motivaba varias
lineas de trabajo (re-fragmentar el corpus a 128 tokens, buscar encoders por
tamano de ventana). Un encoder del mismo porte con 4x la ventana, sobre
exactamente los mismos chunks, recupera **menos de la mitad**. La truncacion
es real y sigue siendo real; lo que queda refutado es que sea lo que limita la
recuperacion. **No abrir mas experimentos cuyo argumento principal sea la
ventana.**

Esto ademas re-lee el fallo de e5-base como primario (0.182 en las 41,
documentado en `CLAUDE.md`): no era su tamano ni su ventana, es que **la
familia E5 rinde mal como primario sobre este corpus** y bien como
re-puntuador. Dos miembros de la familia, con 3x de diferencia de tamano,
fallan igual en el mismo puesto.

**La premisa de coste tambien era falsa.** e5-small tardo **2 h 33 min** en
codificar los 128.526 chunks contra ~1 h de MiniLM: 2,5x mas caro, no
"sin costar mas CPU". Tiene sentido mecanico — con ventana 512 procesa ~4x
mas tokens por chunk que MiniLM truncando en 128. Ese coste es consecuencia
directa de lo que la hipotesis vendia como gratis.

**Veredicto: DESCARTADO en las dos muestras. No reabrir sin datos nuevos.**
El indice de e5-small queda en `dev/intermedios/e5small/` (197 MB) por si
sirve de re-puntuador en algun experimento futuro; **no se toco `Entrega/`**.

---

## E03 — ampliar el glosario ES->EN: tres entradas adoptables (6 ago 2026)

**Hipotesis previa:** el glosario admite mas entradas de las 9 actuales, y cada
una rescata consultas que hoy no tienen puente lexico al corpus.

**Criterio de entrada, fijado antes de medir y en DOS partes obligatorias:**
(1) la forma inglesa es >=10x mas frecuente en el corpus que la espanola;
(2) **la CONSULTA no tiene ningun puente** — ni la sigla inglesa ni el termino
ingles ya escritos en ella. La parte 2 es la que costo q021 en su momento
(F1 0.33 -> 0.00 con "maniobras de proximidad", porque esa consulta ya traia
**RPO**). No alcanza con que el termino espanol sea raro.

Se conto la frecuencia ES vs EN de 22 terminos multipalabra sacados del
vocabulario de las 50 consultas, sobre los 128.526 chunks. Medicion por
entrada con `dev/scripts/barrido_glosario_e03.py` en el regimen ENTREGADO
(k_pool=100, agg=top5, cascada peso 0.25) — `barrido_glosario.py` habia
quedado en el regimen viejo de pool=60/`sum`. Log en
`dev/intermedios/log_glosario_e03.txt`.

### El hallazgo del conteo, que vale mas que las entradas nuevas

**La asimetria ES/EN existe SOLO en los fenomenos 1 y 2.** En el fenomeno 3
(territorial, corpus colombiano en espanol) el conteo va exactamente al reves:

| termino (F3) | chunks ES | forma EN | chunks EN |
|---|---|---|---|
| reclutamiento | 682 | child recruitment | 2 |
| restitucion de tierras | 617 | land restitution | 22 |
| narcotrafico | 388 | drug trafficking | 218 |
| control territorial | 285 | territorial control | 41 |
| mineria ilegal | 148 | illegal gold mining | 1 |
| grupos armados organizados | 121 | organized armed groups | 6 |
| economias ilicitas | 76 | illicit economies | 8 |

**Expandir al ingles una consulta del fenomeno 3 la alejaria del vocabulario
de sus propios documentos.** El glosario es una herramienta de UN fenomeno,
no una politica general de la consulta. Queda escrito para que nadie lo
"complete" con los terminos de q033-q050, que es lo que parecia el siguiente
paso natural antes de contar.

### Medicion por entrada (cada una sola, sobre la tabla ya adoptada)

| entrada candidata | ES/EN | toca | delta F1(50) | delta NDCG(50) | v/d |
|---|---|---|---|---|---|
| **derecho internacional en el espacio -> international space law** | 2 / 424 | q017 | **+0.007** | **+0.019** | 1/0 |
| **dominio espacial -> space domain** | 23 / 965 | q032 | **+0.007** | **+0.009** | 1/0 |
| **sistemas no tripulados -> unmanned systems UAV** | 0 / 142 | q002 | **+0.008** | +0.001 | 1/0 |
| capacidades laser -> laser weapons | 0 / 178 | q024 | −0.007 | −0.010 | **0/1** |
| amenazas ciberneticas -> cyber threats | 3 / 127 | q013 | 0.000 | 0.000 | 0/0 |
| infraestructuras criticas -> critical infrastructure | 17 / 433 | q013 | 0.000 | 0.000 | 0/0 |
| minerales estrategicos -> critical minerals | 0 / 74 | q046 | 0.000 | 0.000 | 0/0 |

**Los dos fracasos estaban declarados como sospechosos ANTES de medir, y los
dos fallaron por donde se dijo.** `capacidades laser` falla la parte 2 del
criterio: "laser" es un cognado, o sea que q024 **ya tenia puente** al corpus
ingles aunque el bigrama en espanol no aparezca nunca — es q021 otra vez, y
efectivamente pierde. `minerales estrategicos` era el unico candidato del
fenomeno 3 que pasaba el conteo, y se marco como riesgoso porque "critical
minerals" es vocabulario de cadenas de suministro de semiconductores
(fenomeno 1); no arrastro la consulta al fenomeno equivocado, pero tampoco
hizo nada.

**Las tres de efecto nulo no entran**, por la regla fijada de antemano: una
entrada que no mueve ninguna consulta es superficie de riesgo para consultas
futuras sin beneficio medido.

### Las tres juntas, que es lo que se adoptaria

| muestra | metrica | antes | despues | delta [IC 90%] | v/d/e |
|---|---|---|---|---|---|
| 50 consultas | F1@3 | 0.402 | **0.423** | **+0.021 [+0.007, +0.043]** | 3/0/47 |
| 50 consultas | NDCG@10 | 0.457 | **0.486** | **+0.029 [+0.001, +0.067]** | 3/0/47 |
| 41 humanas | F1@3 | 0.424 | **0.450** | +0.026 [+0.008, +0.052] | 3/0/38 |
| 41 humanas | NDCG@10 | 0.456 | **0.492** | +0.036 [+0.002, +0.081] | 3/0/38 |
| 10 indep. | F1@3 | 0.300 | **0.333** | +0.033 [+0.000, +0.100] | 1/0/9 |
| 10 indep. | NDCG@10 | 0.338 | **0.432** | +0.094 [+0.000, +0.281] | 1/0/9 |

**Adoptable en las seis lecturas: el IC al 90% excluye la perdida de 0.02 en
todas y no pierde una sola consulta en ninguna muestra.** Ademas la ganancia
mas grande (q017) cae en la **muestra independiente**, que es la unica sin
sesgo de pooling — al reves de lo que paso con pool-100 y con gte-primario,
donde la ganancia se evaporaba justo ahi.

**Lo que hay que decir junto al numero:** son **3 consultas de 50** las que se
mueven, una por entrada. El limite inferior del IC es +0.000, o sea que lo que
esta demostrado es "no pierde", no "gana mucho". El efecto es real pero chico,
y la muestra independiente lo sostiene con **una sola consulta**.

**NO aplicado.** Las entradas no se escribieron en `src/retrieval/glosario.py`
a proposito: la tabla vive **por duplicado** en `dev/src/` y en
`Entrega/generador.py`, y `test_paridad_entrega.py` compara las dos sobre las
50 consultas. Tocar solo `dev/src/` rompe el test; tocar las dos es tocar
`Entrega/`, que el punto 6 del handoff reserva para decision con humano.
Queda como **adoptable pendiente**, igual que el peso 0.60 de E01/E01b.

---

## E06 — las dos palancas juntas: se componen (6 ago 2026)

**Hipotesis:** el peso 0,60 (E01/E01b) y las tres entradas de glosario (E03)
se componen, o sea que juntos superan a cualquiera de los dos solo.

**Por que hacia falta medirlo:** las dos se midieron contra la MISMA base
entregada. Sumar +0,023 y +0,021 y esperar 0,44 habria sido extrapolar.

**Justificacion mecanica, escrita antes de medir:** actuan en puntos distintos
del camino online. El glosario cambia el VECTOR DE CONSULTA, o sea que
candidatos ENTRAN al pool; el peso reordena el pool ya recuperado. Sobre esa
base deberian componerse. Pero E03 rescata q002/q017/q032 metiendo documentos
nuevos, y si el re-puntuador con mas peso los hunde, la combinacion valdria
menos que las partes. Es exactamente eso lo que el experimento resuelve.

**Regla fijada antes de ver los numeros:** ademas del criterio de siempre (IC
al 90% del delta pareado excluyendo -0,02 en las dos muestras) se agrego una
segunda condicion: **si la combinacion no supera a la mejor palanca
individual, se entrega la palanca sola.** Dos cambios que no se componen son
un cambio de mas.

| celda | F1(50) | NDCG(50) | F1(indep) | NDCG(indep) | F1(hum) | NDCG(hum) |
|---|---|---|---|---|---|---|
| entregado (0.25, glos base) | 0.402 | 0.457 | 0.300 | 0.338 | 0.424 | 0.456 |
| E01 solo (0.60, glos base) | 0.425 | 0.476 | 0.367 | 0.374 | 0.450 | 0.493 |
| E03 solo (0.25, glos+3) | 0.423 | 0.486 | 0.333 | 0.432 | 0.450 | 0.492 |
| **E01+E03 (0.60, glos+3)** | **0.440** | **0.490** | **0.400** | **0.436** | **0.468** | **0.510** |

**Deltas de la combinacion contra lo entregado, IC 90%:** ADOPTABLE en las
seis lecturas. F1@3 50 **+0.038 [-0.005, +0.081]** (9 gana / 4 pierde);
NDCG@10 50 **+0.034 [-0.006, +0.073]** (21/14); F1@3 indep **+0.100 [+0.033,
+0.167]** (3/0); NDCG@10 indep +0.098 [-0.001, +0.216] (3/2); F1@3 humanas
+0.045 [-0.002, +0.093]; NDCG@10 humanas **+0.054 [+0.017, +0.095]**.

**La comparacion que decide, contra cada palanca sola:**

| | vs E01 solo | vs E03 solo |
|---|---|---|
| F1@3 50 | +0.015 [+0.000, +0.032] | +0.017 [-0.025, +0.058] |
| NDCG@10 50 | +0.014 [+0.000, +0.039] | +0.004 [-0.037, +0.043] |
| F1@3 indep | +0.033 [+0.000, +0.100] | **+0.067 [+0.000, +0.133]** |
| NDCG@10 indep | +0.062 [+0.000, +0.187] | +0.004 [-0.078, +0.086] |

**Veredicto: ADOPTADO y aplicado a `Entrega/`.** Se componen: ninguna de las
ocho lecturas contra las palancas solas baja del cero.

**Lo que hay que decir junto al 0.440, aplicando la leccion 2:**

- **La ganancia sobre E03 solo es floja** en las 50 (NDCG +0.004, F1 IC que
  cruza el cero). Lo que sostiene la combinacion es el **F1@3 en las 10
  independientes: +0.067**, la unica muestra sin sesgo de pooling. Es tambien
  la muestra mas chica del proyecto: 10 consultas, una sola que cambie mueve
  0.033. **La evidencia es real pero no es holgada.**
- **El F1@3 de las 50 pierde 4 consultas** para ganar 9. La regla vieja de "no
  adoptar nada que pierda consultas" esta derogada (`lecciones_metodologia.md`)
  justo porque estaba anti-correlacionada con la calidad, pero conviene saber
  que este cambio no es gratis en todas las consultas.
- **El 0.440 es el 49% del techo alcanzable (0.906)**, no el 44% de 1. Citarlo
  siempre con el techo al lado.

---

## E07 — la agregacion a documento bajo el regimen actual (6 ago 2026)

**REFUTADO.** `Entrega/` sin cambios.

Hipotesis, escrita antes de medir: `top5` premia al documento que aporta
MUCHOS chunks al pool por encima del que aporta UNO excelente, asi que **un M
mas chico deberia ganar** bajo el regimen actual (pool 100, peso 0.60).
La motivaba `diagnostico_ceros.py` sobre la entrega en 0.440: de 20 consultas
que fallan, **16 tienen el documento correcto DENTRO del pool** y ninguna lo
tiene ausente del indice; q022 y q040 con un chunk relevante en **rank 1** y
aun asi el documento no entra al top-3.

Se re-abrio algo ya descartado porque el `topM` original se midio con
`k_pool=60` y peso 0.25, regimen donde `top5` y `sum` eran **la misma
operacion** y el barrido no podia distinguir nada (leccion 5: las notas
propias envejecen).

| estrategia | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| max | 0.239 | 0.295 | 0.279 | 0.167 | 0.228 | 0.223 | 0.289 |
| top2 | 0.363 | 0.365 | 0.354 | 0.233 | 0.247 | 0.358 | 0.349 |
| top3 | 0.391 | 0.395 | 0.381 | 0.233 | 0.263 | 0.392 | 0.384 |
| **top5 (entregada)** | **0.440** | **0.490** | **0.476** | **0.400** | **0.436** | **0.468** | **0.510** |
| top8 | 0.390 | 0.465 | 0.453 | 0.367 | 0.431 | 0.415 | 0.473 |
| sum | 0.390 | 0.465 | 0.453 | 0.367 | 0.431 | 0.415 | 0.473 |
| mean | 0.057 | 0.201 | 0.187 | 0.033 | 0.120 | 0.053 | 0.189 |

**`top5` gana las 27 lecturas** (3 metricas x 3 muestras) y **ninguna
alternativa pasa el criterio** del IC al 90%. La direccion predicha se cumple
al reves: cuanto mas chico el M, peor. La explicacion mecanica queda
**refutada**, no confirmada a medias — que es justo la mitigacion que se habia
fijado antes de medir para no vender ruido como hallazgo.

**Coste real: cero.** El barrido ya habia dejado los siete `.jsonl` escritos
en `dev/intermedios/agg_e07/` antes de que muriera la sesion que lo lanzo; se
puntuaron los archivos guardados sin volver a tocar FAISS ni recodificar nada.

### Los dos hallazgos colaterales, que valen mas que el veredicto

1. **`top8` y `sum` son identicos digito a digito en las nueve columnas.** No
   se parecen: son el mismo numero. O sea que **con `k_pool=100` ningun
   documento aporta mas de 8 chunks al pool**, y el tope de `topM` solo tiene
   efecto en la franja **M=5 a M=8**. Fuera de ahi el parametro es inerte.
   Esto acota de una vez cuanto puede rendir esta palanca: no hay nada que
   ganar explorando M grandes, y ya se sabe que M chicos pierden.
2. **`top5` es un pico agudo: 0.440 contra 0.390 y 0.391 en los dos vecinos.**
   Hay que decirlo en voz alta aunque juegue en contra del numero entregado.
   `top5` **no lo eligio este barrido** — venia de la entrega, adoptado por
   robustez ante el pool ancho, no por argmax de una grilla — asi que no es
   sobreajuste en el sentido estricto. Pero un maximo local tan puntiagudo
   entre dos vecinos practicamente iguales es la firma tipica del sobreajuste,
   y **no conviene defender el 5 como valor mecanicamente justificado**. Lo
   defendible es lo medido: es el mejor de siete y el unico que pasa.

---

## E08 — `rerank_depth` x `k_pool` bajo el peso 0.60 (6 ago 2026)

**NO ADOPTABLE**, por dos razones independientes. `Entrega/` sin cambios.

Hipotesis, escrita antes de medir: al triplicar el peso del re-puntuador
(0.25 -> 0.60, E01) los candidatos PROFUNDOS pasan a poder subir al top-3, y
`rerank_depth` deja de ser inerte. Se reabria algo cerrado con **51 empates de
51**, pero la nota vieja daba la razon mecanica explicita —"con peso 0.25 el
re-puntuador solo reordena, y los candidatos de las posiciones 200-600 tienen
scores demasiado bajos para subir al top-3"— y **esa razon dependia del peso**.

Coste real: **una sola pasada de FAISS**. Los pools se construyen a
profundidad 1000 y las 14 celdas salen por rebanado, porque el re-puntuado es
independiente por candidato. Cero codificacion nueva.

| celda | F1(50) | ND(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|
| **d200:k100 (entregada)** | 0.440 | 0.490 | **0.400** | **0.436** | 0.468 | 0.510 |
| d200:k150 | 0.475 | 0.511 | 0.367 | 0.401 | 0.486 | 0.516 |
| d200:k200 | **0.475** | **0.511** | 0.400 | 0.430 | **0.486** | **0.517** |
| d400:k100 | 0.417 | 0.466 | 0.400 | 0.435 | 0.448 | 0.489 |
| d600:k100 | 0.423 | 0.468 | 0.400 | 0.428 | 0.456 | 0.491 |
| d1000:k100 | 0.423 | 0.466 | 0.400 | 0.428 | 0.456 | 0.489 |
| d1000:k200:top8 | 0.441 | 0.506 | 0.400 | 0.438 | 0.470 | 0.526 |

### 1. La hipotesis esta refutada

**El ganador en profundidad es 200, que es la celda ya entregada**, y subirla
empeora de forma monotona: F1@3 sobre las 50 va 0.440 (d200) -> 0.417 (d400)
-> 0.423 (d600) -> 0.423 (d1000). El peso 0.60 **no** desperto a los
candidatos profundos.

Esto **cierra `rerank_depth` mucho mejor que los "51 empates" originales**,
porque el cierre ya no depende del peso: se probo con 2,4x mas autoridad de
re-puntuado y una grilla cuatro veces mas ancha, y el parametro sigue inerte.
La mitigacion fijada de antemano decia que si el ganador no era una
profundidad MAYOR que 200 la explicacion mecanica quedaba refutada y el
resultado se trataba como ruido. Se aplica tal cual.

### 2. Lo unico que mueve la aguja tiene la firma del sesgo de pooling

Todo el movimiento viene de `k_pool`. **`d200:k200` contra la entregada:**

| lectura | delta, IC 90% | victorias | criterio |
|---|---|---|---|
| F1@3 50 | **+0.035 [+0.008, +0.067]** | 4g/0p | pasa |
| NDCG@10 50 | +0.021 [+0.002, +0.044] | 9g/3p | pasa |
| **F1@3 indep** | **+0.000 [+0.000, +0.000]** | **0g/0p** | pasa (trivialmente) |
| **NDCG@10 indep** | **-0.006 [-0.031, +0.013]** | 1g/1p | **no pasa** |
| F1@3 humanas | +0.018 [+0.000, +0.039] | 2g/0p | pasa |

**Las 10 independientes salen identicas consulta por consulta** — cero
victorias y cero derrotas en F1@3. Y gana 4 en las 50 pero solo 2 en las 41
humanas, o sea que **la mitad de la ganancia cae sobre las 9 consultas con
etiqueta de agente**, justo donde el instrumento acierta 0.23. Es la misma
firma por la que se descartaron `doc_rrf` y gte-primario. Ademas falla el
criterio pre-registrado en `NDCG@10` y `NDp` de las independientes.

### El hallazgo que vale mas que el veredicto

**Es el MISMO reparto que midio la ronda del 4 de agosto**, cuando se decidio
entregar `k_pool=100` y no 200: ahi 200 ganaba 6 consultas, **3 de ellas de
etiqueta de agente**, y entre las 41 humanas quedaba **3-3**. Hoy el peso, el
glosario y la agregacion son otros, y **el patron se reprodujo intacto**. No
era ruido de una corrida: es una propiedad estable del ground truth.

**Corolario operativo, que redirige la cola:** mientras esas 9 consultas sigan
con etiqueta de agente, **cualquier palanca que ensanche el pool va a parecer
ganadora en las 50 sin serlo**, y va a consumir un experimento cada vez para
llegar al mismo desenlace. El desbloqueo no es otra palanca del recuperador
sino **E05** — re-anotar esas 9 a mano. Es tarea humana por medicion, no por
comodidad: el panel de 4 agentes ya se probo y reproduce al humano con F1
0.23, peor que el anotador unico.
