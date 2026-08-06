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
documentado en `las notas del proyecto`): no era su tamano ni su ventana, es que **la
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
