# E37 — re-calibrar `k_pool` y `topM` despues de E32

**Veredicto: NADA GANA. Se conserva `k_pool=100` + `top5`.**
Decimonoveno negativo del proyecto.

## Por que no era rebuscar

E32 se adopto hace una hora y cambia que chunks sobreviven al pool. `k_pool=100`
se fijo cuando ese filtro no existia, y E33 midio `topM` sobre la base
0.440/0.506, o sea sobre el sistema ANTERIOR a E32. Es el mismo argumento con el
que se adopto E01 (peso 0.25 -> 0.60): *"el peso se fijo con k_pool=60,
agregacion sum y sin glosario, y las tres cosas cambiaron"*. **Sin ese cambio de
regimen, re-barrer estos dos parametros seria la maquina de sobreajuste de la
leccion 2.**

Arnes: `dev/scripts/barrido_pool_topm_e37.py`, en memoria desde
`pools_entregados.json --con-similitudes` + `meta_crudos.json`. Cero FAISS.
`filtrar_por_fenomeno_dominante` y `ordenar_para_fragmentos` se llaman, no se
reimplementan. Base verificada digito a digito **antes** de mirar ninguna celda:
0.455/0.516/0.499 (50), 0.433/0.474/0.467 (10 indep.), 0.486/0.537/0.520 (41
hum.), 11 consultas con F1@3 = 0.

## FASE 1: el pool casi no encoge — la premisa es falsa

| k_pool | consultas donde el filtro actua | pool efectivo ahi (min / mediana / max) | media sobre las 50 |
|---|---|---|---|
| 100 | 19/50 | 84 / 94 / 99 | **97.4** |
| 150 | 21/50 | 124 / 139 / 149 | 145.0 |
| 200 | 26/50 | 162 / 188 / 199 | 192.1 |

**A `k_pool=100` el filtro se lleva entre 1 y 16 candidatos** (q018 es el peor
caso, 100 -> 84; siete de las 19 pierden menos de 5). La hipotesis pre-registrada
decia *"esas consultas agregan sobre un pool EFECTIVO mucho menor que 100"*: **es
falso, el pool efectivo medio es 97.4 de 100**. El experimento esta **muerto por
construccion en su propia premisa**, y lo correcto es decirlo antes de la tabla.
La fase 2 se corrio igual porque estaba pre-registrada y cuesta segundos, no
porque la puerta la dejara pasar.

Razon mecanica de que encoja tan poco: el voto ponderado por score solo dispara
con dominancia >= 0.8, y un pool tan dominado por un fenomeno ya casi no tiene
chunks de los otros dos que descartar. El filtro y el ancho del pool **no
interactuan**: es lo mismo que E32 midio, no un regimen nuevo.

## FASE 2: las nueve lecturas

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | NDp(ind) | F1(hum) | ND(hum) | NDp(hum) |
|---|---|---|---|---|---|---|---|---|---|
| **k100:top5 (entregada)** | **0.455** | **0.516** | **0.499** | **0.433** | **0.474** | **0.467** | **0.486** | **0.537** | **0.520** |
| k100:top8 | 0.405 | 0.490 | 0.474 | 0.367 | 0.431 | 0.418 | 0.433 | 0.498 | 0.483 |
| k150:top5 | 0.471 | 0.523 | 0.506 | 0.433 | 0.480 | 0.474 | 0.482 | 0.537 | 0.520 |
| k150:top8 | 0.364 | 0.460 | 0.443 | 0.367 | 0.444 | 0.432 | 0.384 | 0.474 | 0.457 |
| k200:top5 | 0.471 | 0.519 | 0.501 | 0.433 | 0.480 | 0.474 | 0.482 | 0.534 | 0.517 |
| k200:top8 | 0.336 | 0.423 | 0.411 | 0.367 | 0.444 | 0.438 | 0.358 | 0.443 | 0.433 |

### `top8`: pierde en las tres muestras y dispara el veto

`k100:top8` da F1(50) **−0.050 [−0.087, −0.020]**, 0 victorias contra 6 derrotas,
y sube las consultas con F1@3 = 0 de **11 a 13** -> veto. Empeora ademas con el
pool: `k150:top8` −0.091 y 16 ceros, `k200:top8` −0.119 y **20 ceros**. Es el
mecanismo ya conocido (`sum` sin tope se derrumba con pool ancho) apareciendo
antes de lo esperado: a M=8 el tope deja de morder y un documento con ocho chunks
mediocres desplaza al bueno. **E33 se confirma bajo el regimen nuevo.**

### `k_pool` 150/200 con `top5`: la ganancia es sesgo de pooling

Las dos celdas son casi la misma (20 y 22 lineas cambian, mismas 4 victorias y
2 derrotas de F1). Lecturas de `k150:top5`:

| | delta [IC 90%] | victorias | criterio |
|---|---|---|---|
| F1(50) | +0.017 [−0.018, +0.054] | 4g/2p | pasa (por 0.002) |
| ND(50) | +0.006 [−0.022, +0.032] | 8g/5p | **NO pasa** |
| NDp(50) | +0.007 [−0.020, +0.032] | 8g/6p | **NO pasa** |
| F1(ind) | +0.000 [+0.000, +0.000] | 0g/0p | inerte |
| ND(ind) | +0.006 [+0.000, +0.019] | 1g/0p | pasa |
| F1(hum) | **−0.004** [−0.038, +0.028] | 2g/2p | **NO pasa** |
| ND(hum) | −0.000 [−0.033, +0.027] | 6g/4p | **NO pasa** |

**El desglose lo mata.** Las 4 victorias de F1 son `q011, q012, q043, q047` y
**`q011` y `q012` son de `panel-agentes`**; las 2 derrotas (`q027`, `q030`) son
humanas. En las 41 humanas el reparto es **2-2 y el F1 baja a 0.482**: toda la
ganancia de las 50 vive en 2 de las 9 consultas peor etiquetadas. Es exactamente
la firma que E31 midio y por la que ya se descartaron `doc_rrf`, gte-primario y
200:top5 en agosto.

En las 10 independientes el F1 es **literalmente inerte** (0 victorias, 0
derrotas) y el NDCG sube +0.006, que con n=10 no distingue de cero.

**Ninguna celda pasa el criterio en las dos muestras. Con el empate declarado de
antemano, se conserva lo entregado.** Nada que aplicar a `Entrega/`.

## Lo que deja escrito

1. **El eje `k_pool` queda cerrado por tercera vez** (4 ago con `sum`/`top5`,
   E27 con el recorte intermedio, ahora bajo E32). Ampliar el pool solo mueve
   consultas de etiqueta de agente.
2. **E32 y el ancho del pool son ortogonales**, medido: el filtro se lleva 2,6%
   del pool en promedio. No hay que re-barrer nada mas "porque E32 cambio el
   regimen" — cambio el ranking, no el tamano efectivo.
3. **Trampa del arnes, que costo una corrida:** `pool_variante` de
   `barrido_cascada_e27_e28_e29.py` recorta a su propio `K_POOL=100` por dentro,
   asi que la primera corrida dio 150 y 200 **identicos a 100 (0 lineas
   cambian)** y parecia un resultado. Cualquier barrido futuro de ancho de pool
   tiene que construir el pool aparte.
