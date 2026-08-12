# E20 — el conteo de chunks NO es un sesgo, es la senal (8 ago 2026)

`dev/scripts/barrido_cupo_chunk_e20.py`, cinco celdas sobre el mismo pool.
La hipotesis de E19 esta **refutada, y por goleada en las nueve lecturas.**

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| **entregada** (suma top5) | **0.440** | **0.490** | **0.476** | **0.400** | **0.436** | **0.468** | **0.510** |
| cupo 2+1 (un cupo por mejor chunk) | 0.322 | 0.378 | 0.364 | 0.267 | 0.322 | 0.350 | 0.393 |
| cupo 1+2 (dos cupos por mejor chunk) | 0.280 | 0.335 | 0.319 | 0.167 | 0.231 | 0.273 | 0.335 |
| media de los top5 | 0.115 | 0.245 | 0.229 | 0.100 | 0.167 | 0.115 | 0.243 |
| suma top5 / raiz del conteo | 0.440 | 0.490 | 0.476 | 0.400 | 0.436 | 0.468 | 0.510 |

Deltas pareados con IC al 90%, todos enteramente bajo cero:

    cupo 2+1    F1(50) -0.118 [-0.163, -0.072]   2 gana / 19 pierde
    cupo 1+2    F1(50) -0.160 [-0.230, -0.088]   7 gana / 25 pierde
    media top5  F1(50) -0.325 [-0.395, -0.255]   1 gana / 33 pierde

## La refutacion es MONOTONA en la direccion que importa

Ordenadas por cuanto peso le quitan al conteo:

    suma top5 (lineal en el conteo)     0.440   <- lo entregado
    cupo 2+1  (un tercio por chunk)     0.322
    cupo 1+2  (dos tercios por chunk)   0.280
    media top5 (conteo invisible)       0.115

**Sin un solo cruce.** Cuanto menos cuenta el conteo, peor va todo. El riesgo
pre-registrado —"si 1+2 no es peor que 2+1, el efecto es ruido"— se resolvio
al reves de lo temido: 1+2 **si** es peor que 2+1, o sea que el efecto es real
y apunta en contra de la hipotesis. No es ruido, es una senal invertida.

## Donde se equivoco E19, que es lo que hay que recordar

E19 midio bien y concluyo mal. El dato era: los relevantes que pierden tienen
mejor chunk de 1.647 contra 1.672 de los que ganan, 1.5% de diferencia. La
inferencia fue "entonces el conteo los esta hundiendo injustamente".

**La lectura correcta es la contraria: el mejor score de chunk no discrimina
casi nada, y el conteo si.** Que dos poblaciones tengan la misma mediana en
una variable no significa que esa variable este siendo ignorada
injustamente — significa que esa variable no sirve para separarlas. Ordenar
por ella es ordenar por ruido, y eso es exactamente lo que hacen las celdas
que pierden.

Que un documento aporte 7 chunks a un pool de 100 no es "inundar": es que
siete pasajes distintos de ese documento se parecen a la consulta, que es
justamente lo que `sum` fue elegido para premiar en su dia y lo que E07
confirmo al barrer M.

## El resultado por identidad, que vale mas que la tabla

**`suma top5 / raiz del conteo` produce un archivo byte a byte identico a la
entrega: 0 lineas distintas de 50.** No es casualidad ni un error del arnes.
E19 midio que el 92.7% de los documentos que ocupan cupo saturan el tope de
top5; para todos ellos el divisor es la misma constante `sqrt(5)`, asi que el
orden relativo no puede cambiar. **El top-3 sale casi siempre del conjunto
saturado**, y cualquier normalizacion que trate por igual a los saturados es
inerte por construccion.

Eso tambien explica por que la media SI destruye todo aunque comparta la forma:
para los documentos de menos de 5 chunks divide por menos, o sea que **premia
al que aporta un solo chunk**. Es `max` con otro nombre, y `max` ya habia
perdido 0.226 contra 0.306 en su momento. Se reconfirma con pool 100, cascada
y peso 0.60: no era un artefacto de la configuracion vieja.

## Lo que queda cerrado

**La familia entera de la agregacion a documento esta agotada.** E07 barrio el
parametro M, E20 barre la forma funcional en sus tres regimenes (lineal en el
conteo, sublineal, invariante) y el maximo esta en el extremo lineal, que es
lo entregado. **No reabrir sin datos nuevos**, y en particular no volver con
"normalizar por longitud del documento" ni "penalizar documentos largos": son
la misma idea con otro disfraz y el barrido ya cubre sus tres regimenes.

Con E18 (el pool trae el 93.2%) y E20 juntos, el cuadro es incomodo y hay que
decirlo: **ni la recuperacion ni la agregacion tienen margen visible con el
instrumento actual.** El F1@3 es 0.440 sobre un techo de 0.906 y la mitad que
falta no esta en ninguno de los dos sitios donde se busco. Lo que E19 sigue
sosteniendo es que el 29% de los relevantes cae en las posiciones 4-8 — cerca,
pero el orden ahi lo decide algo que ninguna de las cinco celdas captura.
