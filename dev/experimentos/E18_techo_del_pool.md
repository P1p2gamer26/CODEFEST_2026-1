# E18 — el pool NO es el cuello de botella, y eso da vuelta la nota vigente (8 ago 2026)

`dev/scripts/diagnostico_pools_e18.py`, profundidad 3000, las 50 consultas
evaluables (207 pares consulta-documento relevante). Coste: una pasada, cero
codificacion de pasajes.

## Recall del pool, que nunca se habia medido

Fraccion de los 207 pares cuyo documento aparece con **al menos un chunk**
dentro de la profundidad:

| rama | @100 | @200 | @500 | @1000 | @3000 | nunca |
|---|---|---|---|---|---|---|
| MiniLM, consulta cruda | 0.831 | 0.913 | 0.932 | 0.957 | 0.981 | 4 |
| **MiniLM, expandida (la entregada)** | 0.860 | **0.932** | 0.942 | 0.971 | 0.981 | 4 |
| gte, expandida | 0.879 | 0.932 | 0.976 | 0.981 | 0.990 | 2 |

**A la profundidad que ya usa la entrega, el pool contiene el 93.2% de los
documentos relevantes.** Y **no hay una sola consulta con pool ciego**: las 50
tienen al menos un documento relevante entre sus 200 candidatos.

## Lo que esto da vuelta

`dev/docs/PLAN_MAESTRO.md` cierra la seccion del reordenamiento con: *"ninguna estrategia que
reordene lo que el pool ya trajo va a mover la aguja. Lo que queda es la
construccion del pool."* **Es exactamente al reves.**

El F1@3 es 0.440 sobre un techo de 0.906, o sea que se pierde la mitad de lo
alcanzable. Y el 93.2% de los documentos que hacen falta **ya esta en el
pool**. La perdida no ocurre al recuperar: ocurre entre el pool de 200
candidatos y los 3 cupos de la respuesta. Es un problema de **agregacion y
ranking a nivel documento**, no de recuperacion.

Profundizar mas no compra casi nada: de 200 a 3000 candidatos el recall sube
de 0.932 a 0.981 — 10 pares mas de 207, a cambio de 15x de pool. Y E08 ya
midio que la profundidad extra es inerte.

## E16 queda REFUTADO por construccion, sin correrse

La hipotesis era que el vector mezclado ES+EN pierde documentos que la
consulta cruda si encuentra, y que unir los dos pools los recupera. Medido
directamente: a profundidad 200 el glosario **mete 4 documentos y saca 0**.

    mete  q001: F1-CSET-034, F3-SIPRI-075
    mete  q008: F1-CSET-009
    mete  q015: F1-CSET-076
    saca  (ninguno)

La union no puede rescatar nada porque no hay nada que rescatar. El fallo de
q021 que motivo la hipotesis era real, pero la entrada culpable ya se saco de
la tabla y las 12 que quedaron no producen el efecto. **Cerrado sin gastar una
corrida.** Es el mejor resultado posible de un experimento pre-registrado: la
justificacion mecanica era correcta y el dato dice que el mecanismo no esta
activo.

## E17 sigue vivo pero con techo conocido, y hay que decirlo antes de correrlo

A profundidad 200, de los 207 pares:

| | pares |
|---|---|
| los ven los dos encoders | 185 |
| **solo gte** | **8** |
| solo MiniLM | 8 |
| ninguno | 6 |

Los 8 que solo ve gte se concentran en tres consultas: q005 y q014
(`F1-CSET-076`), q007 (`F1-DAIO-029`, `F3-SIPRI-010/016/074`) y q008
(`F1-DAIO-001`, `F1-DAIO-027`).

**El techo de E17 es 8 pares de 207 (3.9%)**, y solo si la union los subiera
al top-3, que es justo lo que el resto de este experimento dice que el sistema
no sabe hacer. Ademas el reparto es **simetrico** (8 contra 8): unir los pools
mete 8 documentos y diluye el espacio para los otros 8. **Se conserva en la
cola, pero con la expectativa corregida a la baja y por escrito.**

## El detector de dispersion no tiene nada que detectar

`dev/docs/PLAN_MAESTRO.md` propone la dispersion del top-12 como senal barata de "la consulta
se resolvio por su mitad generica". Medida sobre las 50: mediana 0.049, rango
0.022-0.101, y **el grupo de consultas ciegas esta vacio** porque no hay
ninguna. No se puede separar lo que no existe en dos grupos. El detector queda
sin uso, no por malo sino por falta de casos.

## Lo que ordena la ronda siguiente

El experimento se corrio para saber a que fallo apuntar, y contesta: **no
apuntar a la recuperacion.** La palanca es lo que pasa entre los 200
candidatos y los 3 documentos — la agregacion. E07 barrio la estrategia
(`top5` resiste) y E14 propone el piso de longitud, pero **ninguno midio
primero donde caen los documentos relevantes en el ranking de documentos**.
Esa medicion es E19 y es el paso obvio: si estan en las posiciones 4-6, el
problema es de calibracion fina; si estan en la 40, la agregacion esta
descartandolos por una razon estructural.
