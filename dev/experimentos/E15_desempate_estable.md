# E15 — desempate estable por identificador: RECHAZADO (8 ago 2026)

Corrido en la maquina local mientras la VM trabajaba su propia cola, para no
pisarnos. `Entrega/` sin cambios (regla 6). Reproducible con
`dev/scripts/barrido_desempate_e15.py`; log en
`dev/intermedios/log_desempate_e15.txt`.

**Hipotesis, escrita antes de medir:** fijar un desempate explicito por
`chunk_id` (y `doc_id` en la agregacion) **no mueve ninguna de las nueve
lecturas**, y a cambio cierra la mitad reparable del problema de
`hallazgo_reproducibilidad.md`.

**Criterio de aceptacion, fijado antes de medir:** aqui no se pide ganar. Se
pide **no perder**: si el cambio mueve la metrica, entonces no es un desempate
sino una politica de ordenamiento nueva, y se rechaza.

## Lo medido

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| entregada | 0.440 | 0.490 | 0.476 | 0.400 | 0.436 | 0.468 | 0.510 |
| desempate | 0.440 | 0.480 | 0.465 | 0.400 | **0.385** | 0.468 | 0.498 |

Deltas pareados, IC al 90%: ND(50) **-0.010 [-0.031, +0.000]**, ND(ind)
**-0.051 [-0.153, +0.000]**, ND(hum) -0.012 [-0.037, +0.000]. **F1@3 no se
mueve ni una milesima en ninguna muestra**, que es lo esperable: los empates
estan entre fragmentos, no entre documentos.

**Veredicto: rechazado.** Falla el criterio en las seis lecturas de NDCG.

## Por que pierde, que es lo que vale del experimento

**Pierde UNA sola consulta en todas las muestras (0g/1p): q020, que pasa de
NDCG 1.000 a 0.489.** Y la causa no es el ordenamiento:

**Los cuatro fragmentos que cambian tienen texto byte a byte identico.**
`F2-CSIS-135` y `F2-CSIS-136` son el mismo documento duplicado en el corpus;
el desempate por `chunk_id` elige el 135 y el ground truth marca el 136.

O sea que **el -0.051 no mide calidad de recuperacion, mide cual copia de un
documento duplicado nombra la anotacion**. Es exactamente el caso que
`las notas del proyecto` ya tenia cerrado como indecidible: hay 27 grupos de documentos con
texto inicial identico, q020 y q032 gastan 2 de sus 3 cupos en un duplicado, y
el ground truth marca **una sola** copia en q020 y **las dos** en q032 — no hay
forma de saber que criterio uso ADL.

## Lo que queda en pie

1. **No adoptar.** No por la magnitud del delta, sino porque el unico efecto
   medible del cambio es cambiar de copia en un documento duplicado, y no
   tenemos ninguna razon para creer que la copia nueva sea la que ADL espera.
   Cambiar algo que solo mueve una moneda al aire es riesgo sin contrapartida.
2. **La reproducibilidad entre maquinas queda como esta**, y hay que decirla
   con su alcance real: determinista por maquina, 2 lineas de 50 de diferencia
   entre maquinas, metricas identicas. Ver
   `dev/experimentos/hallazgo_reproducibilidad.md`.
3. **Hallazgo colateral: los empates exactos son mas frecuentes de lo que
   sugeria el caso de q026.** El desempate toca 4 consultas (q020, q026, q028,
   q031), no las 2 que divergian entre maquinas. La causa de fondo es la misma
   en todas: documentos duplicados que producen scores identicos bit a bit.
4. **La prediccion de la hipotesis era falsa y hay que registrarlo asi.** Se
   predijo "no mueve ninguna lectura" y movio seis. La leccion 2 aplica a las
   propias predicciones, no solo a las mediciones ajenas.
