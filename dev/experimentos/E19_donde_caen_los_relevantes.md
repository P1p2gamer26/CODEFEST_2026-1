# E19 — el ranking de documentos lo decide el CONTEO de chunks, no la calidad (8 ago 2026)

`dev/scripts/diagnostico_rank_doc_e19.py`, configuracion identica a la entrega
(prof 200, k_pool 100, top5, peso 0.60). 187 de los 207 pares relevantes caen
dentro del pool de 100; 20 se pierden al recortar de 200 a 100.

## Donde caen

| posicion en el ranking agregado | pares | |
|---|---|---|
| 1 | 24 | 12.8% |
| 2 | 19 | 10.2% |
| 3 | 20 | 10.7% |
| **4-5** | **27** | **14.4%** |
| **6-8** | **28** | **15.0%** |
| 9-12 | 21 | 11.2% |
| 13-20 | 24 | 12.8% |
| 21-30 | 13 | 7.0% |
| 31+ | 11 | 5.9% |

Solo **63 de 187 (33.7%)** entran en los 3 cupos. La mediana de los que quedan
fuera es la **posicion 10**. La hipotesis se confirma a medias y eso importa:
**el 29% de los relevantes cae en las posiciones 4-8**, o sea a un pelo. Pero
hay una cola larga real — el 26% cae mas alla de la posicion 12.

Cota de lo que compraria reordenar, si el reto diera mas cupos:

    @3  33.7%   @4  39.6%   @5  48.1%   @6  53.5%   @8  63.1%   @10  68.4%

## Por que pierden, que es el hallazgo

| | n | chunks aportados al pool (mediana) | mejor score de chunk (mediana) |
|---|---|---|---|
| los 3 que ocupan cupo | 150 | **7.0** | 1.672 |
| todos los relevantes | 187 | 4.0 | 1.665 |
| **los relevantes que pierden** | 124 | **3.0** | **1.647** |

**El mejor chunk del que pierde vale casi lo mismo que el del que gana: 1.647
contra 1.672, una diferencia del 1.5%.** Lo que los separa es el conteo.

    saturan el tope de top5 (aportan >= 5 chunks al pool)
      los que ocupan cupo          92.7%
      los relevantes que pierden   18.5%

Y los doce que caen mas lejos —posiciones 30 a 52— **aportan exactamente 1
chunk cada uno**, varios con scores excelentes: `F1-DAIO-031` puntua 1.686 y
queda en la posicion 31; `F2-CSIS-178` puntua 1.680 y queda en la 42.

## La aritmetica que lo explica, y por que `top5` no lo arreglo

Bajo suma de los 5 mejores, un documento con un solo chunk excelente vale
**1.68**. Uno con cinco chunks mediocres vale **5 x 1.6 = 8.0**. **No hay
ningun chunk lo bastante bueno para compensar la diferencia de conteo**: el
techo de un documento con un chunk es una quinta parte del piso de uno con
cinco.

`top5` se adopto justamente para contener esto —`sum` sin tope se derrumbaba a
0.298— y lo contiene, pero solo acota el problema, no lo invierte. E07 barrio
el valor de M y encontro que el tope "solo muerde entre 5 y 8". Lo que ninguno
de los dos toco es la **normalizacion**: el score sigue creciendo linealmente
con el conteo hasta M.

## Lo que esto le hace a E14

E14 propone un piso de longitud de chunk en la agregacion, con la teoria de
que chunks cortos y basura inflan el score. **El dato dice que el problema no
es la calidad de los chunks que suman sino cuantos suman.** Los que pierden
tienen mejor score casi identico; no los esta hundiendo un titulo de seccion,
los esta hundiendo tener 3 chunks en vez de 7. E14 no esta refutado —un piso
podria quitarle chunks al inundador— pero **apunta al sintoma de al lado** y su
prioridad baja.

## Lo que abre: E20

La palanca es la **forma funcional de la agregacion**, no su parametro. Dos
familias, ninguna medida nunca:

1. **Normalizar por el conteo efectivo** (media de los top-M, o suma dividida
   por una funcion sublineal del conteo). Interpola entre `max` (que perdio,
   0.226 contra 0.306) y `sum` (que se derrumba con pool ancho).
2. **Reservar un cupo por mejor chunk.** Los 3 cupos se reparten: 2 por el
   ranking agregado y 1 por el mejor score de chunk individual. Ataca
   directamente a los 124 perdedores cuyo mejor chunk vale lo mismo que el del
   ganador, y no toca los otros dos cupos, asi que el riesgo esta acotado por
   construccion a un tercio de la respuesta.

La segunda es la que sigue el dato de cerca. Va pre-registrada como E20.
