# E30 — el grafo no sabe distinguir hermanos (9 ago 2026)

`dev/scripts/barrido_entidades_e30.py` (+ `mapa_entidades_grafo.py`, que vuelca
`entidad -> doc_id` desde `grafo.graphml` por `iterparse`, 215.971 entidades y
1.687 documentos en 14,4 MB de cache, sin `networkx.read_graphml`).

**Veredicto: NO adoptable. Pierde por poco y, sobre todo, la justificacion
mecanica falla — las 7 consultas de "hermano equivocado" no se mueven ni una.**

## La puerta de entrada ya lo anticipaba

| | |
|---|---|
| consultas con >= 1 entidad reconocida por spaCy | **37 / 50** |
| consultas con >= 1 entidad que TAMBIEN esta en el grafo | 34 / 50 |
| entidades por consulta (mediana) | **1** |
| documentos saturados (>= 5 chunks en el pool), total | 222 |
| ...con alguna entidad de su consulta | **37 (17%)** |
| consultas cuyo top-3 cambia | **3 / 50** |

Con una entidad mediana por consulta y el 83% de los saturados sin ninguna
coincidencia, el criterio es **casi siempre un empate**: no ordena, solo desempata
en tres consultas. El resultado estaba acotado a ser inerte o ruido antes de medir.

## Resultado

Fila base verificada digito a digito (regla de E09): 0.440 / 0.490 / 0.476 en las
50 y 0.400 / 0.436 / 0.429 en las 10 independientes.

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| **entregada** | **0.440** | **0.490** | **0.476** | **0.400** | **0.436** | **0.468** | **0.510** |
| e30 graduada (nº de entidades) | 0.433 | 0.485 | 0.470 | 0.400 | 0.436 | 0.460 | 0.503 |
| e30 binaria (hay / no hay) | 0.433 | 0.486 | 0.471 | 0.400 | 0.436 | 0.460 | 0.505 |

Deltas pareados, IC al 90%:

    graduada  F1(50)  -0.007 [-0.020, +0.000]   0 gana / 1 pierde   NO pasa
              F1(hum) -0.008 [-0.024, +0.000]   0 gana / 1 pierde   NO pasa
              F1(ind) +0.000 [+0.000, +0.000]   0 gana / 0 pierde   (inerte)
    binaria   identico salvo decimales

**Cero victorias en las nueve lecturas.** En las 10 independientes el cambio es
literalmente inerte (ninguna de las 3 consultas que toca esta ahi). El veto de
consultas con F1@3 = 0 no se activa: siguen siendo 11.

Alcance: **3 de 50 lineas** y 7 de 150 documentos cambian (5 en la binaria) —
todas permutaciones dentro del top-3, ningun documento nuevo entra.

## Lo que hay que decir: la justificacion mecanica fallo

Las 7 de hermano equivocado que lista `diagnostico_colecciones.py` son
`q005 q007 q034 q037 q044 q046 q047`. **Delta de F1@3 en las siete: +0.00.**
El criterio no toco ninguna. Es exactamente el patron por el que se rechazo el
control de E24: el promedio se movio (hacia abajo) por consultas que no eran
las que la hipotesis decia atacar.

Las tres que si cambian:

| | relevantes | antes | ahora | |
|---|---|---|---|---|
| q007 | SIPRI/DAIO | UNOOSA-030, ILIA-001, SIPRI-100 | permuta 1-2 | F1 igual (0) |
| q008 | CSET/DAIO | SIPRI-002, DAIO-035, DAIO-031 | permuta 1-2 | F1 igual |
| **q041** | MAPPOEA 020/030/031/034 | 030, 031, **020** | 031, 030, **032** | **pierde** |

q041 es el caso testigo y el mas ironico: es una consulta de hermanos MAPPOEA,
justo el escenario para el que se diseno el desempate, y **el grafo eligio al
hermano equivocado** — saco un relevante (`F3-MAPPOEA-020`) para meter
`F3-MAPPOEA-032`. Toda la perdida del experimento es esa consulta.

## Por que el grafo no sirve para esto

Los nodos son entidades nombradas de spaCy sobre el corpus entero. Los hermanos
de una serie —informes trimestrales del MAPP/OEA, traducciones de SIPRI— hablan
de **las mismas entidades**: Colombia, Clan del Golfo, ONU. La entidad discrimina
la COLECCION, que es justo lo que el recuperador ya acierta el 92% de las veces
(solo 4 de 50 fallan la coleccion). **La senal del grafo esta correlacionada con
la senal que ya funciona y es ciega a la que falta**, que es de que periodo, que
region o que operacion concreta habla cada hermano.

Segunda causa, medible: la consulta aporta **1 entidad mediana**. Una consulta
de 20 palabras en espanol sobre un tema general no nombra casi nada propio.

## Lo que queda cerrado

**El grafo no aporta al ranking de documentos, por ninguna de las dos vias
probadas**: fusionado en la recuperacion perdio 11-0 (4 ago), y como desempate
entre saturados es inerte-negativo (hoy). Se sigue entregando como bonus de la
sec. 8.5, que es para lo que existe. **No reabrir sin una senal distinta de las
entidades nombradas** — la que haria falta es temporal o geografica *fina*, y el
NER que construyo el grafo excluye explicitamente DATE, TIME y CARDINAL.

Tambien queda cerrada, por ahora, la lectura optimista del diagnostico del 9 ago:
que el 14% de "hermano equivocado" fuera el bucket grande y direccionable no
implica que haya una palanca para el. Van dos que lo intentan (E24 por el nombre
del documento, E30 por las entidades) y las dos fallan **sin mover esas
consultas**.

## Instrumento que queda

`mapa_entidades_grafo.py` deja el cache `dev/intermedios/entidades_por_doc.json`
(14,4 MB, entidad -> doc_id) y hace el grafo consultable con RAM cero. Sirve para
cualquier futura hipotesis que necesite las entidades, sin volver a parsear
183 MB de XML.
