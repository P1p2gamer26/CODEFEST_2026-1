# E24 — el nombre del documento acierta donde se predijo, y pierde en todo lo demas (9 ago 2026)

`dev/scripts/barrido_nombre_doc_e24.py` + `dev/scripts/nombres_doc_e24.py`,
configuracion identica a la entrega (prof 200, k_pool 100, top5, peso 0.60).
La fila base reproduce **0.440 / 0.490 / 0.476** digito a digito (regla de E09).
Coste: codificar 1.047 nombres con MiniLM, segundos de CPU.

## Puerta de entrada, antes de puntuar nada

De los 1.818 documentos con `fuente` (1.691 nombres distintos), tras normalizar
—quitar extension, prefijo de coleccion, fechas e identificadores— solo
**1.047 (57.6%) tienen 3 o mas tokens de contenido**. Los 771 restantes son
opacos y reciben score neutro.

| coleccion | docs | informativos |
|---|---|---|
| ALERTAS | 425 | **0** |
| AMAZONUW (tiles) | 75 | **0** |
| CSET | 127 | 127 |
| ATLCOUNCIL | 186 | 176 |
| CSIS | 214 | 165 |
| SWF | 130 | 116 |

**El 23% del corpus es ALERTAS y no tiene un solo nombre puntuable**
(`ALERTAS_informes01.pdf`). El criterio ve poco mas de la mitad del corpus.

Sobre el ranking real, por consulta: **4,4 documentos saturan** el tope de
top5, de ellos **2,8 con nombre informativo (63%)**, y hay **9,9 pares
permutables**. Se reconfirma el numero de E19: **el 92,7% del top-3 entregado
sale del conjunto saturado**. O sea que el criterio actua sobre el sitio
correcto y sobre bastantes documentos — **no sale inerte**, y eso hace legible
el resultado.

Correccion declarada: el umbral de longitud de token se bajo de 3 a 2
caracteres **antes de puntuar nada**, porque `ai` es el termino de dominio del
fenomeno 1 y media coleccion se llama `ai-index`. Subio los informativos de
1.006 a 1.047. No se toco despues de ver ningun resultado.

## Resultado

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| **entregada** | **0.440** | **0.490** | **0.476** | **0.400** | **0.436** | **0.468** | **0.510** |
| **nombre-satura** (sin parametro) | 0.425 | 0.475 | 0.460 | 0.400 | 0.384 | 0.458 | 0.490 |
| nombre-w010 (control) | 0.447 | 0.498 | 0.483 | 0.433 | 0.474 | 0.476 | 0.519 |
| nombre-w030 (control) | 0.433 | 0.478 | 0.466 | 0.433 | 0.474 | 0.460 | 0.495 |

Deltas pareados con IC al 90%, formulacion sin parametro:

    F1(50)   -0.015 [-0.051, +0.021]   3 gana / 6 pierde   no pasa
    ND(50)   -0.016 [-0.062, +0.028]  12 gana / 6 pierde   no pasa
    F1(ind)  +0.000 [-0.067, +0.067]   1 gana / 1 pierde   no pasa
    ND(ind)  -0.052 [-0.195, +0.038]   3 gana / 1 pierde   no pasa

**Ninguna de las nueve lecturas pasa el criterio** (IC al 90% que excluya una
perdida de 0.02). Cambia 31 de las 50 lineas: es un cambio grande y su signo
es negativo.

## La justificacion mecanica SI se cumple, y aun asi el cambio pierde

Esto es lo que hay que recordar de E24. La prediccion era que el efecto
apareciera en las 11 consultas con F1@3 = 0, y **aparece**:

    11 CEROS  F1  0.000 -> 0.082   2 rescatadas (q003 0.000->0.500, q037 0.000->0.400)

El nombre del documento **rescata dos consultas que ninguna palanca del
proyecto habia movido**. E19/E20/E22 decian que esas once tienen el relevante
en el pool y lo pierden en la agregacion porque los scores de chunk no
discriminan; una senal externa las mueve, tal como se escribio.

El problema es el precio. Reparto completo del F1@3:

    gana   q003 (0.00->0.50)  q017 (0.33->0.67)  q037 (0.00->0.40)
    pierde q009 (0.33->0.00)  q020 (0.67->0.33)  q029 (0.33->0.00)
           q032 (0.67->0.33)  q041 (1.00->0.67)  q048 (0.67->0.33)

**Gana donde la respuesta valia cero y pierde donde ya valia mucho.** q041
pasa de 1.000 a 0.667; q020 y q032 de 0.667 a 0.333. El nombre discrimina
cuando el contenido ya fallo, y estorba cuando el contenido ya acerto — que es
el modo de fallo de cualquier senal debil aplicada uniformemente.

Esto **no refuta** la lectura de E19/E22 sobre las once consultas: la refuta
como palanca global. Un criterio que se disparara solo cuando el ranking por
contenido es plano seguiria vivo, pero eso es otra hipotesis y necesita su
propio experimento y su propio umbral pre-registrado.

## Los controles de peso continuo: no adoptables, y por que

`nombre-w010` gana en **las nueve lecturas** y no pierde ni una consulta
(1g/0p en todas). Parece la mejor celda de la tabla. **No se adopta, y estaba
escrito antes de medir:**

1. **Toda la ganancia es UNA consulta.** q017 pasa de 0.333 a 0.667 y las
   otras 49 no se mueven: 10 de 50 lineas distintas, un solo cambio de F1.
   Es exactamente el caso `top5` de la leccion 3 —ruido que cayo del lado
   bueno— y el IC de `[+0.000, +0.020]` toca el cero por abajo en las nueve.
2. **Es el argmax de una grilla de dos puntos.** `w=0.30` pierde en las 50
   (-0.007, 1g/2p) y la formulacion sin parametro tambien. Solo el punto
   intermedio gana: eso es la maquina de sobreajuste de la leccion 2.
3. **La ganancia NO viene de donde la justificacion mecanica la predijo.**
   Las 11 consultas con F1 = 0 se quedan en **0.000, cero rescatadas**. El
   unico cambio, q017, ya valia 0.333. El peso continuo mueve una consulta
   sana; el mecanismo que se pre-registro no esta actuando ahi.

El punto 3 es el decisivo. Adoptar w=0.10 seria cobrar como mejora un cambio
cuya explicacion escrita se acaba de medir y no se cumple.

## Veredicto

**No adoptable.** `Entrega/` no se toca. La formulacion sin parametro pierde en
las nueve lecturas; los controles con peso no son adoptables por diseno y su
ganancia es una consulta sin respaldo mecanico. El veto por consultas de anotacion asistida
no llego a activarse (todas las diferencias del control estan en consultas
humanas).

**Lo que queda vivo y hay que anotar:** el nombre del documento es la primera
senal del proyecto que mueve dos de las 11 consultas con F1@3 = 0. Todas las
palancas anteriores —E07, E12, E15, E19, E20— dejaron esas once en cero. Si
alguna vez se abre un criterio **condicional** (disparar solo cuando la
dispersion del ranking agregado de documentos es plana, o solo cuando ningun
documento satura con margen), el nombre es el material con el que construirlo.
Lo que E24 cierra es el uso **incondicional**, en sus tres regimenes:
permutacion pura, peso chico y peso grande.
