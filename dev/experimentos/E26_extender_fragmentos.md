# E26 — extender los fragmentos con el chunk contiguo

Estado: **la version pre-registrada NO se adopta; la variante `sin-aparato` es
adoptable, con la salvedad de que es POST-HOC.** No aplicado a `Entrega/`.

Arnes: `dev/scripts/barrido_extender_e26.py` (lee el pool volcado y
`metadata.jsonl` por streaming; no carga ningun indice FAISS). Test:
`dev/tests/test_extender_e26.py`, 5 casos.

## Riesgo declarado antes de medir, y se cumplio

Nuestro NDCG@10 hereda la relevancia del `doc_id`. Extender un fragmento no
cambia de que documento viene, asi que **el delta del proxy es 0.000 por
construccion**. Salio **+0.000 exacto**, en las 50 y en las 10. Eso no es un
fallo del experimento: es la respuesta esperada. Lo unico que puede moverse es
el **penalizado**, que si mira el texto.

## Resultado

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | NDp(ind) |
|---|---|---|---|---|---|---|
| entregada | 0.440 | 0.506 | 0.491 | 0.400 | 0.436 | 0.429 |
| e26-extendida | 0.440 | 0.506 | **0.488** | 0.400 | 0.436 | 0.429 |
| **e26-sin-aparato** | 0.440 | 0.506 | **0.496** | 0.400 | 0.436 | **0.433** |

Fila base reproducida digito a digito (0.440 / 0.506 / 0.491), regla de E09.

Deltas pareados, IC al 90%:

- **e26-extendida**: NDp(50) **−0.003 [−0.006, +0.000]**, 8 victorias y **16
  derrotas**. NDp(ind) +0.000.
- **e26-sin-aparato**: NDp(50) **+0.006 [+0.002, +0.009]**, **17 victorias y 3
  derrotas**, IC enteramente sobre cero. NDp(ind) **+0.004 [+0.000, +0.010]**,
  2-0.

F1@3 y NDCG binario **+0.000 exacto en las dos muestras y en las dos celdas**:
la extension se aplica DESPUES de elegir los 10 fragmentos, asi que no puede
mover ni los documentos ni el orden. Solo anade texto.

## Los cinco puntos que habia que reportar

**1. Distribucion de palabras.** p50 **170 → 245**, p90 **195 → 250**, maximo
**215 → 250**. Total entregado **83.006 → 120.802** palabras (+45%). Se
amplian **497 de 500** fragmentos (los 3 restantes ya llenaban el presupuesto).

**2. Cero duplicacion — y costo un arreglo real.** 0 fragmentos con una oracion
de contenido (>= 8 palabras) repetida, igual que la entrega. **El dedup por
igualdad de oracion NO alcanzaba** y se detecto midiendo: el segmentador no es
idempotente cruzando el borde del chunk. En `F2-UNOOSA-030` la oracion solapada
sale como `'23.'` + `'Las directrices reflejan...'` dentro del chunk y como
`'23. Las directrices reflejan...'` entera en el vecino, asi que la igualdad
exacta no la ve y el parrafo **salia dos veces** — exactamente el fallo que
mato a la version de vecino entero. Se cambio a **contencion de subcadena**
sobre el texto acumulado, que lo cubre en los dos sentidos porque
`enforce_word_limit` une las oraciones con un espacio simple. El test
`test_no_duplica_la_oracion_solapada_aunque_el_splitter_la_parta_distinto`
congela el caso real.

Quedan 4 repeticiones nuevas de pseudo-oraciones degeneradas (`'2025.'`,
`'1.'`, `'3.'`) que el segmentador emite en texto de listas y numeracion. Son
de la misma clase que las **6 que la entrega ya tiene** y no son contenido.

**3. Ningun fragmento supera 250 palabras.** Maximo exacto 250, verificado
sobre los 500 en las dos celdas.

**4. Cobertura lexica: baja poco, y hay que decirlo.** 122 → **115** sin
cobertura (sin-aparato: 116). Solo **7 de 122** se rescatan. La hipotesis
implicita de que el vecino traeria el termino de la consulta **no se sostiene**:
si un chunk entro por el tema del documento y no por responder, su vecindario
suele estar igual de lejos del objeto de la consulta. Fragmentos ilegibles: 0
antes y despues.

**5. Lectura manual.** `dev/intermedios/extender_e26/muestra.md` (version
pre-registrada) y `muestra_sin_aparato.md` (variante). El veredicto humano es
**mixto y es lo que decide el experimento**:

- **q026 / F2-SWF-124-c0957 — ganancia clara.** El fragmento original era
  bibliografia pura (notas 150 y 151 con URLs) mas una oracion suelta. Lo
  anadido son 160 palabras de prosa sustantiva sobre la estructura del programa
  contraespacial chino, el Programa 640, el 863 y el KKV del HQ-19. Un r=0
  pasa a ser plausiblemente r=2.
- **q013 / F1-CSET-100-c0199 — ruido puro.** El original era una lista de
  referencias y lo anadido son **26 "oraciones" mas de la misma lista**
  (`Cyber Centre. (2022)...`, `DARPA. (2023)...`). Extender no lo mejora: lo
  alarga. **Este es el caso que la variante `sin-aparato` elimina.**
- **q014 / F1-DAIO-019-c0063 — mixto.** Entra una tabla de programas en
  vinetas (ruido tipografico) y detras prosa util con las cifras de
  presupuesto de innovacion de defensa.

## Por que la variante `sin-aparato`, y por que es post-hoc

La lectura manual mostro que el material anadido a veces es aparato
bibliografico, y el penalizado lo confirmo mecanicamente: la fraccion media de
aparato de los 500 fragmentos sube de **0.0192 a 0.0259**, con 16 derrotas
contra 8 victorias. Ningun fragmento cruza el umbral de 0.60 en ningun sentido
— el efecto es continuo, no de puerta.

La variante descarta al anadir las oraciones que `fraccion_aparato` marca como
aparato. **Es coherente con el sistema, no un parametro nuevo**: el gate de
bibliografia ya ordena los fragmentos por ese mismo criterio, e inyectarles
aparato al extenderlos era contradecirlo. Baja la fraccion media a **0.0147**,
por debajo de la entrega, y da vuelta el signo: **17-3 con IC sobre cero**.

**Salvedad de metodo, obligatoria: la variante NO estaba pre-registrada.** Se
midio despues de ver el resultado de la version pre-registrada. Que su
justificacion mecanica sea buena no la vuelve pre-registrada, y el pre-registro
existe justamente para que esa distincion no se pierda.

## Veredicto

**Si se adopta, se adopta por la leccion 7** —un defecto del dato se arregla
porque es un defecto—, igual que los guiones, el idioma y el gate de
bibliografia. **Nunca vendiendolo como mejora de metrica**: el NDCG@10 es
+0.000 exacto por construccion y lo unico que se mueve es una cota inferior del
proxy. Lo que se entrega de verdad es **45% mas de texto** en el campo que la
sec. 10.2.1 juzga, dentro del limite que la sec. 9.2.1 autoriza.

**Lo que argumenta en contra, y no es poco:** la cobertura lexica casi no
mejora (7 de 122), un tercio del material leido a mano es ruido tipografico, y
diluir un pasaje que si responde con 90 palabras de tabla puede enterrar la
respuesta para un evaluador que lee de arriba abajo. Ese ultimo riesgo **no lo
mide ningun instrumento que tengamos**.
