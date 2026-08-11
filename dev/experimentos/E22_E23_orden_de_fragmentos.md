# E22 y E23 — el orden de los fragmentos

Estado: **ADOPTADOS** (9 de agosto de 2026). Los dos primeros cambios adoptados
tras siete resultados negativos seguidos el mismo dia.

## Resultado

| | NDCG@10 | NDCG penalizado | F1@3 | ilegibles | sin cobertura |
|---|---|---|---|---|---|
| antes | 0.490 | 0.476 | 0.440 | 19 | 175 |
| E22 solo | 0.497 | 0.482 | 0.440 | **0** | 162 |
| E23 solo (binaria) | 0.500 | 0.484 | 0.440 | 19 | **138** |
| **E22 + E23 (entregado)** | **0.506** | **0.491** | **0.440** | **0** | **122** |

NDCG@10 **+0.016 [+0.004, +0.029]**, 11 victorias contra 4, **IC enteramente
por encima de cero**. F1@3 sin moverse una milesima, como corresponde: ninguno
de los dos toca el ranking de documentos.

Entrega regenerada, sha256 **`987293ac…`**, reproduce byte a byte en frio desde
fuera del repo. 138 tests. Informe de vuelta en 8/8 paginas.

## E22 — el idioma por encima de la alineacion

`ordenar_para_fragmentos` ordenaba por (1) pertenecer al top-3, (2) idioma
legible, (3) aparato bibliografico. Con la alineacion primero, **un fragmento
en coreano del documento nº 1 le ganaba a uno legible del nº 4**. Para un
evaluador que lee espanol/ingles el primero vale **r=0 con certeza**; el
segundo tiene probabilidad no nula de valer mas. Cambiar un cero seguro por
una loteria no puede perder en esperanza, y el argumento sale de la definicion
de la metrica (sec. 10.2.1 juzga el campo `text`), no de nuestro proxy.

Datos de la puerta de entrada, corridos antes de medir:

- 19 de 500 fragmentos ilegibles, **todos `ko`**, en 5 consultas (q003, q006,
  q009, q011, q012).
- Salen de **dos documentos y solo dos**: `F3-SIPRI-002` (122 chunks, 0
  legibles) y `F3-SIPRI-100` (115 chunks, 0 legibles), traducciones al coreano.
- En las 5 consultas hay al menos un documento del top-3 **completamente
  legible** con ~100 chunks disponibles, asi que el reemplazo sale del propio
  top-3 y la alineacion casi no se rompe.

**Coste medido, que hay que decir junto a la ganancia:** la alineacion al top-3
baja de **499/500 a 480/500**, exactamente los 19 intercambiados y ni uno mas.
Cambian 5 lineas de `resultados.jsonl` y 19 fragmentos.

### La prediccion pre-registrada fallo, y se deja escrita

Se pre-registro que *"el proxy va a mostrar una perdida"*, razonando que
`eval_mini` le da 1 al fragmento coreano de un documento relevante y 0 al
legible de un documento fuera del top-3. **El dato de la puerta la contradijo
antes de medir** y la medicion lo confirmo: en **4 de las 5 consultas los
documentos coreanos son los irrelevantes**, y en q009/q012 el reemplazo venia
de un documento relevante. Gana 3-0 con IC por encima de cero.

Vale como recordatorio de que una prediccion mecanica bien razonada puede ser
falsa, y de por que la puerta de entrada se corre **antes** que la medicion.

## E23 — cobertura lexica como ultimo desempate

La sec. 10.2.1 exige que el pasaje **responda**, con escala graduada. Un pasaje
del documento correcto que no menciona el objeto de la consulta en ninguno de
sus dos idiomas entro por el TEMA del documento, no por responder.

- Cobertura cero con la consulta cruda: **257/500**. Con la consulta expandida
  por el glosario: **175/500** -> **138** tras el cambio.
- **No es inerte, y eso habia que probarlo primero**: E11 permuto criterios y
  salio byte a byte identico porque sus criterios nunca chocaban. Aca hay
  **162 pares invertidos por cobertura y 156 empatan en los tres criterios
  actuales**, en 24 consultas.
- Cambian 25 lineas y 159 fragmentos.

**La version GRADUADA se midio y se descarto**: misma cobertura final (138),
NDCG penalizado **−0.001** con 17 victorias y 12 derrotas, y **355 de 500
fragmentos movidos**. Mucha rotacion para efecto nulo.

**Salvedad que va junto al numero:** la evidencia de E23 es **enteramente de
las 50** y su IC cruza el cero (+0.009 [−0.001, +0.021]). Pasa por el criterio
de dano, no por demostrar ganancia.

## Lo que NO confirma, dicho con precision

En las **10 consultas independientes** el efecto de los dos es **+0.000
exacto**. Pero **no es la firma del sesgo de pooling**: es **ausencia de
casos** — ninguna de las 5 consultas con fragmentos ilegibles esta en esa
muestra. Es distinto de `doc_rrf` y de gte-primario, que ahi **perdian**. No
confirma y tampoco pierde nada.

## Implementacion

Cambia la clave de orden en `dev/src/retrieval/truncate.py` y en su copia
aplanada de `Entrega/generador.py`:

```python
return (ilegible, fuera_del_top, aparato, sin_cobertura)
```

`build_result_object` recibe un `texto_consulta` **opcional** con la consulta
ya expandida por el glosario. Es opcional a proposito: sin el, `toks` queda
vacio, el criterio de cobertura es constante y por tanto **inerte**, con lo que
los ~12 barridos historicos que llaman con `(qid, hits)` siguen midiendo lo que
median.

El test `test_ordenar_para_fragmentos_el_documento_manda_sobre_el_idioma` **se
invirtio en vez de borrarse**, con la medicion en el docstring: si alguien
vuelve a poner la alineacion primero, el test le dice por que esta mal y con
que numeros.

## Justificacion de fondo

Ambos se adoptan por la **leccion 7** —un defecto del dato se arregla porque es
un defecto—, igual que la reparacion de guiones y la prioridad de idioma. Que
el proxy este ademas de acuerdo en E22 es bienvenido, pero no es la razon.
