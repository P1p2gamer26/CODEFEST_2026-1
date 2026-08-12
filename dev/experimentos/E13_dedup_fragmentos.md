# E13 - deduplicar fragmentos casi identicos entre los 10 entregados (12 ago 2026)

## Hipotesis (copiada de cola.jsonl, previa a la medicion)

Entre los 10 fragmentos entregados hay pares de texto casi identico, y
reemplazarlos por el siguiente candidato del pool mejora el NDCG@10 real;
ademas es un defecto del dato, no una mejora de metrica.

## Justificacion mecanica (copiada de cola.jsonl, previa a la medicion)

El chunker solapa oraciones entre chunks consecutivos POR DISENO
(`CHUNK_OVERLAP_SENTENCES`), y ya esta medido que el 32% de los pares
consecutivos comparte sus ultimas 25 palabras. La alineacion de fragmentos
(E22/E23) empeora esto de forma mecanica: concentrar el 98% de los
fragmentos en 3 documentos hace mas probable que dos chunks consecutivos del
mismo documento caigan los dos en el top-10. Un cupo de 10 gastado en
repetir texto ya entregado no puede aportar nada a un evaluador humano por
la sec. 9.3.2, aunque el proxy binario le de 1 igual que al primero (hereda
la relevancia del documento y no mira el texto).

**Riesgo principal, declarado antes de medir:** el proxy NO PUEDE MEDIR esto
si el reemplazo viene del mismo documento (delta 0.000 por construccion); la
cifra de decision real es el CONTEO de pares casi-identicos, no el NDCG.

## Arnes: reusado, no reescrito

El arnes (`dev/scripts/barrido_dedup_e13.py`, commit `07ce1ef` de otra
sesion) y su test (`dev/tests/test_dedup_fragmentos.py`) ya existian. Esta
tarea audito el arnes, corrio la puerta de fidelidad y la grilla; no se
modifico ninguna linea de codigo.

**Auditoria del arnes (Step 1):**
- `solapamiento()`: Jaccard de shingles de 5 palabras (`SHINGLE = 5`, fijo,
  no se toca), normalizado NFKD sin acentos. Correcto: identico da 1.0,
  disjunto da 0.0, con textos cortos (<5 palabras) cae a comparar el
  fragmento entero como un solo shingle.
- `dedup()`: por cada fragmento que solapa >= umbral con alguno YA ACEPTADO
  en la salida, busca en la reserva (candidatos 11..50 del pool ya ordenado
  por `ordenar_para_fragmentos`, ver mas abajo) el primero que no solape con
  nada de lo ya aceptado; si no hay reemplazo limpio, conserva el original
  (la sec. 9.2 exige 10 fragmentos siempre). Los 6 tests de
  `test_dedup_fragmentos.py` cubren identico/disjunto/parcial y los tres
  casos de reemplazo, y los 6 pasan.
- `resultado()`: replica el camino entregado (`filtrar_por_fenomeno_dominante`
  con umbral 0.8 de E32, `aggregate_documents` con `top5`,
  `ordenar_para_fragmentos` + `enforce_word_limit`), y **antes** de deduplicar
  arma la reserva con `enforce_word_limit(resto, max_fragments=RESERVA_MAX)`
  sobre los candidatos que NO entraron a los 10 -- es decir, la reserva
  respeta el mismo orden (alineacion top-3 + idioma + cobertura lexica) que
  los 10 originales, no es un pool crudo sin ordenar.
- Consume `expandir_consulta()` del glosario antes de tokenizar, que el
  camino entregado tambien aplica -- de no hacerlo la puerta de fidelidad
  fallaria en las consultas con termino de glosario.

Sin bugs encontrados en la auditoria (a diferencia del `break` de E42).

## Puerta de fidelidad (Step 3, patron de `barrido_norm_doc_e42.py`)

```
docs identicos: 50 de 50
frags identicos: 50 de 50
```

La celda base (`umbral=None`, sin deduplicar) reproduce `Entrega/resultados.jsonl`
digito a digito en documentos **y** en fragmentos -- mas estricto que la
puerta de E42, que solo comparaba `documents`. F1(50)/ND(50)/NDp(50) de la
celda base: 0.4547 / 0.5162 / 0.4992 (E32 vigente).

## Efecto estructural (Step 4): cuenta los pares antes de mirar metricas

Sobre los 500 fragmentos entregados (10 por consulta, 45 pares posibles por
consulta, 2.250 pares en total):

| umbral | pares con solapamiento >= umbral |
|---|---|
| >= 0.5 | **17** |
| >= 0.7 | **9** |
| >= 0.9 | **0** |

**Par de mayor solapamiento: 0.8876**, q019, `F2-SWF-123-c0215` vs.
`F2-SWF-124-c0261` -- por debajo del umbral mas alto de la grilla (0.9), que
por eso resulta un no-op exacto (0 fragmentos movidos).

**Reparto mismo-documento vs. documento-distinto (riesgo 3 pre-registrado):**
de los 17 pares >= 0.5, **4 son del mismo `doc_id`** (`F2-CSIS-035` x2,
`F2-CSIS-037`, `F3-MAPPOEA-018`) y **13 son de documentos distintos** --
casi todos entre ediciones consecutivas del atlas anual de SWF
(`F2-SWF-121/122/123/124`, cuatro anos del mismo informe con parrafos de
metodologia identicos) y entre dos informes MAPP/OEA (`F3-MAPPOEA-030/031`).
**El caso dominante NO es el solape de chunks consecutivos del mismo
documento que motivo la hipotesis**: es contenido boilerplate repetido
entre documentos hermanos, la misma familia de duplicados de 27 grupos con
texto inicial identico que ya se documento al descartar la deduplicacion de
documentos (q020/q032). Confirma el riesgo 3: en 13 de 17 pares el
reemplazo mide "deduplicacion entre documentos distintos", que es un eje ya
cerrado por ambiguedad del ground truth.

## Grilla (Step 5, celdas pre-registradas: base, u0.5, u0.7, u0.9)

| celda | F1(50) | ND(50) | NDp(50) | ceros | frags movidos |
|---|---|---|---|---|---|
| **base** | **0.4547** | **0.5162** | **0.4992** | **11** | **0** |
| u0.5 | 0.4547 | 0.5107 | 0.4937 | 11 | 18 |
| u0.7 | 0.4547 | 0.5167 | 0.4997 | 11 | 11 |
| u0.9 | 0.4547 | 0.5162 | 0.4992 | 11 | 0 |

F1@3 sale +0.000 exacto en las tres celdas, tal como predecia el arnes
(`aggregate_documents` no se toca).

**IC al 90% del delta pareado contra la base:**

| celda | ND(50) | ND(ind) |
|---|---|---|
| u0.5 | -0.0054 [-0.0118, +0.0009] | -0.0085 [-0.0255, +0.0000] |
| u0.7 | +0.0006 [-0.0048, +0.0064] | -0.0085 [-0.0255, +0.0000] |
| u0.9 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

**Fragmentos fuera del top-3 declarado (riesgo 2 pre-registrado, deshacer
E22):** base 20/500, u0.5 21/500, u0.7 21/500, u0.9 20/500. El reemplazo casi
no saca fragmentos del top-3 (+1 de 500 en las dos celdas que mueven algo).

**Consultas que cambian de NDCG@10, por procedencia (u0.7, la celda que mas
se acerca a pasar):**

- q019 (humana): 0.4100 -> 0.4833 (+0.073) -- el reemplazo SI ataca el par de
  mayor solapamiento (0.888, F2-SWF-123/124), que es exactamente el caso para
  el que se escribio la hipotesis.
- q022 (humana): 0.4323 -> 0.5424 (+0.110)
- q026 (**independiente**): 0.9364 -> 0.8512 (**-0.085**)
- q031 (humana): 0.2874 -> 0.2180 (-0.069)

En u0.5 se suman ademas q030 (humana, -0.073) y q036 (humana, -0.064) y
q048 (anotacion-asistida, -0.066), sin ganancias nuevas.

## Veredicto: E13 CERRADO, NEGATIVO

Aplicando la regla en orden (veto de ceros, luego IC al 90% de NDCG@10
excluyendo -0.02 en las **dos** muestras, luego -- si sobrevive mas de una --
la que menos fragmentos mueve):

1. **Veto:** ninguna celda sube los 11 ceros. Ninguna cae aqui.
2. **IC en las dos muestras:** u0.5 y u0.7 fallan en las 10 independientes --
   el limite inferior (-0.0255) supera la perdida tolerada de -0.02, y la
   causa es una sola consulta (q026) que pierde -0.085 sin compensacion en
   esa muestra de 10. **u0.9 pasa el IC en las dos muestras porque el delta
   es exactamente 0.0000: es un no-op**, no una mejora -- el par de mayor
   solapamiento del corpus (0.888) queda por debajo de ese umbral.
3. No hay ninguna celda que simultaneamente mueva fragmentos Y pase el
   criterio. **No hay argmax que aplicar.**

**Razon estructural, no solo estadistica:** solo 17 pares de 2.250 posibles
superan 0.5 de solapamiento (0.76%), y de esos 17 la mayoria (13) no es el
mecanismo que motivo la hipotesis (chunks consecutivos del mismo documento)
sino boilerplate compartido entre documentos hermanos -- un eje ya cerrado
por otra razon (q020/q032). El problema que E13 se propuso arreglar es real
pero **raro**: no hay masa suficiente de pares casi-identicos en los 500
fragmentos entregados para que ninguna variante de umbral pase el criterio
de las dos muestras sin, de paso, tropezar con una perdida real en q026.

**No se adopta nada.** El eje "deduplicar fragmentos casi identicos" queda
cerrado por escasez del defecto que se buscaba corregir, siguiendo la misma
disciplina de E27 (recorte intermedio inerte a N=150): un negativo bien
medido, con la puerta de fidelidad exacta (50/50 en documentos Y fragmentos)
y el conteo estructural por delante de cualquier metrica.
