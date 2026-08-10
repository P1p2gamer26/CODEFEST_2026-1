# E39 - cross-encoder bge-reranker-v2-m3 como re-puntuador (10 ago 2026)

**Estado: MEDIDO Y NO ADOPTADO.** El eje del cross-encoder queda cerrado por
medicion. Arnés: `dev/scripts/barrido_cross_encoder.py` (fase 1 en GPU,
fase 2 en CPU) y `dev/scripts/smoke_cross_encoder.py`.

## Por que se probo

ADL confirmo en la Q&A final que la restriccion de la sec. 8.3 aplica a las
**arquitecturas decoder**, no a los encoder-only: un cross-encoder como
re-puntuador esta permitido. El eje "otro encoder" se habia cerrado tres veces
(E04, E25, E31) pero siempre con **bi-encoders** usados como producto punto
sobre vectores almacenados (`reconstruct(fila)`). Un cross-encoder relee el
par (consulta, fragmento), que es otra arquitectura de uso, y la condicion de
E31 para reabrir ("arquitectura distinta Y pool anotado propio") se cumple a
medias: las 10 consultas independientes son el pool anotado propio.

**Justificacion mecanica:** el modo de fallo documentado del pool es
cross-lingual (NBQR/CBRN, "reabastecimiento en orbita"/on-orbit servicing);
bge-reranker-v2-m3 (568M, XLM-RoBERTa, Apache 2.0, 100+ idiomas) fue
entrenado en pares multilingues de relevancia (MIRACL/BEIR), o sea el
instrumento justo para ese fallo. Smoke test previo OK: separa relevante de
irrelevante en q001 (ingles y espanol), es sensible al orden de las palabras
(no es la trampa de gte) y la direccion (consulta, documento) es la correcta.

## Costo medido (smoke test)

| | ms/pair | 100 pares/consulta | 50 consultas |
|---|---|---|---|
| CPU (evaluador) | ~850 | ~85 s | ~71 min |
| GPU (GTX 1650) | ~125 | ~12 s | ~10 min |

El costo online del evaluador es el riesgo principal declarado antes de medir.

## Celdas (pre-registradas, peso 0.60 salvo la de sensibilidad)

Base de comparacion: la entregada (MiniLM -> gte+e5 @0.60, pool 100, top5,
glosario). El arnes reconstruye la base desde `pools_entregados.json`
(similitudes alineadas por posicion con los 200 candidatos; el pool es el
top-100 por base, verificado con error 0.0) y la celda base **reproduce
`resultados.jsonl` byte a byte (50/50)**: el arnes mide lo mismo que se
entrega. 10.000 pares puntuados en GPU (~24 min).

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | F1(hum) | ND(hum) |
|---|---|---|---|---|---|---|---|
| **BASE entregada (gte+e5@0.60)** | **0.455** | **0.516** | **0.499** | **0.433** | **0.474** | **0.486** | **0.537** |
| S100 add cross@0.60 (pool) | 0.448 | 0.507 | 0.490 | 0.367 | 0.436 | 0.470 | 0.529 |
| S200 add cross@0.60 | 0.451 | 0.480 | 0.463 | 0.333 | 0.409 | 0.458 | 0.493 |
| S200 add cross@0.25 | 0.443 | 0.491 | 0.477 | 0.333 | 0.409 | 0.472 | 0.519 |
| S200 add cross@0.60 minmax | 0.451 | 0.482 | 0.468 | 0.333 | 0.409 | 0.458 | 0.495 |
| S100 replace cross@0.60 (pool) | **0.465** | **0.521** | **0.504** | 0.367 | 0.430 | 0.482 | 0.533 |
| S200 replace cross@0.60 | 0.456 | 0.479 | 0.462 | 0.333 | 0.396 | 0.472 | 0.491 |

## El resultado que hay que recordar

**S100-replace es la celda con mejor F1@3 de TODO el proyecto sobre las 50**
(0.465, por encima del 0.455 entregado) y aun asi **no se adopta**, porque:

- **En las 10 independientes (la unica muestra sin sesgo de pooling) pierde
  todo**: F1 -0.067 [-0.133, 0.000] y NDCG -0.043 [-0.103, +0.011]. Las
  derrotas son q017 y q033, dos de las independientes.
- **Sus 6 victorias incluyen q011 y q012 (etiqueta panel-agentes).** En las 41
  humanas el reparto es **4-3 y el F1 queda plano** (-0.004). Es la firma
  exacta de E04, E25, E31, gte-primario, `doc_rrf` y 200:top5: la ganancia
  vive en las consultas con sesgo de pooling y se evapora en las humanas.
- **El veto corre al reves**: S100-replace baja los ceros de 11 a 10 (rescue
  uno), pero las celdas de SUMA (cross agregado a gte+e5) suben los ceros de
  11 a 12. Sumar la opinion de un tercero a los dos re-puntuadores tira
  documentos que los tres juntos ya no se ponen de acuerdo en subir.

## Lectura honesta

- **La celda que luce como mejora del dia es peor sistema.** Igual que E37
  (0.471 de k150), la ganancia sobre las 50 es pooling, no senal.
- El control de escala (minmax) no cambia el veredicto: el problema no es la
  escala del termino, es que la opinion del cross-encoder sobre los candidatos
  que la cascada ya trajo no es mejor que la de la cascada.
- **El eje queda cerrado por medicion, con arquitectura nueva y con el pool
  anotado propio.** No es un cuarto bi-encoder: es un mecanismo distinto y aun
  asi no confirma. La leccion: lo que la cascada de MiniLM+gte+e5 deja mal
  ordenado no lo arregla otro opinador sobre el mismo pool.
- **Queda el dato de costo para quien lo vuelva a mirar:** 85 s/consulta en
  CPU del evaluador habria sido el trade-off aun con ganancia. No hay
  ganancia, el costo es inmoral.

**CERRADO. No reabrir sin un pool anotado que no venga de la cascada.**
