# E46 - Calibracion de encoders y agregacion sobre FAISS

Estado: **ADOPTADO** (12 de agosto de 2026). Rama `Thomas`.

## Problema medido

La configuracion anterior daba F1@3 0.455, NDCG@10 0.516 y NDCG penalizado
0.499. FAISS no era el cuello de botella: `IndexFlatIP` hace busqueda exacta y
el diagnostico E18 ya habia mostrado que el pool a profundidad 200 contenia
el 93.2% de los pares consulta-documento relevantes. La perdida principal
ocurria al reordenar chunks y reducirlos a tres documentos.

La cascada sumaba cosenos crudos con peso 0.60 para GTE y E5. Compartir el
intervalo teorico [-1,1] no significa compartir distribucion. Sobre los 200
candidatos de las 50 consultas, la dispersion mediana fue:

| encoder | rango | desviacion | media |
|---|---:|---:|---:|
| MiniLM | 0.120 | 0.024 | 0.682 |
| GTE multilingual base | 0.276 | 0.050 | 0.681 |
| multilingual E5 base | 0.103 | 0.018 | 0.807 |

Con pesos iguales, GTE tenia aproximadamente 2.7 veces el rango de E5 y
dominaba mas de lo declarado por el parametro.

## Metodo

`dev/scripts/barrido_thomas.py` guarda una sola pasada de 200 candidatos y
los tres cosenos por consulta. Primero hizo screening documental y despues
calculo NDCG completo solo para finalistas. El control reprodujo exactamente
0.455 / 0.516 / 0.499.

Se probaron suma cruda, min-max, normalizacion robusta y rango; pesos separados
para GTE/E5; `k_pool` de 100 a 200; agregacion top3 a top6; y contexto de
chunks vecinos. La vecindad no sobrevivio los vetos. La refinacion local mostro
una meseta, no un punto aislado, alrededor de min-max, GTE 0.45-0.50, E5
1.00-1.10, pool 200 y top6.

## Resultado adoptado

Configuracion: min-max por encoder y consulta, GTE 0.50, E5 1.00,
`rerank_depth=200`, `k_pool=200`, agregacion documental `top6`.

| lectura | anterior | E46 | delta |
|---|---:|---:|---:|
| F1@3, 50 | 0.455 | **0.499** | +0.045 |
| NDCG@10, 50 | 0.516 | **0.558** | +0.042 |
| NDCG penalizado, 50 | 0.499 | **0.539** | +0.040 |
| F1@3, 41 humanas | 0.486 | **0.518** | +0.032 |
| NDCG@10, 41 humanas | 0.537 | **0.573** | +0.036 |
| NDCG penalizado, 41 humanas | 0.520 | **0.554** | +0.034 |
| F1@3, 10 independientes | 0.433 | **0.433** | 0.000 |
| NDCG, 10 independientes | 0.474 | **0.477** | +0.003 |
| NDCG penalizado, 10 independientes | 0.467 | **0.470** | +0.003 |
| consultas con F1 cero | 11 | **8** | -3 |

Intervalos pareados al 90% contra la entrega anterior: F1
[-0.007,+0.096], NDCG [+0.007,+0.077] y NDCG penalizado [+0.008,+0.072].
Los tres excluyen la perdida tolerada de -0.02. La prueba de signos aislada
fue p=0.180 (10 victorias, 4 derrotas, 36 empates), por lo que no se presenta
la ganancia como prueba definitiva fuera de esta muestra.

## Chunking

Se intento reconstruir 384 tokens sin solape desde el `metadata.jsonl`
oficial de 128,526 filas. El checkpoint local `chunks_intermedios.jsonl` tenia
solo 68,145 filas y se rechazo por incompleto. En esta maquina CPU, la ruta
historica no termino en 20 minutos. Una variante de segmentacion global cambio
los limites (36 chunks frente a 60 en un documento real) y tambien fue lenta;
se descarto, sin construir indices ni atribuirle una mejora inexistente.

Esto es importante: el chunking de produccion se conserva porque 128 tokens
ya perdio con claridad en E21 y ninguna alternativa nueva supero la puerta de
screening. E46 mejora como se explotan los chunks: amplifica evidencia
coherente hasta seis chunks por documento y deja que FAISS aporte sus 200
candidatos completos.

## Reproduccion

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe Entrega\generador.py `
  --consultas dev\consultas_prueba\consultas_50_oficiales.jsonl `
  --out Entrega\resultados.jsonl
.\.venv\Scripts\python.exe dev\scripts\eval_mini.py `
  --resultados Entrega\resultados.jsonl
```

Verificacion local del 12 de agosto de 2026: `159 passed, 1 skipped`. Las
cifras de la tabla se reprodujeron desde `Entrega/resultados.jsonl`; son
metricas contra el ground truth local y no una calificacion oficial de ADL.

## Paridad de la GUI

La GUI llego a mostrar F1@3 0.361 sobre las 41 etiquetas humanas porque
activaba el grafo automaticamente al encontrar `grafo.graphml` y lo fusionaba
como una lista mas por RRF, que esta medido y pierde 11-0. Se corrigio el
arranque, se le pasa la consulta ya expandida al ordenamiento de fragmentos
—sin eso el criterio de cobertura lexica de E23 queda inerte y la GUI ordena
distinto que la entrega— y se muestra el promedio total de las 50 junto a los
desgloses por procedencia de la etiqueta.

## Integracion en `main` y verificacion independiente (12 ago 2026)

Este experimento se hizo en la rama `Thomas`, que salio de `66ced30`. El
historial de `main` fue reescrito despues, asi que su base comun con `main` es
el **primer commit del repo** y un `git merge` producia conflictos `add/add` en
todo el arbol. **Se integro por `cherry-pick` de los tres commits**, que
conserva su autoria y aplica exactamente el diff de 17 archivos.

Dos cosas se resolvieron a favor de `main` al integrar:

- **El grafo NO se apaga.** `main` lo habia convertido en **desempate no
  desplazante activo por defecto** (bonus de la sec. 7), que es distinto de la
  fusion RRF que este experimento apago con razon. Se conservo esa version y
  se volvio a medir bajo los scores calibrados: **0 de 50 lineas cambian** al
  pasar `--sin-grafo`. Sigue siendo integracion del artefacto, no una mejora.
- **La etiqueta se llama `asistido`, no `agente`** (renombrada en `main`).

**Verificado en la maquina local, sin dar por buena ninguna cifra ajena:**
189 tests, `validar_entrega.py --esperar-50` limpio, `eval_mini.py` devuelve
**0.499 / 0.558 / 0.539** con **8 consultas en cero**, y `resultados.jsonl`
**reproduce byte a byte en dos corridas en frio consecutivas** desde fuera del
repo, con `PYTHONPATH` vacio y sin mas flag que `--consultas`
(sha256 `fcd5f423...`).

## Por que esto no reabre dos ejes que estaban cerrados

Hay dos precedentes que obligan a justificarlo, y los dos se sostienen:

- **E09 ya midio min-max y lo rechazo** (0.476, falla 3 de 9 lecturas). Fallo
  porque **perdia en la muestra independiente** (ND −0.004) con la firma del
  sesgo de pooling. Esta variante —pesos separados por encoder, pool 200,
  top6— **no pierde en ninguna de las tres lecturas independientes**.
- **`k_pool=200` y `topM` estaban cerrados** (E33, E37), pero se cerraron
  **bajo score crudo**. Cambiar la escala de los sumandos es el cambio de
  regimen que E01 exige para volver a calibrar: es la misma regla por la que
  fue legitimo mover el peso de 0.25 a 0.60.

**Lo que hay que decir siempre junto al 0.499:** en las 10 independientes la
ganancia es **plana** (F1 +0.000, ND +0.003) y la prueba de signos aislada da
**p = 0.180**. Lo que sostiene la adopcion es el criterio vigente del proyecto
—IC al 90% del delta pareado excluyendo −0.02 en las tres metricas de las 50—
mas el veto de ceros, que **mejora de 11 a 8**. No es prueba definitiva fuera
de esta muestra, y el propio experimento lo declara.

## Efecto colateral: E42 muere con este cambio

`E42` (normalizacion por tamano de documento) estaba **medido como adoptable y
sin aplicar**, con `top5` y `k_pool=100`. Su justificacion mecanica era que el
documento ganador **satura los cinco sumandos** de `top5`. Con `top6` sobre un
pool del doble ese regimen desaparece, y re-medido encima de E46 **pierde en
las tres metricas de las 50 con los IC enteramente bajo cero, y sube los ceros
de 8 a 9**. Detalle en `E42_normalizacion_tamano.md`.
