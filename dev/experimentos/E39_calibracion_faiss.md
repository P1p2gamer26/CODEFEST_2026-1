# E39 - Calibracion de encoders y agregacion sobre FAISS

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

| lectura | anterior | E39 | delta |
|---|---:|---:|---:|
| F1@3, 50 | 0.455 | **0.499** | +0.045 |
| NDCG@10, 50 | 0.516 | **0.558** | +0.042 |
| NDCG penalizado, 50 | 0.499 | **0.539** | +0.040 |
| F1@3, 41 humanas | 0.486 | **0.518** | +0.032 |
| F1@3, 10 independientes | 0.433 | **0.433** | 0.000 |
| NDCG, 10 independientes | 0.474 | **0.477** | +0.003 |
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
screening. E39 mejora como se explotan los chunks: amplifica evidencia
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
