# Anotación asistida: medida y descartada como sustituto del criterio humano (2 ago 2026)

Segundo y **último** intento de cribar el ground truth por una vía asistida en
lugar de a mano. El primero usó una sola ronda y dio F1 0.28 contra las
etiquetas humanas; quedó la salvedad de que el consenso de varias rondas
podría ser mejor. **No lo es.** Este experimento la cierra.

## Diseño

Cuatro rondas independientes con criterios deliberadamente opuestos
—**estricto**, **inclusivo**, **dominio militar**, **literal/terminológico**—
sobre el mismo dossier de 15 consultas (`build_dossier.py`), a ciegas: **9 sin
anotar mezcladas con 6 ya anotadas a mano**, barajadas, sin señalar cuáles eran
cuáles. Las 6 de calibración son las *independientes* (sus candidatos no los
propuso el recuperador), así que son el patrón más limpio disponible.

Criterio de adopción fijado **antes** de correr nada: el consenso se adopta
solo si su F1 contra el humano supera **0.50**. No es arbitrario — el
recuperador que se quiere evaluar saca 0.344, y un instrumento que acierta
menos que eso mide parecido entre métodos, no acierto.

## Resultado: 0.23. No adoptable.

| ronda | P | R | F1 | marcas/consulta |
|---|---|---|---|---|
| estricta (`ronda_1`) | 0.18 | 0.19 | 0.18 | 2.7 |
| **inclusiva (`ronda_2`)** | 0.24 | 0.55 | **0.33** | 10.5 |
| de dominio (`ronda_3`) | 0.28 | 0.36 | 0.31 | 5.6 |
| literal (`ronda_4`) | 0.23 | 0.26 | 0.23 | 4.3 |

Y el consenso, según cuántos votos se exijan:

| umbral | F1 vs. humano |
|---|---|
| ≥1 voto (unión) | 0.33 |
| ≥2 votos | 0.30 |
| ≥3 votos | 0.23 |
| ≥4 votos (unanimidad) | **0.19** |

## Las dos lecturas que importan

**1. Más rondas y más acuerdo empeoran el resultado, no lo mejoran.** La
progresión 0.33 → 0.30 → 0.23 → 0.19 es monótona, y la mejor ronda individual
(la inclusiva, 0.33) iguala al mejor consenso. Combinarlas no aportó nada
sobre correr una sola.

**2. El acuerdo entre rondas no es señal de acierto.** Precisión de cada marca
según cuántas rondas la votaron:

| votos | aciertos/marcas | precisión |
|---|---|---|
| 4/4 | 6/23 | 0.26 |
| 3/4 | 2/9 | 0.22 |
| 2/4 | 3/8 | 0.38 |
| 1/4 | 6/33 | 0.18 |

**Plana.** La unanimidad no predice mejor que un voto suelto. Cuatro criterios
opuestos coinciden entre sí y se equivocan juntos: en q044 seis documentos
sacaron 4/4 votos y el humano solo había marcado uno. Lo que el consenso mide
es el sesgo compartido del método, no la relevancia.

Además, **13 de los 30 documentos que el humano marcó no los marcó ninguna
ronda**. El techo de recall es 57% antes de contar precisión.

## Cómo se usan estas etiquetas

Las 9 consultas anotadas así llevan `"anotador": "anotacion-asistida"` en
`dev/eval/ground_truth_mini.jsonl` y **no son intercambiables con las
humanas**. `eval_mini.py` desglosa por procedencia y ofrece `--solo-humanas`
para excluirlas. Cualquier cifra que dependa de ellas se cita con esa
salvedad.

## Conclusión

**No reabrir sin datos nuevos.** El camino del ground truth automático está
cerrado por medición, dos veces y con dos diseños distintos. Ampliar el ground
truth exige juicio humano.

Lo que sí sirvió de acá: los **términos discriminantes por consulta** que
produjo la ronda literal, y las notas de las cuatro sobre qué consultas son
irresolubles con el pool actual. Eso es diagnóstico del recuperador, no
etiquetas.

## Reproducir

```bash
python dev/eval/anotacion_asistida/build_dossier.py /tmp/dossier.md
# ...anotar el dossier cuatro veces con los criterios de arriba...
python dev/eval/anotacion_asistida/calibrar_rondas.py 3 dev/eval/anotacion_asistida/ronda_*.jsonl
python dev/eval/anotacion_asistida/precision_por_voto.py
```
