# Panel de anotadores-agente: medido y descartado (2 ago 2026)

Segundo y **último** intento de cribar el ground truth con agentes. El primero
usó un solo anotador y dio F1 0.28 contra las etiquetas humanas; quedó la
salvedad de que el consenso de varios podría ser mejor. **No lo es.** Este
experimento la cierra.

## Diseño

Cuatro anotadores con sesgos deliberadamente opuestos —**estricto**,
**inclusivo**, **dominio militar**, **literal/terminológico**— sobre el mismo
dossier de 15 consultas (`build_dossier.py`), a ciegas: **9 sin anotar
mezcladas con 6 ya anotadas a mano**, barajadas, sin decirles cuáles eran
cuáles. Las 6 de calibración son las *independientes* (sus candidatos no los
propuso el recuperador), así que son el patrón más limpio disponible.

Criterio de adopción fijado **antes** de correr nada: el consenso se adopta
solo si su F1 contra el humano supera **0.50**. No es arbitrario — el
recuperador que se quiere evaluar saca 0.344, y un anotador que acierta menos
que eso mide parecido entre modelos, no acierto.

## Resultado: 0.23. No adoptable.

| anotador | P | R | F1 | marcas/consulta |
|---|---|---|---|---|
| estricto | 0.18 | 0.19 | 0.18 | 2.7 |
| **inclusivo** | 0.24 | 0.55 | **0.33** | 10.5 |
| dominio | 0.28 | 0.36 | 0.31 | 5.6 |
| literal | 0.23 | 0.26 | 0.23 | 4.3 |

Y el consenso, según cuántos votos se exijan:

| umbral | F1 vs. humano |
|---|---|
| ≥1 voto (unión) | 0.33 |
| ≥2 votos | 0.30 |
| ≥3 votos | 0.23 |
| ≥4 votos (unanimidad) | **0.19** |

## Las dos lecturas que importan

**1. Más anotadores y más acuerdo empeoran el resultado, no lo mejoran.** La
progresión 0.33 → 0.30 → 0.23 → 0.19 es monótona, y el mejor anotador
individual (el inclusivo, 0.33) iguala al mejor consenso. El panel no aportó
nada sobre correrlo solo.

**2. El acuerdo entre agentes no es señal de acierto.** Precisión de cada
marca según cuántos agentes la votaron:

| votos | aciertos/marcas | precisión |
|---|---|---|
| 4/4 | 6/23 | 0.26 |
| 3/4 | 2/9 | 0.22 |
| 2/4 | 3/8 | 0.38 |
| 1/4 | 6/33 | 0.18 |

**Plana.** La unanimidad no predice mejor que un voto suelto. Cuatro agentes
con sesgos opuestos coinciden entre sí y se equivocan juntos: en q044 seis
documentos sacaron 4/4 votos y el humano solo había marcado uno. Lo que el
consenso mide es el sesgo compartido del modelo, no la relevancia.

Además, **13 de los 30 documentos que el humano marcó no los marcó ningún
agente**. El techo de recall del panel es 57% antes de contar precisión.

## Conclusión

**No reabrir sin datos nuevos.** El camino del ground truth automático está
cerrado por medición, dos veces y con dos diseños distintos. Ampliar el ground
truth exige juicio humano.

Lo que sí sirvió de acá: los **términos discriminantes por consulta** que
produjo el anotador literal, y las notas de los cuatro sobre qué consultas son
irresolubles con el pool actual. Eso es diagnóstico del recuperador, no
etiquetas.

## Reproducir

```bash
python dev/eval/panel_agentes/build_dossier.py /tmp/dossier.md
# ...pasarle el dossier a cuatro agentes con los sesgos de arriba...
python dev/eval/panel_agentes/calibrar_panel.py 3 dev/eval/panel_agentes/panel_*.jsonl
python dev/eval/panel_agentes/precision_por_voto.py
```
