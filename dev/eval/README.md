# Mini ground truth (anotado por el equipo)

`ground_truth_mini.jsonl` anota, para 10 de las 50 consultas oficiales, los
documentos que consideramos relevantes. **No es el ground truth de ADL** — ADL
no lo publica. Sirve para una sola cosa: comparar configuraciones del sistema
entre sí (un encoder vs. dos, con grafo vs. sin grafo, un chunking vs. otro).

Se corre con:

```bash
python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl
```

## Cómo se construyó

Para cada consulta se contaron las menciones de un conjunto de palabras clave
sobre `intermedios/chunks_intermedios.jsonl`, y de los documentos con más
menciones se conservaron los que, por su título y observatorio de origen,
responden efectivamente a la pregunta (p. ej. para q049, "restitución de
tierras", los cuatro monográficos de MAPP/OEA sobre restitución).

Se eligieron 10 consultas repartidas entre los tres fenómenos y con temas
específicos, donde la relevancia es discutible menos veces: q005 y q014 (F1),
q017, q020 y q026 (F2), q033, q040, q042, q044 y q049 (F3).

## Limitaciones (importantes al leer los números)

- **La anotación es parcial.** Solo se marcaron 4–6 documentos por consulta de
  un corpus de 1826. Un documento no anotado cuenta como irrelevante aunque
  quizá no lo sea, así que el F1@3 que reporta `eval_mini.py` es una **cota
  inferior**: el real será mayor o igual. Los valores absolutos no significan
  nada; lo que importa es el orden entre configuraciones.
- **Sesgo léxico.** Los candidatos salieron de contar palabras, así que el
  conjunto favorece a los documentos que usan literalmente el vocabulario de la
  consulta. Un encoder semántico puede traer un documento igualmente bueno que
  no esté anotado y salir castigado.
- **Con |relevantes| > 3 el F1@3 nunca llega a 1.** Con 5 relevantes y los 3
  aciertos posibles, el techo es 0.75. No es un error.
- No se anotó relevancia de fragmentos, así que **no se calcula NDCG@10**.
