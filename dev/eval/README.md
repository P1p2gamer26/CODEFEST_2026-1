# Mini ground truth

`ground_truth_mini.jsonl` anota, para **41 de las 50 consultas oficiales**, los
documentos que consideramos relevantes. **No es el ground truth de ADL** — ADL
no lo publica. Sirve para una sola cosa: comparar configuraciones del sistema
entre sí (un encoder vs. dos, con grafo vs. sin grafo, un chunking vs. otro).

```bash
python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl
python dev/scripts/eval_mini.py --resultados Entrega/resultados.jsonl --sin-pooling
```

## Dos procedencias, y por qué importa la diferencia

Cada línea lleva un campo `pool` que dice de dónde salieron sus candidatos:

- **Sin `pool` (10 consultas)** — anotación independiente: los candidatos se
  obtuvieron contando palabras clave sobre `intermedios/chunks_intermedios.jsonl`,
  sin pasar por el recuperador. Son q005, q014 (F1), q017, q020, q026 (F2),
  q033, q040, q042, q044 y q049 (F3).
- **Con `pool` (31 consultas)** — anotación por *pooling*: los candidatos los
  propuso el propio recuperador (`anotar_candidatos.py`, `k_pool=200`) y se
  marcaron leyendo sus extractos. Es la técnica de TREC.

**Para comparar dos encoders hay que usar `--sin-pooling`.** Una consulta cuyos
candidatos propuso el encoder X favorece a X: un documento que X nunca recuperó
jamás pudo marcarse como relevante. Lo mismo vale para calibrar `k_pool`, porque
el pool se generó con un valor concreto.

Medido: la diferencia entre ambos conjuntos resultó pequeña (0.302 sobre 19 vs.
0.300 sobre las 10 independientes), porque se anota con `k_pool=200` y la
entrega usa 60, así que las marcas no se cumplen solas.

## Limitaciones (importantes al leer los números)

- **La anotación es parcial**, de un corpus de 1826 documentos. Un documento no
  anotado cuenta como irrelevante aunque quizá no lo sea, así que el F1@3 es una
  **cota inferior**. Los valores absolutos no significan nada; lo que importa es
  el orden entre configuraciones.
- **El promedio no basta para decidir.** Con ~30 consultas cada una pesa 0.033,
  así que dos que cambien de lado por azar mueven la media más que un efecto
  real. Usar `barrido_retrieval.py --comparar A B`, que cuenta en cuántas
  consultas gana cada configuración. Ejemplo real: `k_pool` 30 parecía superar a
  60 por promedio (0.309 vs 0.274) y resultó ser un reparto 5-3 sobre 8
  consultas, indistinguible del azar.
- **Se juzga el documento, pero solo se ve un fragmento.** `candidatos.md`
  muestra el mejor chunk de cada documento, que no siempre es la mejor evidencia
  de relevancia: en q046 el mejor chunk de `F3-MAPPOEA-016` hablaba de
  conflictividad social, aunque el documento sí trata la minería ilícita. Esto
  hace que se marquen de menos, nunca de más.
- **Las 31 consultas por pooling las juzgó el asistente, no el equipo.** Están
  para revisarse. Las de fenómeno 1 son las menos fiables: casi todo candidato
  es "un informe de IA de defensa que roza el tema" y separarlos exige criterio
  de dominio.
- No se anotó relevancia de fragmentos, así que **no se calcula NDCG@10**, que
  es la mitad del puntaje oficial.

## Estado: las 50 consultas están procesadas

**41 anotadas** y **9 revisadas sin ninguna marca**. No queda ninguna pendiente.

## Las 9 sin ningún documento marcado

En estas se revisaron los 10 candidatos y **ninguno respondía** a la consulta,
así que no están en el ground truth. Son fallos de recuperación documentados,
no consultas por anotar: q001, q007, q008, q011, q012, q015, q028, q038, q048.

Vale la pena mirarlas juntas, porque son el mejor diagnóstico disponible de
dónde falla el recuperador. Casos con causa identificada:

- **q001** (amenazas NBQR): el término no existe en el corpus, pero **CBRN**, su
  equivalente inglés, sale en 62 chunks de 20 documentos. Irrecuperable tanto
  por vía densa como léxica.
- **q015** (dependencia de semiconductores y hardware): ningún candidato trata
  el tema, pese a que CSET sí tiene documentos sobre infraestructura de cómputo
  (aparecen como candidatos de q014).
- **q013** (amenazas cibernéticas a IA en infraestructura crítica): 7 de 10
  candidatos eran secciones de ciberseguridad **espacial** de los informes
  Global Counterspace. La palabra "cibernéticas" arrastró el dominio equivocado.
- **q008, q011, q012** (fenómeno 1): los candidatos son informes de IA de
  defensa temáticamente adyacentes, sin que ninguno responda a lo preguntado.

Los que siguen sin causa clara:

- **q007** (sistemas autónomos frente al DIH): devolvió directrices espaciales
  de UNOOSA, atlas de RESDAL e informes de MAPP/OEA, pese a que los informes de
  SIPRI sobre LAWS están en el corpus.
- **q038** (innovaciones tácticas de los grupos armados) y **q034**: devolvieron
  presupuestos de defensa de EEUU y atlas de RESDAL sobre la organización de las
  fuerzas armadas estatales.

### Sobre la supuesta confusión "grupos armados" vs. "fuerzas armadas"

Medida y **redimensionada**, conviene no repetir el error. La primera medición
usó el patrón estrecho `grupos? armados? (ilegal|organizado)` y dio 0 menciones
en documentos con 218–329 de "fuerzas armadas", lo que parecía un fallo
categórico. Con un patrón justo, que incluya "crimen organizado" y "grupos
criminales", **solo 4 de 48 cupos de documento (8%) en las 16 consultas sobre
grupos armados carecen de toda mención del sujeto** — y q038 y q039 dan 0/3.

Lo que queda en pie es más débil y más matizado: los atlas de RESDAL mencionan
"crimen organizado" 6 veces en 984.000 caracteres (20 en 1,2 MB para el de
2024), es decir densidad de ruido de fondo en documentos que tratan de otra
cosa. Son mala respuesta, pero **no por ausencia del término**. Cualquier
propuesta de recuperación léxica (BM25) apoyada en este fenómeno debe
justificarse midiendo densidad, no presencia.

## Cómo ampliarlo

```bash
python dev/scripts/anotar_candidatos.py --generar      # -> candidatos.md
# marcar [x] a mano en candidatos.md
python dev/scripts/anotar_candidatos.py --recolectar
```
