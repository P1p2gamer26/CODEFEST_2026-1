# Consultas

## Las oficiales

- **`consultas_50_oficiales.jsonl`** -- las 50 consultas `q001`-`q050` que
  entregó ADL, transcritas de `corpus_meta/Extracto_Preguntas_50_v2.pdf`. Es el
  archivo que se usa para generar `Entrega/resultados.jsonl`:

  ```bash
  python Entrega/generador.py --consultas dev/consultas_prueba/consultas_50_oficiales.jsonl
  ```

  Las 50 vienen en español y con tildes; se transcriben tal cual, sin
  normalizar. Se reparten entre los tres fenómenos: `q001`-`q016` IA y
  capacidades estratégicas, `q017`-`q032` seguridad del entorno espacial,
  `q033`-`q050` dinámicas territoriales. No traen ground truth.

## Las de prueba (provisionales, del período previo)

Consultas escritas a mano por el equipo antes de que llegaran las oficiales.
Se conservan porque cubren los tres idiomas del corpus y los casos con errores
de tipeo, que las oficiales no ejercitan.

- `consultas_prueba.jsonl` -- 15 consultas, la versión original más corta.
- `consultas_50.jsonl` -- 50 consultas `p001`-`p050`, balanceadas entre los 3
  fenómenos y los 3 idiomas (~17 ES / 17 EN / 16 PT). Incluye a propósito unas
  8 consultas con errores de tipeo (`p036`, `p037`, `p038`, `p039`, `p042`,
  `p044`, `p045`) para comprobar que la recuperación aguanta consultas reales
  de usuario, no solo texto perfecto.

El formato esperado (`query_id`/`id` + `text`/`query`) está aislado en
`load_consultas()` dentro de `Entrega/generador.py`.
