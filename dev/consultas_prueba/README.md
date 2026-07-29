# Consultas de prueba (PROVISIONAL)

Ninguno de los archivos de esta carpeta son las 50 consultas oficiales
`q001`-`q050` de ADL -- esas todavia no se entregan. Son consultas escritas a
mano por el equipo, en el mismo estilo, solo para probar el pipeline mientras
tanto.

- `consultas_prueba.jsonl` -- 15 consultas, la version original mas corta.
- `consultas_50.jsonl` -- 50 consultas, balanceadas entre los 3 fenomenos y
  los 3 idiomas del corpus (~17 ES / 17 EN / 16 PT). Incluye a proposito unas
  8 consultas con errores de tipeo (tildes faltantes, letras trocadas, una
  palabra mal escrita: ver `p036`, `p037`, `p038`, `p039`, `p042`, `p044`,
  `p045`) para poner a prueba que la recuperacion siga funcionando razonablemente
  bien ante consultas reales de usuario, no solo ante texto perfecto.

Cuando ADL entregue el archivo oficial de consultas, se usa en su lugar con
`--consultas <archivo_oficial>` (CLI) o desde el selector de archivo en la GUI
(`scripts/gui_app.py`) -- el formato esperado (`query_id`/`id` + `text`/`query`)
esta aislado en `load_consultas()` dentro de `Entrega/generador.py`.
