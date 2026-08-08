# La entrega ya no reproduce byte a byte entre maquinas (7 ago 2026, noche)

`CLAUDE.md` afirma que `Entrega/resultados.jsonl` "reproduce byte a byte" en
corrida en frio. **Esa afirmacion envejecio y hay que acotarla** (leccion 5):
se verificaba en la MISMA maquina donde se habia generado el archivo, y desde
E06 el archivo lo genera la VM.

## Lo medido

Corrida en frio en la maquina local (Windows, fuera del repo, `PYTHONPATH`
vacio, sin ningun flag salvo `--consultas`), contra el `resultados.jsonl`
generado en la VM (Rocky 9, Xeon):

| | resultado |
|---|---|
| lineas distintas | **2 de 50** (q005, q026) |
| F1@3 sobre las 50 | **0.440 en las dos** |
| NDCG@10 | **0.490 en las dos** |
| dos corridas locales entre si | **sha256 identico** (`247c46de...`) |

O sea: **el pipeline es determinista dentro de una maquina** y las metricas no
se mueven. Lo que cambia es el desempate entre candidatos casi iguales.

## Por que, con los numeros

Son **dos causas distintas**, y solo una es arreglable:

1. **q005 — casi-empate real, NO arreglable.** La brecha entre el 3er y el 4o
   documento es de **2.6e-05** (`F3-RESDAL-093` 8.299890863895417 contra
   `F1-ILIA-003` 8.299865067005157). No es un empate: son dos documentos que
   puntuan casi igual, y la aritmetica de coma flotante de dos CPU distintas
   los ordena distinto. Ningun criterio de desempate arregla esto, porque los
   scores realmente difieren.

2. **q026 — empates EXACTOS, si arreglable.** Cuatro fragmentos de
   `F2-CSIS-049` comparten el score **1.6724899768829347 bit a bit**. Con
   empate exacto, el orden lo decide el orden de llegada desde FAISS, que
   depende de la maquina. Un desempate estable por `chunk_id` los volveria
   reproducibles en cualquier entorno.

## Que NO se hizo, y por que

No se toco `Entrega/` (regla 6 del handoff: es decision humana). El
desempate estable es un cambio de una linea por cada `sort` de
`generador.py`, pero:

- **No es gratis en metrica.** Resolver un empate al reves cambia que
  fragmento entra al top-10; los scores son iguales pero la relevancia no
  tiene por que serlo. Hay que medirlo como cualquier otra palanca, con el
  criterio de la regla 4.
- Obliga a re-aplanar el cambio en `generador.py` (punto 14) y a regenerar
  `resultados.jsonl`.

## Lo que importa para la sec. 1.4

La especificacion pide que la entrega sea **reproducible**, no que sea
bit-identica en cualquier CPU del mundo — eso ultimo no lo cumple ningun
sistema con aritmetica de punto flotante. Lo que el evaluador va a obtener es
**el mismo sistema con las mismas metricas** y 48 de 50 lineas identicas.
El riesgo real seria que las metricas se movieran, y no se mueven.

**Lo que si hay que corregir es el texto de `CLAUDE.md`**, que promete mas de
lo que se puede cumplir. La propiedad correcta es: "determinista por maquina;
entre maquinas difiere en los casi-empates, sin mover las metricas".
