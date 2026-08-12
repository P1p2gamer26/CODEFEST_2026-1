# Handoff — corrida de gte en GPU (2 ago 2026)

Escrito para que esta corrida se pueda retomar **sin la sesión de chat que la
lanzó**. Si estás leyendo esto y no sabés qué está pasando, empezá acá.

---

> **TERMINADO (2 ago 2026, 18:14).** El índice está construido y verificado.
> Codificación: **440,7 min (7,34 h)** para 128.526 chunks a 4,9 chunks/s —
> la estimación era 7,3 h. `verificar_alineacion.py` pasa los cinco
> controles, y el re-rank con el índice completo reproduce los números del
> experimento barato **al decimal** (F1@3 0.378 / NDCG 0.393 sobre las 41).
>
> Nota para quien compare archivos: `res_gte_full.jsonl` **no** es
> byte-idéntico a `res_gte_w0.25.jsonl`. Se comprobó y es **benigno**: los
> vectores de ambas vías tienen similitud coseno ≥ 0,999999 y cero chunks
> descolocados. La diferencia es redondeo de fp32 —los lotes se agrupan
> distinto, el padding cambia y las reducciones en punto flotante no son
> asociativas— que altera algún desempate. Las métricas son idénticas.
>
> **ADOPTADO E INTEGRADO (2 ago 2026).** gte reemplazó a e5 como
> re-puntuador. `Entrega/resultados.jsonl` regenerado (sha256 `4d54533c…`),
> 94 tests en verde, validador correcto y **corrida en frío byte a byte**.
> Métricas entregadas: **F1@3 0.378 · NDCG@10 0.393** sobre las 41 anotadas;
> 0.333 · 0.362 sobre las 10 independientes.
>
> Este documento queda como registro de cómo se construyó y verificó el
> índice. El estado vigente está en `dev/docs/PLAN_MAESTRO.md`, sección "Adoptado:
> gte-multilingual-base como re-puntuador".
>
> Cómo reproducir las mediciones sin reconstruir nada (segundos, CPU):
>
> ```bash
> python dev/scripts/verificar_alineacion.py dev/intermedios/encoder_gte-multilingual-base
> python dev/scripts/eval_mini.py --resultados dev/intermedios/gte_rerank/res_gte_full.jsonl \
>        --solo-humanas --comparar-con Entrega/resultados.jsonl
> python dev/scripts/eval_mini.py --resultados dev/intermedios/gte_rerank/res_gte_full.jsonl \
>        --sin-pooling --comparar-con Entrega/resultados.jsonl
> python dev/scripts/victorias_ndcg.py dev/intermedios/gte_rerank/res_gte_full.jsonl \
>        Entrega/resultados.jsonl
> ```
>
> Y para regenerar `res_gte_full.jsonl` desde cero (~2 min, usa el índice ya
> construido, no re-codifica):
>
> ```bash
> python dev/scripts/rerank_indice_completo.py \
>     --rerank-index dev/intermedios/encoder_gte-multilingual-base \
>     --rerank-encoder gte-multilingual-base --peso 0.25 \
>     --out dev/intermedios/gte_rerank/res_gte_full.jsonl
> ```

## 1. Qué hay corriendo AHORA

Construcción del índice FAISS de `gte-multilingual-base` sobre los 128.526
chunks del corpus, **en GPU**, lanzada desacoplada de la sesión.

| | |
|---|---|
| script | `dev/scripts/indexar_desde_metadata.py` |
| intérprete | **`.venv-cuda\Scripts\python.exe`** (no `.venv`) |
| entrada | `Entrega/base_vectorial/encoder_paraphrase-multilingual-MiniLM-L12-v2/metadata.jsonl` |
| salida | `dev/intermedios/encoder_gte-multilingual-base/` |
| log | `dev/intermedios/gte_full/build.log` (y `.err`) |
| duración estimada | **~7,3 h** (medido: 204 ms/chunk en GPU) |
| vigilante de RAM | `dev/scripts/vigilante_ram.ps1`, umbral 450 MB |

**Ver cómo va** (tiempo transcurrido, faltante y recursos):

```powershell
powershell -File dev\scripts\estado_corrida.ps1 -Log "dev\intermedios\gte_full\build.log"
```

**Es reanudable.** Si el proceso muere, relanzá el mismo comando y retoma
desde el último checkpoint (graba cada 512 chunks). El checkpoint valida
contenido —filas, `chunk_id` de la última y **hash del texto**— así que si el
corpus cambió, el cache se descarta solo en vez de producir un índice con
vectores viejos en silencio.

```powershell
Start-Process -FilePath ".venv-cuda\Scripts\python.exe" -ArgumentList `
  "dev\scripts\indexar_desde_metadata.py","--metadata", `
  "Entrega\base_vectorial\encoder_paraphrase-multilingual-MiniLM-L12-v2\metadata.jsonl", `
  "--encoder-name","gte-multilingual-base", `
  "--out-dir","dev\intermedios\encoder_gte-multilingual-base" `
  -WindowStyle Hidden -RedirectStandardOutput "dev\intermedios\gte_full\build.log" `
  -RedirectStandardError "dev\intermedios\gte_full\build.err"
```

---

## 2. Qué hacer CUANDO TERMINE (en orden, sin saltarse nada)

### 2.1 Verificar la alineación — **obligatorio, no es opcional**

```bash
python dev/scripts/verificar_alineacion.py dev/intermedios/encoder_gte-multilingual-base
```

La cascada lee el vector del secundario con `reconstruct(fila)` asumiendo que
la fila N es el mismo chunk en los dos índices. **Si no lo es, no hay
excepción: solo resultados peores**, que se atribuirían al encoder. Si este
script falla, el índice no sirve — no seguir.

### 2.2 Re-medir con el índice completo

El re-rank ya se midió con los vectores del pool y dio los números de la
sección 3. Con el índice completo debería dar **exactamente lo mismo** (el
re-puntuador solo usa vectores de candidatos). Si difiere, hay un bug de
alineación que la verificación no detectó.

```bash
python dev/scripts/eval_mini.py --resultados dev/intermedios/gte_rerank/res_gte_w0.25.jsonl \
    --solo-humanas --comparar-con Entrega/resultados.jsonl
python dev/scripts/victorias_ndcg.py dev/intermedios/gte_rerank/res_gte_w0.25.jsonl Entrega/resultados.jsonl
```

### 2.3 Decidir si entra a la entrega

**La decisión ya está tomada en principio** (el usuario aprobó construir el
índice sabiendo los números), pero la regla de adopción sigue siendo la de
`docs/lecciones_metodologia.md`: IC al 90% que excluya una pérdida de 0.02,
**más justificación mecánica previa**, decidiendo **con NDCG@10** porque el
cambio toca fragmentos. Ver la salvedad de la sección 3.

Si entra: gte **reemplaza a e5** como re-puntuador (la cascada tiene uno
solo). Eso libera los 395 MB del índice de e5.

Pasos para adoptarlo:
1. Copiar el índice a `Entrega/base_vectorial/encoder_gte-multilingual-base/`.
2. Cambiar el default de `--rerank-encoder` en `Entrega/generador.py`.
3. **Re-aplanar** cualquier cambio del camino online en `generador.py`
   (punto 14 de las notas del proyecto: es autocontenido).
4. Regenerar `Entrega/resultados.jsonl` y correr:
   `pytest dev/tests -v` · `python dev/scripts/validar_entrega.py` ·
   **corrida en frío** desde fuera del repo con `PYTHONPATH` vacío,
   verificando que reproduce byte a byte.
5. Publicar el índice por **GitHub Release**, no por LFS (punto 16 de
   `dev/docs/PLAN_MAESTRO.md`: la cuota de LFS está casi agotada y no se puede liberar).

---

## 3. Los números medidos, y la salvedad honesta

Re-rank con MiniLM primario + gte re-puntuando (peso 0.25, el mismo esquema
que la cascada entregada):

| | F1@3 | NDCG@10 |
|---|---|---|
| entrega actual (MiniLM→e5), 41 anotadas | 0.344 | 0.338 |
| **MiniLM→gte, 41** | **0.378** | **0.393** |
| entrega actual, 10 independientes | 0.333 | 0.329 |
| **MiniLM→gte, 10 independientes** | 0.333 | **0.362** |

Deltas pareados, IC al 90%:

- 41: F1 **+0.034 [−0.010, +0.078]** · NDCG **+0.055 [+0.013, +0.100]**
- 10: F1 **−0.000 [−0.067, +0.067]** · NDCG **+0.033 [−0.015, +0.101]**

**La salvedad que hay que respetar:** el 0.378 sería el mejor F1@3 del
proyecto, pero **la ganancia de F1 desaparece por completo en la muestra
independiente**. Es la firma del sesgo de pooling, la misma por la que se
descartó `doc_rrf`. Lo que se sostiene en las dos muestras es el NDCG@10.
**No vender el +0.034 de F1 como resultado firme.**

Justificación mecánica, medida ANTES del experimento (no post-hoc):
penalización por idioma —mismo contenido en inglés vs. español— de
**gte −0.027/+0.036 contra MiniLM +0.052/+0.091**. El fallo documentado del
pool es cross-lingual (NBQR/CBRN), así que el mecanismo encaja.

---

## 4. Trampas de gte que ya costaron tiempo

1. **Se carga ROTO.** Declara los buffers de RoPE con `persistent=False`, no
   viajan en el checkpoint y `transformers` los materializa desde memoria sin
   inicializar. `position_ids` revienta con `IndexError`; **`inv_freq` NO
   revienta** — codifica sin información posicional y devuelve vectores
   normalizados y basura. Lo arregla `_reparar_buffers_no_persistentes()` en
   `src/embedding/encoders.py`. **Si se toca ese archivo, no quitarla.**
2. **En CPU hay que apagar** `unpad_inputs` y `use_memory_efficient_attention`
   (ya está en `KNOWN_ENCODERS`), y eso vuelve la atención cuadrática.
3. **`max_seq_length` se recorta a 512** (viene en 8192). Cubre el 99,3% de
   los chunks y evita padding inútil.
4. **En CPU son 97 h; en GPU 7,3 h.** Medir `ms/chunk` sobre 64 chunks reales
   antes de lanzar cualquier corrida larga — `scratchpad/medir_ritmo.py`.

---

## 5. Todo lo demás que cambió en esta sesión

Ninguno de estos puntos depende de la corrida de gte; están hechos y
verificados.

- **Fallo de exclusión corregido**: `Entrega/generador.py` usaba sintaxis de
  Python 3.10+ y ADL evalúa con **≥3.9.5**. Sin `from __future__ import
  annotations` el script moría al importar. Ver la sección "Cumplimiento" de
  `dev/docs/PLAN_MAESTRO.md`.
- **Fragmentos alineados con los documentos** (`ordenar_para_fragmentos`):
  NDCG@10 de **0.206 → 0.338**, F1@3 intacto. Ver "Adoptado" en `dev/docs/PLAN_MAESTRO.md`.
- **Ground truth en 50/50**, con las 9 últimas etiquetadas por panel de
  agentes y **marcadas como tales** (`anotador: "anotacion-asistida"`). Usar
  `eval_mini.py --solo-humanas` para comparar contra cualquier medición
  anterior al 2 ago 2026.
- **Panel de agentes: medido y descartado** como fuente de ground truth
  (F1 0.23 contra el humano). Ver `dev/eval/anotacion_asistida/README.md`.
- **Rescate léxico**: sirve para ANOTAR, no para producción (pierde 15-2).
- `.venv-cuda` es un entorno **aparte**; `.venv` sigue siendo el que produce
  la entrega y no hay que tocarlo.

---

## 6. Estado de la entrega en este momento

**Válida, verificada y sin depender de nada de lo anterior.**

- `pytest dev/tests` → 94 passed
- `python dev/scripts/validar_entrega.py` → verde
- corrida en frío desde fuera del repo → reproduce byte a byte
  (sha256 `570c092c…`)
- F1@3 **0.344** · NDCG@10 **0.338** sobre las 41 anotadas a mano

Nada de lo que está corriendo puede romperla: el índice de gte se escribe en
`dev/intermedios/`, nunca en `Entrega/`.
