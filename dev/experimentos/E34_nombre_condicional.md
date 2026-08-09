# E34 — el desempate condicional pierde exactamente igual que el incondicional (9 ago 2026)

`dev/scripts/dispersion_doc_e34.py` (puerta) + `dev/scripts/barrido_nombre_cond_e34.py`.
Sin FAISS: pools volcados + el encoder MiniLM para 326 nombres. Segundos de CPU.
La fila base reproduce **0.455 / 0.516 / 0.499** y **0.433 / 0.474 / 0.467**
digito a digito (regla de E09), o sea que mide la entrega ACTUAL con E32 dentro.

## Puerta de entrada: el detector SI induce una particion

Al reves que E29. La dispersion de los scores **agregados de documento** (sobre
el pool ya filtrado por fenomeno) es bimodal, con un hueco vacio en medio:

| magnitud | particion | hueco |
|---|---|---|
| dispersion relativa top-5 `(s1-s5)/s1` | **18 abajo / 32 arriba** | 0.051 → 0.190 (vacio de 0.14) |
| gap relativo doc3→doc4 `(s3-s4)/s3` | 41 abajo / 9 arriba | 0.037 → 0.161 (vacio de 0.12) |
| dispersion dentro de los saturados | **ninguna** | continua, 0.001 a 0.084 |

Umbrales fijados con esto a la vista y antes de puntuar nada: **0.10 en las dos
primeras**, que es el punto medio de cada hueco. La tercera se descarta por
continua. No hay grilla.

`disp5 < 0.10` dispara en **18/50** — un subconjunto de verdad, no E24
disfrazado. `gap34 < 0.10` dispara en 41/50 y va como CONTROL, para comprobar
el veto (c).

## Resultado

| celda | F1(50) | ND(50) | NDp(50) | F1(ind) | ND(ind) | NDp(ind) | F1(hum) | ND(hum) | NDp(hum) |
|---|---|---|---|---|---|---|---|---|---|
| **entregada** | **0.455** | **0.516** | **0.499** | **0.433** | **0.474** | **0.467** | **0.486** | **0.537** | **0.520** |
| **cond-disp5** (18/50) | 0.431 | 0.476 | 0.459 | 0.400 | 0.406 | 0.400 | 0.466 | 0.497 | 0.480 |
| cond-gap34 (41/50) | 0.433 | 0.484 | 0.467 | 0.400 | 0.406 | 0.400 | 0.467 | 0.498 | 0.480 |
| e24-incondicional (50/50) | 0.433 | 0.484 | 0.467 | 0.400 | 0.406 | 0.400 | 0.467 | 0.498 | 0.480 |

Deltas pareados con IC al 90%, celda que decide (`cond-disp5`):

    F1(50)   -0.023 [-0.053, +0.007]   1g/5p   NO pasa
    ND(50)   -0.040 [-0.082, -0.003]   5g/6p   NO pasa
    NDp(50)  -0.040 [-0.080, -0.006]   5g/7p   NO pasa
    F1(ind)  -0.033 [-0.100, +0.000]   0g/1p   NO pasa
    ND(ind)  -0.068 [-0.223, +0.021]   2g/1p   NO pasa
    NDp(ind) -0.066 [-0.220, +0.021]   2g/1p   NO pasa
    F1(hum)  -0.020 [-0.053, +0.012]   1g/4p   NO pasa
    ND(hum)  -0.040 [-0.087, +0.002]   5g/5p   NO pasa
    NDp(hum) -0.040 [-0.084, -0.001]   5g/6p   NO pasa

**Las nueve lecturas pierden**, y tres tienen el IC entero bajo cero. Cambia
45 de 150 documentos.

Desglose humanas / agente: F1 humanas(41) 0.486 → 0.466, agente(9) 0.311 →
0.274. **Pierde por igual en las dos**; no hay aqui nada de la firma de E31
(ganancia que vive en las 9 peor etiquetadas) — simplemente pierde en todas
partes.

## Lo que mata la hipotesis: condicionar no separa nada

La idea era que el nombre acertara donde el contenido esta empatado y estorbara
donde no. **Las cinco derrotas de E24 caen TODAS dentro de las 18 consultas
empatadas.** Reparto de F1@3 de `cond-disp5`:

    gana   q003 (0.00 -> 0.50)
    pierde q009 (0.33 -> 0.00)  q020 (0.67 -> 0.33)  q032 (0.67 -> 0.33)
           q041 (1.00 -> 0.67)  q048 (0.67 -> 0.33)

Condicionar **conserva intacto todo el dano de E24 y pierde una de sus dos
victorias**: q037 (0.00 → 0.40 en E24) tiene el top-5 disperso, no dispara, y
se queda en cero. Por eso `cond-disp5` es **peor que el incondicional** en las
50 (ND 0.476 contra 0.484).

Los vetos:

- **(a) SE ACTIVA: q041 baja de 1.00 a 0.67**, en las tres celdas. Es el mismo
  fallo por el que murio E24 y el filtro no lo toca.
- (b) sube q003, que valia 0. La justificacion mecanica se cumple otra vez —
  igual que en E24 — y otra vez no alcanza.
- (c) `cond-gap34` dispara en 41/50 y da **exactamente los mismos numeros que
  el incondicional en las nueve lecturas**: es E24 con otro nombre, tal como el
  riesgo pre-registrado anticipaba.

## Veredicto

**No adoptable, y el camino queda cerrado.** `Entrega/` no se toca.

E24 cerro diciendo "un criterio que se disparara solo cuando el ranking por
contenido es plano seguiria vivo". **Ya no lo esta.** El detector existe y es
bueno —la particion es limpia, con un hueco vacio de 0.14, cosa que E29 no
consiguio— y aun asi no sirve: **la planitud del ranking agregado no
correlaciona con que el nombre acierte.** El nombre acierta en q003 y falla en
q020/q032/q041/q048, y las cinco son consultas planas.

Lo que queda vivo del material de E24 es solo la observacion, no la palanca:
el nombre del documento sigue siendo la unica senal del proyecto que mueve
alguna de las 11 consultas con F1@3 = 0. Para usarlo haria falta un detector
que separe **por que documento**, no por consulta — y eso es otra hipotesis
sin candidato a la vista.
