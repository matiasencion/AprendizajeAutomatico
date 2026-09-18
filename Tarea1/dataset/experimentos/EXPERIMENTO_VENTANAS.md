# Experimento: sensibilidad a las ventanas temporales

## 2026-09-16 — window_years vs. years_limit/matches_limit

Este documento aclara una posible confusión de nombres y registra un
experimento sobre **dos conceptos distintos que ambos involucran "años"**:

1. **`window_years`**: vive en los notebooks (`bayesNotebook.ipynb`,
   `treeNotebook.ipynb`). Es cuántos años hacia atrás se usan para
   **entrenar** cada fold de la validación cruzada temporal (hoy fijo en
   5: para validar 2020 se entrena con 2015-2019).
2. **`years_limit` / `matches_limit`**: viven en `load.py`, dentro de
   `load_attributes`. Son la ventana usada para **calcular los atributos
   históricos de cada partido** (`record_difference`, `draw_rate_average`,
   goles a favor/en contra, etc.). Antes eran valores fijos por defecto
   (`years_limit=1`, `matches_limit=5`) que no se podían variar sin tocar
   el código; no estaban en ningún espacio de búsqueda de hiperparámetros.

Como en la conversación anterior "lo de los años" se prestaba a
confusión entre estos dos, se probaron los dos por separado. **En ningún
paso se usó el conjunto de test 2024-2025.**

## Metodología

- Se extendió la validación cruzada de 2013-2022 (10 años) a **2005-2022
  (18 años)**, aplicando la recomendación de reducir el riesgo de
  sobreajustar la elección de hiperparámetros a pocos años de validación.
  La distribución de resultados de la década 2000 ya es similar a la
  actual (ver `EXPERIMENTO_PARIDAD.md`), así que esto no reintroduce el
  problema de "otro fútbol".
- Se aplicó el **criterio de 1 error estándar**: además de la media de
  F1 macro entre folds, se calculó el error estándar (`std / sqrt(n_folds)`)
  y se reportan como *estadísticamente equivalentes* todos los candidatos
  cuya media cae dentro de 1 SE del mejor, en vez de quedarse ciegamente
  con el máximo.
- El resto de los hiperparámetros (márgenes de discretización, `m` de
  Bayes, `min_info_gain` del árbol, umbrales de `match_evenness`) se
  mantuvo fijo en los mejores valores ya encontrados, para aislar el
  efecto de lo que se estaba probando en cada parte.
- **Advertencia importante**: los márgenes de discretización fueron
  calibrados originalmente para `years_limit=1, matches_limit=5`. Si se
  cambia esa ventana, la escala de `record_difference` y
  `last_matches_difference` cambia, y esos márgenes dejan de ser
  necesariamente los óptimos. Los resultados de `years_limit`/
  `matches_limit` de abajo son entonces una **primera señal**, no una
  comparación 100% justa (para serlo habría que re-buscar los márgenes
  para cada ventana, mucho más costoso).

## Resultados

### A) `window_years` (ventana de entrenamiento de la CV), 18 folds

| window_years | Bayes F1 macro (± SE) | Árbol F1 macro (± SE) |
|---|---|---|
| 3 | 0.4113 ± 0.0091 | 0.3817 ± 0.0051 |
| **5 (actual)** | **0.4199 ± 0.0088** | **0.4001 ± 0.0062** |
| 7 | 0.4190 ± 0.0093 | 0.3938 ± 0.0076 |
| 10 | 0.4201 ± 0.0085 | 0.3814 ± 0.0100 |

- **Bayes**: 5, 7 y 10 son estadísticamente equivalentes (todos dentro de
  1 SE del mejor); sólo 3 años queda claramente peor.
- **Árbol**: 5 es claramente el mejor, ningún otro valor entra en la banda
  de 1 SE.

**Conclusión: `window_years=5` (el valor que ya estaban usando) es óptimo
o empatado con el óptimo en ambos modelos.** No había evidencia previa de
esto — era una elección no probada — y ahora sí la hay: no hace falta
cambiarlo.

### B) `years_limit` (ventana de atributos históricos), `matches_limit=5` fijo

| years_limit | Bayes F1 macro (± SE) | Árbol F1 macro (± SE) |
|---|---|---|
| **1 (actual)** | 0.4199 ± 0.0088 | **0.4001 ± 0.0062** |
| 2 | 0.4244 ± 0.0093 | 0.3758 ± 0.0062 |
| 3 | 0.4261 ± 0.0087 | 0.3829 ± 0.0064 |

- **Bayes**: hay una tendencia creciente (1→2→3), pero los tres valores
  quedan dentro de 1 SE entre sí — no es una diferencia estadísticamente
  sólida con 18 folds, aunque la tendencia monótona sugiere que vale la
  pena revisarlo con una búsqueda más seria.
- **Árbol**: `years_limit=1` es claramente mejor; 2 y 3 quedan bien por
  fuera de la banda de 1 SE (peor, no sólo distinto).

**Conclusión: los dos modelos no coinciden.** El árbol empeora claramente
con ventanas más largas; Bayes insinúa una mejora que no llega a ser
concluyente. Como ambos modelos comparten el mismo dataset/atributos, no
tiene sentido subir `years_limit` sólo para Bayes a costa de degradar el
árbol. Además, por la advertencia de arriba, la comparación para Bayes no
es limpia porque sus márgenes no se re-buscaron para `years_limit=2` o `3`.

### C) `matches_limit` (ventana de "forma reciente"), `years_limit=1` fijo

| matches_limit | Bayes F1 macro (± SE) | Árbol F1 macro (± SE) |
|---|---|---|
| 3 | 0.4136 ± 0.0090 | 0.3846 ± 0.0071 |
| **5 (actual)** | **0.4199 ± 0.0088** | **0.4001 ± 0.0062** |
| 8 | 0.4167 ± 0.0093 | 0.3877 ± 0.0079 |

- **Bayes**: los tres valores quedan dentro de 1 SE entre sí (empate
  técnico), con 5 como el de mayor media.
- **Árbol**: 5 es claramente el mejor.

**Conclusión: `matches_limit=5` (el valor actual) es el mejor o empatado
con el mejor en ambos modelos.** Tampoco hacía falta cambiarlo.

## Qué cambié en el código y por qué

1. **`load.py` — `load_dataset` ahora acepta `years_limit=1,
   matches_limit=5` como parámetros opcionales** (antes estaban
   hardcodeados dentro de la función, sin forma de variarlos sin editar
   el código fuente). Los valores por defecto son exactamente los que ya
   se venían usando, así que **no cambia el comportamiento de las
   notebooks existentes**; sólo lo hace configurable para poder correr
   este tipo de experimentos. Aproveché para borrar una función interna
   `transform(self, X)` que quedó de algún refactor anterior: no se
   llamaba desde ningún lado y hacía referencia a `self.years_limit` /
   `self.matches_limit`, que no existían en ese contexto (código muerto y
   roto).

2. **No cambié ningún valor por defecto** (`window_years` sigue en 5 en
   los notebooks, `years_limit`/`matches_limit` siguen en 1 y 5 en
   `load.py`). La razón, a diferencia del atributo de paridad del
   experimento anterior, es que **la evidencia de este experimento
   respalda los valores que ya tenían**, no uno distinto. Cambiar algo
   sin evidencia de que mejora sería agregar riesgo sin beneficio.

3. **Actualización posterior (confirmada por el usuario):** se actualizó
   `validation_years` a `list(range(2005, 2023))` en ambos notebooks y se
   volvió a correr la búsqueda completa de 500 candidatos con 18 folds
   (≈9000 ajustes) en cada uno. El detalle de esa corrida, las fórmulas de
   error estándar y los resultados finales quedaron en
   `INFORME_VALIDACION_CRUZADA.md`. Los hiperparámetros ganadores no
   cambiaron sustancialmente y el F1 macro medio anual se mantuvo
   prácticamente igual; lo que mejoró fue la confiabilidad estadística de
   la elección (error estándar del ganador ~31-34% menor).

## Reproducibilidad

Script: `window_experiment.py`. Resultados completos (todas las
combinaciones, ambos modelos, con reportes por clase) en
`resultados/experimento_ventanas_resultados.json`. No se usó el conjunto de test
2024-2025 en ningún paso.
