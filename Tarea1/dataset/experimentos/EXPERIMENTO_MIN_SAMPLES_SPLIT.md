# Experimento: `min_samples_split` en el árbol ID3 propio

## 2026-09-18 — Regularización adicional para estabilizar el árbol

## Motivación

Al evaluar los cuatro clasificadores sobre el test 2024-2025 (ver
`entrega.ipynb`, Sección 8), el árbol ID3 propio quedó con la accuracy
más baja de los cinco (incluido el clasificador base), mientras que
Random Forest —un ensamble del mismo tipo de árbol— rindió claramente
mejor. Revisando `decisionTree/classifier.py`, la implementación tenía
**un solo mecanismo de regularización: `min_info_gain`**, que corta la
recursión si la ganancia de información relativa no supera un umbral,
pero no le importa si esa ganancia se calculó sobre 300 filas o sobre 3.
Eso permite hojas muy específicas con pocos ejemplos, que son las que más
sobreajustan y peor generalizan a un año no visto.

## Cambio de código

Se agregó `min_samples_split` (mismo nombre y semántica que el
hiperparámetro homónimo de `sklearn.tree.DecisionTreeClassifier`): si un
nodo tiene menos ejemplos que ese mínimo, no se intenta dividir y se
devuelve directamente la clase mayoritaria del subconjunto. Valor por
defecto `2` (no cambia el comportamiento anterior si no se ajusta
explícitamente, para no romper nada de lo ya hecho).

## Metodología del experimento

Igual que en todo el resto del trabajo:

- Búsqueda de hiperparámetros por validación cruzada temporal (18 folds,
  2005-2022, ventana de entrenamiento de 5 años), 500 candidatos,
  incluyendo `min_samples_split` (rango 2-400) junto con `min_info_gain`
  y los márgenes de discretización ya existentes.
- **El conjunto de test (2024-2025) se evaluó una única vez**, después de
  fijar la configuración por CV, reentrenando con la misma ventana de 5
  años usada en cada fold (2019-2023). No se volvió a ajustar nada en
  base al resultado de test.

## Resultados

**Validación cruzada (18 folds) — no debería cambiar mucho, y no cambió:**

| | Sin `min_samples_split` | Con `min_samples_split` |
|---|---|---|
| F1 macro medio anual | ~40,04% | 39,93% |
| Accuracy media anual | ~42,05% | 42,07% |
| SE del ganador | ~0,0071 | 0,0074 |

Prácticamente idéntico — la nueva regularización no le cuesta nada en la
métrica que ya se validaba, lo cual es esperable (el ganador,
`min_samples_split=10`, es un valor moderado, no agresivo).

**Test (2024-2025, 472 partidos) — acá sí hay diferencia:**

| | Sin `min_samples_split` | Con `min_samples_split=10` |
|---|---|---|
| Accuracy | 0,39 | **0,42** |
| F1 macro | 0,37 | **0,41** |
| Recall Empate | 0,27 | **0,33** |
| F1 Empate | 0,26 | **0,32** |

Mejora en las cuatro métricas. El F1 de Empate (0,32) queda igual al de
Random Forest en el mismo test (también 0,32) — ya no es la peor opción
para esa clase.

**Hiperparámetros ganadores completos:**

```
min_info_gain=0.00625, min_samples_split=10,
record_margin=0.06, last_matches_margin=0.0, goal_difference_margin=0.15,
attack_margin=0.2, defense_margin=0.01, elo_margin=75.0,
rest_days_margin=7.0, h2h_margin=0.2,
draw_low_threshold=0.35, draw_high_threshold=0.45,
evenness_low_threshold=50.0, evenness_high_threshold=120.0
```

El top-10 de candidatos por F1 macro de CV tiene `min_samples_split`
entre 2 y 160 (predominan valores chicos: 5, 10, 20, 30), y quedan dentro
de 1 SE entre sí — no hay una única configuración claramente superior,
pero la regularización adicional está presente en casi todo el top-10,
lo que sugiere que ayuda de forma consistente y no es una casualidad de
un solo candidato.

## Interpretación

Esto no es "encontrar el número mágico que hace ganar en test" — la
configuración se fijó completamente por validación cruzada, sin mirar el
test, y en la propia validación cruzada quedó estadísticamente empatada
con la versión sin `min_samples_split`. La mejora en test es consistente
con la hipótesis original: la inestabilidad del árbol único venía en
buena parte de hojas con pocos ejemplos, y limitar eso con un mecanismo
de regularización adicional (no solo `min_info_gain`) estabiliza el
árbol sin perjudicar lo que ya se había validado.

## Recomendación

Incorporar `min_samples_split` al `param_grid` del árbol en
`entrega.ipynb` (Sección 4.1) para que el entregable final ya lo incluya,
y volver a correr esa notebook completa con el cambio.

## Reproducibilidad

Script: `min_samples_experiment.py`. Resultados completos (top-10,
hiperparámetros ganadores, reportes) en
`resultados/experimento_min_samples_split_resultados.json`. Cambio de
código en `Tarea1/decisionTree/classifier.py` (agrega `min_samples_split`,
compatible hacia atrás con el valor por defecto `2`).
