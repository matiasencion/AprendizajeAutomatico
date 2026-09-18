# Experimento: comparación contra scikit-learn (RandomForest y CategoricalNB)

## 2026-09-16 — ¿Cuánto se pierde/gana con implementaciones de referencia?

La letra exige que el modelo entregado sea **un único árbol de decisión**
(ID3 propio), así que esto no reemplaza esa entrega — es una comparación
de referencia para entender si el techo de rendimiento observado se debe
al enfoque (atributos discretizados + historial de partidos) o a las
implementaciones propias (ID3 y M-estimador). Se probaron:

- `sklearn.ensemble.RandomForestClassifier` (permitido por la letra):
  ensamble de muchos árboles, el candidato con mayor potencial de mejora
  que identificamos en la conversación anterior.
- `sklearn.naive_bayes.CategoricalNB`: la implementación de referencia de
  Naive Bayes categórico (equivalente conceptual del M-estimador propio,
  con suavizado aditivo `alpha` en vez de `m`).

## Metodología (idéntica a la ya usada para ID3/Bayes propios)

- Solo `train` (partidos anteriores a 2024-01-01); el test 2024-2025 no se
  tocó en ningún momento.
- Validación cruzada temporal: 18 folds, validando cada año 2005-2022,
  entrenando con los 5 años anteriores (mismo esquema adoptado en
  `EXPERIMENTO_VENTANAS.md`).
- Búsqueda aleatoria de 500 candidatos (semilla 42) sobre los mismos
  rangos de márgenes de discretización que ya usan los notebooks, más los
  hiperparámetros propios de cada modelo, **más los umbrales de
  `match_evenness`** (esta vez sí incluidos en la búsqueda, a diferencia
  de la corrida de `validation_years` anterior).
- Selección por F1 macro medio anual, igual que los notebooks propios.
- Mismo procedimiento de reporte: reentrenar el ganador en cada fold y
  concatenar las predicciones para el reporte por clase agrupado.

## Resultados: los cuatro clasificadores, mismo protocolo

| Clasificador | F1 macro (media anual) | Accuracy | Balanced accuracy | F1 Empate | Recall Empate |
|---|---|---|---|---|---|
| Árbol ID3 propio | 40,04% | 42,05% | 40,26% | 0,29 | 0,30 |
| Bayes M-estimador propio | 42,70% | 45,51% | 43,24% | 0,27 | 0,25 |
| **RandomForestClassifier (sklearn)** | **42,51%** | 43,72% | **42,94%** | **0,32** | **0,35** |
| CategoricalNB (sklearn) | 42,26% | 44,42% | 42,57% | 0,28 | 0,28 |

(F1/recall de Empate tomados del reporte de validación agrupada, 4737
partidos, reentrenando el ganador en cada uno de los 18 folds.)

## Lectura de los resultados

**Random Forest sí mejora sobre el árbol único, y justo donde más
importaba.** +2,5 puntos de F1 macro y +2,7 de balanced accuracy sobre el
ID3 propio, pero el salto más interesante es en la clase que más costaba:
el recall de Empate pasa de 0,30 a 0,35 y su F1 de 0,29 a 0,32 — la mejor
detección de empates de los cuatro clasificadores. El costo es que baja
el recall de Local (de 0,55 a 0,45): el mejor candidato encontrado usa
`class_weight="balanced_subsample"`, que fuerza al modelo a prestarle más
atención a las clases minoritarias (Empate, Visitante) a costa de la
mayoritaria (Local). Confirma la hipótesis de la conversación anterior:
ensamblar reduce la varianza de un árbol único y permite aprovechar mejor
la misma información.

**CategoricalNB (sklearn) prácticamente empata con el M-estimador
propio** (42,26% vs 42,70% de F1 macro, diferencia menor al error
estándar de ambos, ~0,008). Esto es una buena noticia para la
implementación propia: no le está dejando rendimiento importante sobre la
mesa frente a la versión de referencia de la librería — confirma que el
M-estimador implementado a mano está bien hecho, no que el problema esté
en el código sino en la información disponible.

**Ningún modelo, ni siquiera con la implementación de referencia de
sklearn, superó ampliamente a los propios.** El techo sigue estando
alrededor de 40-43% de F1 macro con este conjunto de atributos. Esto
refuerza lo que se viene sosteniendo en la conversación: el límite es la
información (resultados históricos, sin cuotas ni datos de plantilla), no
la calidad de la implementación de los algoritmos.

## Recomendación

Si la letra permite mencionar una comparación de referencia (sin que sea
el modelo entregado), **Random Forest es el punto de referencia más útil
para justificar decisiones de diseño del árbol entregado** — por ejemplo,
sirve para argumentar en el informe que la varianza de un árbol único es
una limitación conocida y medida, y no una omisión. Los hiperparámetros
ganadores de Random Forest (`class_weight="balanced_subsample"`,
`max_depth=10`, `min_samples_leaf=40`) también son una pista útil para el
árbol propio: capar la profundidad y exigir más ejemplos por hoja son
formas de reducir sobreajuste en un único árbol, en la misma dirección
que ya hace `min_info_gain`.

## Reproducibilidad

Script: `sklearn_experiment.py`. Resultados completos (top-10 candidatos,
hiperparámetros ganadores completos, reportes por clase) en
`resultados/experimento_sklearn_resultados.json`. No se usó el conjunto de test
2024-2025 en ningún paso.
