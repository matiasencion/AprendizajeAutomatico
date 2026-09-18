# Estructura de esta carpeta

## Para corregir / entrega final

- **`entrega.ipynb`** — el notebook único de entrega. Carga los atributos,
  hace la validación cruzada temporal para elegir hiperparámetros de los
  cuatro clasificadores (Árbol ID3 propio, Bayes M-estimador propio,
  Random Forest y Naive Bayes categórico de scikit-learn) más el
  clasificador base de referencia, los compara con tablas y gráficas, y
  cierra con la evaluación final sobre el conjunto de test 2024-2025.
- **`load.py`** / **`pipeline.py`** — librería compartida: carga del
  dataset, cálculo de atributos históricos y pipeline de discretización.
  Los importa `entrega.ipynb` y todos los notebooks de `desarrollo/`.
- **`reportes.py`** — funciones de gráficas y tablas que usa la Sección 6
  y 8 de `entrega.ipynb` (no entrena ni evalúa nada por sí solo).
- **`futbol_uruguayo.csv`** — dataset crudo.
- **`CORRECCIONES_DATASET.md`** — registro de correcciones puntuales
  aplicadas al CSV.

## `desarrollo/`

Notebooks individuales por modelo, usados durante el desarrollo para
buscar hiperparámetros e ir registrando resultados (`baseNotebook`,
`bayesNotebook`, `treeNotebook`, `nbNotebook`, `rfNotebook`). No son el
entregable — `entrega.ipynb` consolida y reproduce lo mismo en un solo
notebook. Se mantienen como referencia/trazabilidad del proceso.

## `experimentos/`

Documentación y código de los experimentos metodológicos que sustentan
varias decisiones del trabajo (no hace falta ejecutarlos para correr
`entrega.ipynb`, que ya usa sus resultados cacheados en `resultados/`):

- `EXPERIMENTO_PARIDAD.md` + `evenness_experiment.py` — efecto del
  atributo `match_evenness`.
- `EXPERIMENTO_VENTANAS.md` + `window_experiment.py` — sensibilidad a
  `window_years`, `years_limit` y `matches_limit`.
- `EXPERIMENTO_SKLEARN.md` + `sklearn_experiment.py` — comparación contra
  las implementaciones de referencia de scikit-learn.
- `INFORME_VALIDACION_CRUZADA.md` — por qué se extendió la validación
  cruzada de 10 a 18 años, con las fórmulas de error estándar (texto
  pensado para copiar directo al informe).
- `resultados/` — JSON con los resultados detallados de cada experimento,
  más los CSV de registro histórico de búsquedas (`experimentos_*.csv`,
  generados automáticamente al correr los notebooks de `desarrollo/`).

## Notas

- El conjunto de test (partidos desde el 1° de enero de 2024) sólo se
  usa en la Sección 8 de `entrega.ipynb`. En todo lo demás —incluidos
  todos los experimentos de `experimentos/`— la evaluación es por
  validación cruzada temporal sobre `train`.
- Los notebooks de `desarrollo/` importan `load`/`pipeline` desde `..`
  (un nivel arriba) y, cuando corresponde, los paquetes propios
  (`decisionTree`, `naiveBayes`, `baseClassifier`) desde `../..`.
