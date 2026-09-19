# Tarea 1 — Predicción de resultados del fútbol uruguayo

## Archivos necesarios para ejecutar la entrega

La entrega es independiente de las herramientas de desarrollo: no importa scripts de `experimentos/`, no carga sus JSON y no ejecuta otros notebooks. Sí utiliza el dataset original y el código modular de `src/`.

El conjunto mínimo, conservando las rutas, es:

```text
requirements.txt
Tarea1/
├── notebooks/
│   └── entrega.ipynb
├── datos/
│   └── futbol_uruguayo.csv
    └── CORRECCIONES_DATASET.md
└── src/
    ├── load.py
    ├── pipeline.py
    ├── evaluacion.py
    ├── reportes.py
    ├── decisionTree/
    │   ├── classifier.py
    │   └── tree.py
    ├── naiveBayes/
    │   └── bayes.py
    └── baseClassifier/
        └── base_classifier.py
```

## Flujo de ejecución

1. `load.py` lee y limpia el CSV original, y construye los atributos históricos.
2. `evaluacion.py` define los folds temporales; `entrega.ipynb` configura y ejecuta las búsquedas de hiperparámetros.
3. `pipeline.py` transforma los atributos y los entrega al clasificador correspondiente.
4. Se generan predicciones y métricas de validación con `evaluacion.py`.
5. La sensibilidad a las ventanas se calcula con `evaluar_ventanas`, dentro de `evaluacion.py`, a partir de los modelos seleccionados en la propia ejecución.
6. `reportes.py` recibe las métricas y predicciones en memoria para construir tablas y figuras.
7. Se ajusta un único modelo por clasificador con 2019–2023 y se evalúa sobre todo el test disponible.

La validación comprende 19 folds anuales, de 2005 a 2023. Se evalúan 500 candidatos por modelo y se selecciona por F1 macro medio anual. El árbol propio utiliza `min_samples_leaf` y `min_info_gain`; no utiliza un umbral adicional de tamaño mínimo del nodo padre. El modelo final permanece fijo durante el test, aunque sus atributos se actualizan con los partidos anteriores a cada encuentro.

## Módulos utilizados por la entrega

| Archivo | Responsabilidad |
| --- | --- |
| `src/load.py` | Lectura del CSV; limpieza de tipos, orden y duplicados; generación de la etiqueta L/E/V y atributos históricos, de goles, forma reciente, enfrentamientos directos, descanso, experiencia y Elo. `load_base_dataset` omite la ingeniería de atributos; `load_dataset` la incluye. |
| `src/pipeline.py` | Listas y orden de atributos, discretizadores de diferencias, tendencia al empate y paridad; codificación ordinal y composición del preprocesamiento con el modelo. |
| `src/evaluacion.py` | Construcción de folds, resumen de métricas, ajuste y predicción por fold, ajuste final con ventana fija y cálculo de sensibilidad a ventanas. No requiere archivos de resultados. |
| `src/reportes.py` | Tablas resumen, comparación de modelos, evolución anual, matrices de confusión y gráficas de sensibilidad. No entrena modelos ni lee datos o resultados externos. |
| `src/decisionTree/classifier.py` | Implementación de ID3: entropía, ganancia de información, selección de divisiones válidas, regularización por soporte de hoja, entrenamiento y predicción compatibles con scikit-learn. |
| `src/decisionTree/tree.py` | Representación de cada nodo y sus ramas, junto con la impresión textual del árbol. Se utiliza desde `classifier.py`. |
| `src/naiveBayes/bayes.py` | Naive Bayes categórico propio con suavizado mediante un M-estimador y acumulación de log-probabilidades. |
| `src/baseClassifier/base_classifier.py` | Regla de referencia basada en la comparación de tasas históricas de victorias de ambos equipos, con historial anterior a la fecha del partido. |

Random Forest y CategoricalNB se importan directamente de scikit-learn; no cuentan con una implementación local adicional.

## Ejecución

Con las dependencias instaladas, el notebook debe ejecutarse desde `Tarea1/notebooks/`:

```sh
cd Tarea1/notebooks
jupyter lab entrega.ipynb
```

La ruta del dataset es `../datos/futbol_uruguayo.csv` y los imports locales provienen de `../src`. No se realiza una búsqueda automática de la raíz del proyecto. La búsqueda completa y la reconstrucción de atributos pueden requerir un tiempo considerable.
