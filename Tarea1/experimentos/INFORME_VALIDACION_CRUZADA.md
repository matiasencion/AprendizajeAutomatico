# Validación cruzada temporal

Se valida cada año de 2005 a 2023 inclusive mediante 19 folds, con entrenamiento sobre los cinco años anteriores. Para validar 2023 se utiliza 2018–2022. Los partidos desde 2024 no participan en esta selección.

Los hiperparámetros se seleccionan por el mayor F1 macro medio anual entre 500 candidatos reproducibles por modelo. También se calculan accuracy y balanced accuracy. Se distingue la media anual del F1 agrupado: promediar métricas no equivale a recalcularlas sobre todas las predicciones.

La validación se reutiliza durante la selección, por lo que no se interpreta el desempeño del ganador como una estimación independiente de ese proceso. Se informa la variación anual para mostrar la estabilidad de las configuraciones.

Se utiliza `s / sqrt(n)` como error estándar descriptivo, donde `s` es la desviación muestral entre años. Las ventanas comparten historia y puede existir dependencia temporal: no se interpreta la banda de un error estándar como prueba de equivalencia o significación. La selección implementada es por el máximo de F1; no se aplica una regla de selección de 1-SE.

Para el test se ajusta una sola vez cada modelo con 2019–2023 y se mantiene fijo durante 2024 y el tramo disponible de 2025. La CV anual orienta la selección; la evaluación final mide el comportamiento de ese modelo fijo durante un período más largo. Los atributos se actualizan con resultados anteriores a cada partido.

La corrida anterior utilizaba 18 folds, hasta 2022. Sus resultados se conservan como antecedentes en `resultados/entrega_corrida_anterior.ipynb`. Sus promedios y errores estándar no se trasladan al protocolo de 19 folds sin recalcularlos.
