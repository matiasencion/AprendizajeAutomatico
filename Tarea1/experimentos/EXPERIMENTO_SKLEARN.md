# Comparación con implementaciones de referencia

Se comparan el ID3 y el M-estimador propios con Random Forest y Naive Bayes categórico de sklearn. Se mantienen los mismos partidos de validación y las mismas variables de entrada, aunque cada modelo selecciona sus propios márgenes y parámetros.

## Corrida anterior: 18 folds, 2005–2022

| Modelo | F1 macro medio anual | F1 macro agrupado | Accuracy agrupada |
| --- | ---: | ---: | ---: |
| ID3 | 39,93% | 40,12% | 42,01% |
| Bayes propio | 42,30% | 42,86% | 44,71% |
| Random Forest | 42,51% | 42,76% | 43,70% |
| Bayes sklearn | 42,26% | 42,76% | 44,63% |

Random Forest obtuvo el mayor F1 medio anual, mientras que Bayes propio obtuvo el mayor F1 agrupado. Se distinguen ambas formas de agregar las predicciones. La diferencia entre los dos Bayes fue pequeña en validación; esto muestra competitividad, pero no sustituye pruebas de corrección de la implementación.

El bosque usa árboles binarios de sklearn, no copias del ID3 implementado aquí. La comparación cambia varios mecanismos a la vez y no permite atribuir toda la mejora exclusivamente al ensamble. Tampoco se concluye que estas búsquedas establezcan un límite de rendimiento del dataset.

La entrega actual utiliza 19 folds, hasta 2023, e incorpora `min_samples_leaf` al ID3. Las cifras de esta tabla pertenecen a la corrida archivada en `resultados/entrega_corrida_anterior.ipynb` y no describen automáticamente la nueva ejecución.
