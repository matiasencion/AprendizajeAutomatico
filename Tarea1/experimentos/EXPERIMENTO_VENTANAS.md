# Sensibilidad a las ventanas temporales

Se distinguen tres parámetros: `window_years` controla cuántos años se utilizan para entrenar cada fold; `years_limit` controla el historial usado para construir atributos; `matches_limit` limita los últimos partidos usados para la forma reciente. Se establece como referencia cinco años de entrenamiento, un año de atributos y cinco partidos recientes.

En la entrega las variantes se calculan durante la ejecución, con los modelos seleccionados en esa corrida y sin leer tablas de resultados. Los hiperparámetros se mantienen fijos y se modifica una ventana por vez. Al variar la historia se reconstruyen los atributos solo sobre desarrollo. Las gráficas se generan en `../src/reportes.py`.

## Antecedentes: 18 folds, 2005–2022

| Ventana de entrenamiento | F1 macro Bayes | F1 macro ID3 |
| --- | ---: | ---: |
| 3 | 41,13% | 38,61% |
| 5 | 41,99% | 39,93% |
| 7 | 41,90% | 39,92% |
| 10 | 42,01% | 38,36% |

Con uno, dos y tres años de atributos se obtuvieron respectivamente 41,99%, 42,44% y 42,61% para Bayes; 39,93%, 38,28% y 38,68% para ID3. Con tres, cinco y ocho partidos recientes se obtuvieron 41,36%, 41,99% y 41,67% para Bayes; 39,39%, 39,93% y 39,38% para ID3.

Estos antecedentes respaldan cinco años como una opción razonable entre las probadas, pero no demuestran un óptimo global. Tampoco se interpretan diferencias dentro de un error estándar como equivalencia estadística. Al cambiar la ventana de atributos se mantienen los márgenes fijos, por lo que se describe sensibilidad y no una búsqueda completa para cada variante.

La versión actual incluye 2023 y regularización por hoja. Los resultados se obtienen ejecutando el notebook; los JSON históricos conservan su protocolo original. El script independiente permite repetir la sensibilidad con sus parámetros de referencia explícitos.
