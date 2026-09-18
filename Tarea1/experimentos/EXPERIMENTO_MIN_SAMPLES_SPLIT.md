# Regularización por tamaño mínimo de hoja

Se utiliza `min_samples_leaf` como única restricción explícita de tamaño en el ID3. Para cada atributo candidato se comprueba que todas sus ramas observadas tengan al menos ese número de ejemplos. Se descartan las divisiones inválidas y se selecciona la de mayor ganancia entre las válidas. Si ninguna cumple, se genera una hoja con la clase mayoritaria del nodo.

No se requiere además un umbral independiente para el tamaño del nodo padre: si no hay ejemplos suficientes para dos ramas válidas, el nodo no se divide. Una raíz con menos ejemplos que el mínimo también permanece como hoja.

Se mantiene `min_info_gain` como condición adicional basada en información. La ganancia se calcula a partir de proporciones; duplicar los ejemplos no cambia su valor.

Se evalúa `min_samples_leaf` en `[1, 5, 10, 20, 30, 50]`, junto con la ganancia y los márgenes. La selección se realiza por F1 macro medio anual sobre 19 folds (2005–2023). Se ajusta un único modelo final con 2019–2023 para evaluar todo el test.

Las cifras de la corrida archivada describen su configuración anterior. Los resultados del árbol revisado se calculan al ejecutar la entrega; no se trasladan métricas antiguas al nuevo espacio de búsqueda.
