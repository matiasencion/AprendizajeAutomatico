# Aprendizaje automático

Se comparan implementaciones propias de ID3 y Naive Bayes con modelos de scikit-learn para predecir resultados del fútbol uruguayo.

La entrega principal es **`Tarea1/notebooks/entrega.ipynb`**, además se encuentra una copia ya ejecutado en **`Tarea1/notebooks/entrega_corrida_anterior.ipynb`**. Su ejecución requiere el CSV de `Tarea1/datos/`, los módulos de `Tarea1/src/` y las bibliotecas indicadas en `requirements.txt`. No utiliza los scripts de experimentación, sus JSON ni otros notebooks.

Las dependencias se instalan desde la raíz del repositorio:

```sh
python -m pip install -r requirements.txt
cd Tarea1/notebooks
jupyter lab entrega.ipynb
```

Se utilizan rutas relativas fijas desde la carpeta del notebook. Las métricas y gráficas se calculan durante la ejecución, sin resultados precalculados. La función de cada archivo y el conjunto mínimo de entrega se documentan en `Tarea1/README.md`.

Pruebas desde la raíz: `python -m unittest discover -s Tarea1/tests -v`.
