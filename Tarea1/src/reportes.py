"""
Funciones para armar las tablas y graficas de la seccion de comparacion
del notebook de entrega (entrega.ipynb).

Este archivo no entrena ni evalua nada: recibe resultados ya calculados
(diccionarios, listas, arrays) y devuelve figuras de matplotlib o
DataFrames de pandas listos para mostrar/exportar. Se separa del notebook
para poder llamarlo con una sola linea desde cualquier celda y mantener
el notebook enfocado en la narrativa del trabajo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


# Paleta fija por modelo para que el mismo color represente siempre al
# mismo clasificador en todas las graficas del informe.
MODEL_COLORS = {
    "Arbol ID3 (propio)": "#4C72B0",
    "Bayes M-estimador (propio)": "#DD8452",
    "Random Forest (sklearn)": "#55A868",
    "Naive Bayes categorico (sklearn)": "#C44E52",
    "Clasificador base (referencia)": "#8C8C8C",
}


def tabla_resumen_modelos(resultados: dict) -> pd.DataFrame:
    """
    resultados: {nombre_modelo: {"f1_macro": .., "accuracy": .., "balanced_accuracy": ..,
                                   "f1_empate": .., "recall_empate": ..}}
    Devuelve una tabla en porcentaje con el mejor valor de cada columna
    resaltado en negrita (para mostrarla con `.style` en un notebook, o
    exportarla a HTML/LaTeX para el informe).
    """
    df = pd.DataFrame(resultados).T
    df = df[["f1_macro", "accuracy", "balanced_accuracy", "f1_empate", "recall_empate"]]
    df.columns = ["F1 macro", "Accuracy", "Balanced accuracy", "F1 Empate", "Recall Empate"]

    def resaltar_mejor(columna):
        mejor = columna.max()
        return ["font-weight: bold" if valor == mejor else "" for valor in columna]

    formato = {
        "F1 macro": "{:.2%}", "Accuracy": "{:.2%}", "Balanced accuracy": "{:.2%}",
        "F1 Empate": "{:.2%}", "Recall Empate": "{:.2%}",
    }
    return df.style.apply(resaltar_mejor, axis=0).format(formato)


def grafico_comparacion_modelos(
    resultados: dict,
    metricas=("f1_macro", "accuracy", "balanced_accuracy"),
    etiquetas_metricas=("F1 macro", "Accuracy", "Balanced accuracy"),
    titulo="Comparacion de clasificadores (validacion cruzada temporal)",
):
    """Barras agrupadas: un grupo de barras por modelo, una barra por metrica."""
    modelos = list(resultados.keys())
    x = np.arange(len(metricas))
    ancho = 0.8 / len(modelos)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, modelo in enumerate(modelos):
        valores = [resultados[modelo][m] for m in metricas]
        posiciones = x + (i - (len(modelos) - 1) / 2) * ancho
        color = MODEL_COLORS.get(modelo)
        barras = ax.bar(posiciones, valores, width=ancho, label=modelo, color=color)
        ax.bar_label(barras, fmt="%.1f%%", padding=2, fontsize=8, labels=[f"{v*100:.1f}%" for v in valores])

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_metricas)
    ax.set_ylabel("Valor de la metrica")
    ax.set_ylim(0, max(v for r in resultados.values() for v in [r[m] for m in metricas]) * 1.25)
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_f1_por_anio(series_por_modelo: dict, titulo="F1 macro por año de validación"):
    """
    series_por_modelo: {nombre_modelo: {anio: f1_macro, ...}, ...}
    Una linea por modelo, mostrando la variabilidad ano a ano que motiva
    reportar el error estandar en vez de mirar solo el promedio.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for modelo, serie in series_por_modelo.items():
        anios = sorted(serie.keys())
        valores = [serie[a] for a in anios]
        color = MODEL_COLORS.get(modelo)
        ax.plot(anios, valores, marker="o", label=modelo, color=color)

    ax.set_xlabel("Año de validación")
    ax.set_ylabel("F1 macro")
    ax.set_title(titulo)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_sensibilidad(
    valores_x, medias, errores_se, xlabel, titulo, valor_actual=None,
):
    """
    Linea con barras de error (SE) mostrando F1 macro medio anual al
    variar un hiperparametro (window_years, years_limit, matches_limit).
    `valor_actual` (opcional) marca con una linea vertical el valor
    finalmente adoptado.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(valores_x, medias, yerr=errores_se, marker="o", capsize=4, color="#4C72B0")
    if valor_actual is not None:
        ax.axvline(valor_actual, color="#C44E52", linestyle="--", alpha=0.7, label=f"valor adoptado = {valor_actual}")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 macro (media anual) ± SE")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def tabla_sensibilidad(filas: list, columna_valor: str, etiqueta_valor: str) -> pd.DataFrame:
    """
    filas: lista de dicts con al menos {columna_valor, "mean_f1_macro", "se_f1_macro"}
    Se genera una tabla con
    una columna "Dentro de 1 SE del mejor" para dejar explicito cuando los
    candidatos cumplen una regla descriptiva, sin afirmar equivalencia estadistica.
    """
    df = pd.DataFrame(filas)
    mejor = df["mean_f1_macro"].max()
    se_mejor = df.loc[df["mean_f1_macro"].idxmax(), "se_f1_macro"]
    umbral = mejor - se_mejor
    df["Dentro de 1 SE del mejor"] = df["mean_f1_macro"] >= umbral
    df = df.rename(columns={
        columna_valor: etiqueta_valor,
        "mean_f1_macro": "F1 macro medio",
        "se_f1_macro": "Error estándar (SE)",
    })
    columnas = [etiqueta_valor, "F1 macro medio", "Error estándar (SE)", "Dentro de 1 SE del mejor"]
    return df[columnas].style.format({"F1 macro medio": "{:.4f}", "Error estándar (SE)": "{:.4f}"})


def grafico_matrices_confusion(y_true, predicciones: dict, labels=("L", "E", "V"), display_labels=("Local", "Empate", "Visitante")):
    """
    predicciones: {nombre_modelo: array_de_predicciones}
    Grilla de matrices de confusion normalizadas (recall por clase real),
    una por modelo, para comparar visualmente donde falla cada uno.
    """
    modelos = list(predicciones.keys())
    n = len(modelos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, modelo in zip(axes, modelos):
        ConfusionMatrixDisplay.from_predictions(
            y_true, predicciones[modelo], labels=list(labels), display_labels=list(display_labels),
            normalize="true", cmap="Blues", values_format=".2f", ax=ax, colorbar=False,
        )
        ax.set_title(modelo, fontsize=10)
    fig.suptitle("Matrices de confusión normalizadas (recall por clase real)")
    fig.tight_layout()
    return fig


def reporte_por_clase(y_true, y_pred, labels=("L", "E", "V"), target_names=("Local", "Empate", "Visitante")) -> str:
    """Atajo para el reporte de texto de sklearn, con los nombres en espanol ya fijados."""
    return classification_report(y_true, y_pred, labels=list(labels), target_names=list(target_names), zero_division=0)


# ---------------------------------------------------------------------
# Sensibilidad a hiperparametros (a partir de cv_results_ de una
# RandomizedSearchCV ya corrida: una fila por candidato probado).
# ---------------------------------------------------------------------

def grafico_mejor_f1_por_hiperparametro(
    cv_results: pd.DataFrame,
    param_col: str,
    xlabel: str,
    titulo: str,
    metrica: str = "mean_test_f1_macro",
    color: str = "#4C72B0",
    selected_value=None,
):
    """Mejor resultado observado para cada valor de un hiperparametro.

    La funcion reutiliza los candidatos evaluados por la busqueda anterior;
    no ajusta modelos nuevos. Cada punto es el maximo de `metrica` entre las
    configuraciones que contienen ese valor. Como los demas hiperparametros
    tambien cambian, la figura compara los mejores candidatos observados y no
    estima el efecto aislado del parametro.
    """
    datos = cv_results[[param_col, metrica]].dropna(subset=[metrica])
    mejores = datos.groupby(param_col, sort=True)[metrica].max()
    valores = list(mejores.index)
    puntajes = mejores.to_numpy()
    x = np.arange(len(valores))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, puntajes, marker="o", color=color, label="mejor candidato por valor")
    for posicion, puntaje in zip(x, puntajes):
        ax.annotate(
            f"{puntaje:.4f}",
            (posicion, puntaje),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    if selected_value is not None:
        selected_x = valores.index(selected_value)
        ax.scatter(
            selected_x,
            puntajes[selected_x],
            marker="*",
            s=170,
            color="#C44E52",
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
            label=f"valor de la configuración seleccionada ({selected_value})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in valores], rotation=45 if len(valores) > 6 else 0, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mejor F1 macro medio anual observado")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig



def grafico_menor_error_por_hiperparametros(paneles):
    """Grafica el menor error observado para valores de varios hiperparametros.

    Cada panel debe indicar ``cv_results``, ``param_col``, ``xlabel``,
    ``titulo``, ``color`` y, opcionalmente, ``selected_value``. La tasa de
    error se calcula como 1 - accuracy de validacion. Se reutilizan los
    candidatos de la busqueda: no se vuelven a ajustar modelos.
    """
    fig, axes = plt.subplots(1, len(paneles), figsize=(11, 4.2), squeeze=False)

    for ax, panel in zip(axes[0], paneles):
        datos = panel["cv_results"][[panel["param_col"], "mean_test_accuracy"]].copy()
        datos = datos.dropna(subset=["mean_test_accuracy"])
        datos["error"] = 1.0 - datos["mean_test_accuracy"]
        menores = datos.groupby(panel["param_col"], sort=True)["error"].min()

        valores = list(menores.index)
        errores = menores.to_numpy()
        x = np.arange(len(valores))
        ax.plot(x, errores, marker="o", color=panel["color"])

        for posicion, error in zip(x, errores):
            ax.annotate(
                f"{error:.4f}",
                (posicion, error),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        selected_value = panel.get("selected_value")
        if selected_value is not None:
            selected_x = valores.index(selected_value)
            ax.axvline(
                selected_x,
                color="#C44E52",
                linestyle="--",
                linewidth=1.2,
                label=f"valor seleccionado ({selected_value})",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(valor) for valor in valores],
            rotation=45 if len(valores) > 6 else 0,
            ha="right" if len(valores) > 6 else "center",
        )
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Menor error medio anual observado")
        ax.set_title(panel["titulo"])
        ax.grid(alpha=0.3)
        if selected_value is not None:
            ax.legend(fontsize=8)

    fig.tight_layout()
    return fig

def grafico_ventanas(resultados, window_years=5):
    """Grafica resultados calculados por evaluacion.evaluar_ventanas."""
    dimensions = [("window_years", window_years), ("years_limit", 1), ("matches_limit", 5)]
    colors = {"bayes": MODEL_COLORS["Bayes M-estimador (propio)"],
              "tree": MODEL_COLORS["Arbol ID3 (propio)"]}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (dimension, current) in zip(axes, dimensions):
        for model, rows in resultados[dimension].items():
            ax.errorbar([row[dimension] for row in rows],
                        [row["mean_f1_macro"] for row in rows],
                        yerr=[row["se_f1_macro"] for row in rows],
                        marker="o", capsize=4, label=model, color=colors.get(model))
        ax.axvline(current, color="gray", linestyle="--", alpha=0.6)
        ax.set(xlabel=dimension, ylabel="F1 macro ± SE descriptivo", title=f"Sensibilidad a {dimension}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_importancia_hiperparametros(
    cv_results: pd.DataFrame,
    params: dict,
    titulo: str,
    metrica: str = "mean_test_f1_macro",
):
    """
    params: {columna_param: etiqueta_legible}. Para cada hiperparametro,
    agrupa los candidatos por su valor, promedia `metrica` en cada grupo,
    y mide el rango (maximo-minimo) entre esos promedios -- una medida
    simple de "cuanto mueve la aguja" ese hiperparametro dentro de la
    busqueda. Sirve para comparar de un vistazo muchos hiperparametros
    (por ejemplo, los margenes de discretizacion) sin tener que graficar
    cada uno por separado. Complementa las graficas del mejor F1 observado
    para los 2-3 hiperparametros que interesa ver en detalle.
    """
    filas = []
    for columna, etiqueta in params.items():
        promedios = cv_results.groupby(columna)[metrica].mean()
        filas.append((etiqueta, promedios.max() - promedios.min()))
    filas.sort(key=lambda f: f[1])

    etiquetas = [f[0] for f in filas]
    rangos = [f[1] for f in filas]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(etiquetas) + 1.5))
    ax.barh(etiquetas, rangos, color="#55A868")
    ax.set_xlabel("Rango de F1 macro entre valores probados")
    ax.set_title(titulo)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
