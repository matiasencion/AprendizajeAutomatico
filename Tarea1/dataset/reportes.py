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
        "F1 Empate": "{:.2f}", "Recall Empate": "{:.2f}",
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
    (el formato que ya devuelve window_experiment.py). Arma una tabla con
    una columna "Dentro de 1 SE del mejor" para dejar explicito cuando los
    candidatos son estadisticamente equivalentes.
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
