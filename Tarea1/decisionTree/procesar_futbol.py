"""Preprocesamiento reproducible del dataset de futbol uruguayo.

El script construye atributos disponibles antes de cada partido, separa los
datos cronologicamente y guarda tres archivos CSV:

* dataset_procesado.csv: todos los partidos con atributos historicos.
* entrenamiento_hasta_2023.csv: partidos jugados hasta 2023 inclusive.
* evaluacion_2024_2025.csv: partidos jugados durante 2024 y 2025.

Ejemplo de uso:

    python procesar_futbol.py \
        --entrada /ruta/futbol_uruguayo.zip \
        --salida /ruta/datos_procesados
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import OrdinalEncoder


COLUMNAS_CATEGORICAS = [
    "ventaja_historica",
    "ventaja_forma",
    "ataque_local",
    "defensa_local",
    "ataque_visitante",
    "defensa_visitante",
    "ventaja_localia",
    "experiencia_local",
    "experiencia_visitante",
]

COLUMNAS_BINARIAS = [
    "historial_suficiente",
    "localia_suficiente",
]

CATEGORIAS_ORDINALES = [
    ["visitante", "parejo", "local"],
    ["visitante", "parejo", "local"],
    ["bajo", "medio", "alto"],
    ["bajo", "medio", "alto"],
    ["bajo", "medio", "alto"],
    ["bajo", "medio", "alto"],
    ["visitante", "parejo", "local"],
    ["ninguna", "poca", "media", "alta"],
    ["ninguna", "poca", "media", "alta"],
]


def cargar_y_limpiar(ruta_entrada: str | Path) -> pd.DataFrame:
    """Carga el CSV o ZIP, elimina duplicados y crea la clase ganador."""
    ruta_entrada = Path(ruta_entrada)
    compresion = "zip" if ruta_entrada.suffix.lower() == ".zip" else "infer"
    df = pd.read_csv(ruta_entrada, compression=compresion)

    columnas_requeridas = {"home", "away", "date", "gh", "ga"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {sorted(faltantes)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["gh"] = pd.to_numeric(df["gh"], errors="raise").astype(int)
    df["ga"] = pd.to_numeric(df["ga"], errors="raise").astype(int)

    df = (
        df.drop_duplicates()
        .sort_values(["date", "home", "away"], kind="stable")
        .reset_index(drop=True)
    )

    df["ganador"] = np.select(
        [df["gh"] > df["ga"], df["gh"] < df["ga"]],
        ["L", "V"],
        default="E",
    )
    return df


def nivel_goles(promedio: float) -> str:
    """Discretiza un promedio de goles en tres niveles."""
    if promedio < 1.0:
        return "bajo"
    if promedio < 2.0:
        return "medio"
    return "alto"


def nivel_experiencia(cantidad: int) -> str:
    """Discretiza la cantidad de antecedentes recientes de un equipo."""
    if cantidad == 0:
        return "ninguna"
    if cantidad < 5:
        return "poca"
    if cantidad < 20:
        return "media"
    return "alta"


def comparar(valor_local: float, valor_visitante: float, margen: float) -> str:
    """Indica que lado presenta una ventaja mayor que el margen dado."""
    diferencia = valor_local - valor_visitante
    if diferencia > margen:
        return "local"
    if diferencia < -margen:
        return "visitante"
    return "parejo"


def construir_dataset_modelo(
    df: pd.DataFrame,
    anios_historial: int = 10,
    partidos_forma: int = 5,
    minimo_localia: int = 5,
) -> pd.DataFrame:
    """Crea atributos historicos sin incluir el partido actual ni el futuro.

    Los encuentros de una misma fecha se procesan juntos: primero se crean sus
    atributos y luego se actualizan los historiales. De esa forma, un resultado
    del mismo dia tampoco puede filtrarse hacia otra instancia.
    """
    historial: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    nuevas_filas: list[dict[str, Any]] = []

    def obtener_historial(equipo: str, fecha: pd.Timestamp) -> list[dict[str, Any]]:
        registros = historial[equipo]
        limite = fecha - pd.DateOffset(years=anios_historial)
        while registros and registros[0]["date"] < limite:
            registros.popleft()
        return list(registros)

    for fecha, partidos_fecha in df.groupby("date", sort=True):
        partidos_pendientes = []

        for partido in partidos_fecha.itertuples(index=False):
            historial_local = obtener_historial(partido.home, fecha)
            historial_visitante = obtener_historial(partido.away, fecha)

            ultimos_local = historial_local[-partidos_forma:]
            ultimos_visitante = historial_visitante[-partidos_forma:]
            historial_suficiente = (
                len(ultimos_local) == partidos_forma
                and len(ultimos_visitante) == partidos_forma
            )

            if historial_suficiente:
                tasa_local = float(np.mean([p["win"] for p in historial_local]))
                tasa_visitante = float(
                    np.mean([p["win"] for p in historial_visitante])
                )
                ventaja_historica = comparar(
                    tasa_local, tasa_visitante, margen=0.05
                )

                forma_local = float(
                    np.mean([p["points"] for p in ultimos_local]) / 3.0
                )
                forma_visitante = float(
                    np.mean([p["points"] for p in ultimos_visitante]) / 3.0
                )
                ventaja_forma = comparar(
                    forma_local, forma_visitante, margen=0.10
                )

                ataque_local = nivel_goles(
                    float(np.mean([p["gf"] for p in ultimos_local]))
                )
                defensa_local = nivel_goles(
                    float(np.mean([p["ga"] for p in ultimos_local]))
                )
                ataque_visitante = nivel_goles(
                    float(np.mean([p["gf"] for p in ultimos_visitante]))
                )
                defensa_visitante = nivel_goles(
                    float(np.mean([p["ga"] for p in ultimos_visitante]))
                )
            else:
                # Categorias neutrales acompanadas por el indicador de falta de datos.
                ventaja_historica = "parejo"
                ventaja_forma = "parejo"
                ataque_local = "medio"
                defensa_local = "medio"
                ataque_visitante = "medio"
                defensa_visitante = "medio"

            partidos_local_en_casa = [
                p for p in historial_local if p["venue"] == "home"
            ]
            partidos_visitante_fuera = [
                p for p in historial_visitante if p["venue"] == "away"
            ]
            localia_suficiente = (
                len(partidos_local_en_casa) >= minimo_localia
                and len(partidos_visitante_fuera) >= minimo_localia
            )

            if localia_suficiente:
                tasa_local_casa = float(
                    np.mean([p["win"] for p in partidos_local_en_casa])
                )
                tasa_visitante_fuera = float(
                    np.mean([p["win"] for p in partidos_visitante_fuera])
                )
                ventaja_localia = comparar(
                    tasa_local_casa, tasa_visitante_fuera, margen=0.05
                )
            else:
                ventaja_localia = "parejo"

            nuevas_filas.append(
                {
                    "date": fecha,
                    "home": partido.home,
                    "away": partido.away,
                    "ventaja_historica": ventaja_historica,
                    "ventaja_forma": ventaja_forma,
                    "ataque_local": ataque_local,
                    "defensa_local": defensa_local,
                    "ataque_visitante": ataque_visitante,
                    "defensa_visitante": defensa_visitante,
                    "ventaja_localia": ventaja_localia,
                    "experiencia_local": nivel_experiencia(len(historial_local)),
                    "experiencia_visitante": nivel_experiencia(
                        len(historial_visitante)
                    ),
                    "historial_suficiente": int(historial_suficiente),
                    "localia_suficiente": int(localia_suficiente),
                    "ganador": partido.ganador,
                }
            )
            partidos_pendientes.append(partido)

        # Actualizar solo despues de construir todas las instancias de la fecha.
        for partido in partidos_pendientes:
            empate = partido.gh == partido.ga

            historial[partido.home].append(
                {
                    "date": fecha,
                    "venue": "home",
                    "win": int(partido.gh > partido.ga),
                    "points": 1 if empate else (3 if partido.gh > partido.ga else 0),
                    "gf": partido.gh,
                    "ga": partido.ga,
                }
            )
            historial[partido.away].append(
                {
                    "date": fecha,
                    "venue": "away",
                    "win": int(partido.ga > partido.gh),
                    "points": 1 if empate else (3 if partido.ga > partido.gh else 0),
                    "gf": partido.ga,
                    "ga": partido.gh,
                }
            )

    return pd.DataFrame(nuevas_filas)


def separar_temporalmente(
    df_modelo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento hasta 2023 y evaluacion durante 2024-2025."""
    entrenamiento = df_modelo[df_modelo["date"].dt.year <= 2023].copy()
    evaluacion = df_modelo[
        df_modelo["date"].dt.year.isin([2024, 2025])
    ].copy()
    return entrenamiento, evaluacion


def codificar_datos(
    entrenamiento: pd.DataFrame,
    evaluacion: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    OrdinalEncoder,
    preprocessing.LabelEncoder,
]:
    """Codifica entradas y objetivo, ajustando codificadores solo con train."""
    encoder_atributos = OrdinalEncoder(
        categories=CATEGORIAS_ORDINALES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )

    x_train = pd.DataFrame(
        encoder_atributos.fit_transform(entrenamiento[COLUMNAS_CATEGORICAS]),
        columns=COLUMNAS_CATEGORICAS,
        index=entrenamiento.index,
    )
    x_test = pd.DataFrame(
        encoder_atributos.transform(evaluacion[COLUMNAS_CATEGORICAS]),
        columns=COLUMNAS_CATEGORICAS,
        index=evaluacion.index,
    )
    x_train[COLUMNAS_BINARIAS] = entrenamiento[COLUMNAS_BINARIAS]
    x_test[COLUMNAS_BINARIAS] = evaluacion[COLUMNAS_BINARIAS]

    encoder_objetivo = preprocessing.LabelEncoder()
    y_train = encoder_objetivo.fit_transform(entrenamiento["ganador"])
    y_test = encoder_objetivo.transform(evaluacion["ganador"])

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        encoder_atributos,
        encoder_objetivo,
    )


def guardar_resultados(
    df_modelo: pd.DataFrame,
    entrenamiento: pd.DataFrame,
    evaluacion: pd.DataFrame,
    directorio_salida: str | Path,
) -> None:
    """Guarda las versiones legibles y codificadas del dataset."""
    directorio = Path(directorio_salida)
    directorio.mkdir(parents=True, exist_ok=True)

    df_modelo.to_csv(directorio / "dataset_procesado.csv", index=False)
    entrenamiento.to_csv(
        directorio / "entrenamiento_hasta_2023.csv", index=False
    )
    evaluacion.to_csv(directorio / "evaluacion_2024_2025.csv", index=False)

    x_train, y_train, x_test, y_test, _, encoder_objetivo = codificar_datos(
        entrenamiento, evaluacion
    )

    train_codificado = x_train.copy()
    train_codificado["ganador"] = y_train
    test_codificado = x_test.copy()
    test_codificado["ganador"] = y_test

    train_codificado.to_csv(
        directorio / "entrenamiento_codificado.csv", index=False
    )
    test_codificado.to_csv(
        directorio / "evaluacion_codificada.csv", index=False
    )

    clases = pd.DataFrame(
        {
            "ganador_original": encoder_objetivo.classes_,
            "ganador_codificado": encoder_objetivo.transform(
                encoder_objetivo.classes_
            ),
        }
    )
    clases.to_csv(directorio / "codificacion_ganador.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye el dataset para los clasificadores de la tarea."
    )
    parser.add_argument(
        "--entrada",
        default="/Users/agustinmachado/Downloads/futbol_uruguayo.zip",
        help="Ruta del CSV o ZIP original.",
    )
    parser.add_argument(
        "--salida",
        default="datos_procesados",
        help="Directorio donde se guardan los CSV generados.",
    )
    args = parser.parse_args()

    df = cargar_y_limpiar(args.entrada)
    df_modelo = construir_dataset_modelo(df)
    entrenamiento, evaluacion = separar_temporalmente(df_modelo)
    guardar_resultados(df_modelo, entrenamiento, evaluacion, args.salida)

    # TimeSeriesSplit queda preparado para usarlo al ajustar min_info_gain.
    particiones = model_selection.TimeSeriesSplit(n_splits=5)

    print(f"Partidos originales limpios: {len(df)}")
    print(f"Instancias procesadas: {len(df_modelo)}")
    print(f"Entrenamiento hasta 2023: {len(entrenamiento)}")
    print(f"Evaluacion 2024-2025: {len(evaluacion)}")
    print(f"Particiones temporales disponibles: {particiones.get_n_splits()}")
    print(f"Archivos guardados en: {Path(args.salida).resolve()}")


if __name__ == "__main__":
    main()
