"""Evaluacion temporal compartida; los graficos viven en reportes.py."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

VALIDATION_YEARS = tuple(range(2005, 2024))
TEST_START = pd.Timestamp("2024-01-01")
LABELS = ["L", "E", "V"]
TARGET_NAMES = ["Local", "Empate", "Visitante"]


def construir_folds(train, validation_years=VALIDATION_YEARS, window_years=5):
    """Indices posicionales; cada validacion contiene un anio completo."""
    if window_years < 1 or not isinstance(window_years, int):
        raise ValueError("window_years debe ser un entero positivo")
    if train.empty or (train["date"] >= TEST_START).any():
        raise ValueError("Los folds solo pueden usar datos anteriores al test")
    years = train["date"].dt.year.to_numpy()
    splits = []
    for year in validation_years:
        fit = np.flatnonzero((years >= year - window_years) & (years < year))
        val = np.flatnonzero(years == year)
        if not len(fit) or not len(val):
            raise ValueError(f"Fold vacio para {year}")
        splits.append((fit, val))
    if not splits:
        raise ValueError("Se necesita al menos un fold")
    return splits


def resumen_metricas(y_real, y_pred):
    report = classification_report(y_real, y_pred, labels=LABELS,
                                   target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    return {
        "f1_macro": f1_score(y_real, y_pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "f1_empate": report["Empate"]["f1-score"],
        "recall_empate": report["Empate"]["recall"],
    }


def evaluar_folds(template, X, y, splits):
    real, pred, annual = [], [], []
    for fit, val in splits:
        model = clone(template).fit(X.iloc[fit], y.iloc[fit])
        predictions = model.predict(X.iloc[val])
        annual.append(resumen_metricas(y.iloc[val], predictions))
        real.extend(y.iloc[val])
        pred.extend(predictions)
    return resumen_metricas(real, pred), np.asarray(real), np.asarray(pred), pd.DataFrame(annual)


def ajustar_final(template, train, attributes, window_years=5, test_start=TEST_START):
    """Un solo modelo para todo el test; corte fijo y ventana reciente."""
    start = test_start - pd.DateOffset(years=window_years)
    fit = train[(train["date"] >= start) & (train["date"] < test_start)]
    if fit.empty:
        raise ValueError("No hay partidos en la ventana final")
    return clone(template).fit(fit[attributes], fit["result"])


def evaluar_ventanas(dataset, modelos, validation_years=VALIDATION_YEARS, window_years=5):
    """Sensibilidad con parametros fijos, calculada en memoria sin leer resultados.

    Los atributos se reconstruyen solo sobre desarrollo. No se seleccionan
    nuevas configuraciones ni se consulta test en este analisis descriptivo.
    """
    from load import load_attributes
    from pipeline import pipeline_input_attributes
    train_default = dataset[dataset["date"] < TEST_START].copy()
    raw = train_default[["home", "away", "date", "gh", "ga", "result"]].copy()
    datasets = {(1, 5): train_default}
    results = {}
    dimensions = {"window_years": [3, 5, 7, 10], "years_limit": [1, 2, 3], "matches_limit": [3, 5, 8]}
    for dimension, values in dimensions.items():
        results[dimension] = {name: [] for name in modelos}
        for value in values:
            key = (value if dimension == "years_limit" else 1,
                   value if dimension == "matches_limit" else 5)
            if key not in datasets:
                datasets[key] = load_attributes(raw, years_limit=key[0], matches_limit=key[1])
            train = datasets[key]
            window = value if dimension == "window_years" else window_years
            splits = construir_folds(train, validation_years, window)
            for name, template in modelos.items():
                _, _, _, annual = evaluar_folds(template, train[pipeline_input_attributes], train["result"], splits)
                f1 = annual["f1_macro"]
                results[dimension][name].append({dimension: value, "n_folds": len(splits),
                    "mean_f1_macro": float(f1.mean()),
                    "se_f1_macro": float(f1.std(ddof=1) / np.sqrt(len(f1))) if len(f1) > 1 else 0.0})
    return results
