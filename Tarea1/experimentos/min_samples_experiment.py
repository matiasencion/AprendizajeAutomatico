"""Se selecciona la regularización del ID3 mediante CV 2005-2023.
La evaluacion final de test queda exclusivamente en entrega.ipynb.
"""
import json
import random
import sys
import time
from math import prod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
)


def main():
    # La estructura del proyecto es fija: experimentos/, datos/ y src/ son hermanos.
    task_dir = Path(__file__).resolve().parents[1]
    dataset_path = task_dir / "datos" / "futbol_uruguayo.csv"
    sys.path.insert(0, str(task_dir / "src"))

    from load import load_dataset
    from pipeline import create_model_pipeline, pipeline_input_attributes
    from decisionTree.classifier import Classifier as DecisionTreeClassifier

    test_start = pd.Timestamp("2024-01-01")

    print("Cargando dataset...")
    t0 = time.time()
    dataset = load_dataset(dataset_path)
    print(f"Cargado en {time.time()-t0:.1f}s")

    train = dataset[dataset["date"] < test_start].copy()
    test = dataset[dataset["date"] >= test_start].copy()
    print(f"Partidos en train: {len(train)}; test reservado: {len(test)}")

    X_train = train[pipeline_input_attributes].copy()
    y_train = train["result"].copy()

    validation_years = list(range(2005, 2024))
    window_years = 5
    train_years = train["date"].dt.year.to_numpy()
    temporal_splits = []
    for validation_year in validation_years:
        fit_idx = np.flatnonzero(
            (train_years < validation_year) & (train_years >= validation_year - window_years)
        )
        val_idx = np.flatnonzero(train_years == validation_year)
        temporal_splits.append((fit_idx, val_idx))
    print(f"Folds: {len(temporal_splits)} (2005-2023, ventana de {window_years} anios)")

    labels = ["L", "E", "V"]
    target_names = ["Local", "Empate", "Visitante"]
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}

    param_grid = {
        "preprocessing__differences__discretizer__h2h_margin": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "preprocessing__differences__discretizer__rest_days_margin": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "preprocessing__differences__discretizer__record_margin": [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.1],
        "preprocessing__differences__discretizer__last_matches_margin": [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "preprocessing__differences__discretizer__goal_difference_margin": [0.15, 0.18, 0.2, 0.22, 0.25, 0.5, 0.7, 0.725, 0.75, 0.8, 0.85, 0.9],
        "preprocessing__differences__discretizer__attack_margin": [0.0, 0.02, 0.05, 0.1, 0.2, 0.22, 0.25, 0.28, 0.3],
        "preprocessing__differences__discretizer__defense_margin": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3],
        "preprocessing__differences__discretizer__elo_margin": [25.0, 50.0, 75.0, 100.0, 150.0],
        "preprocessing__draw_rate__discretizer__low_threshold": [0.14, 0.16, 0.2, 0.25, 0.3, 0.32, 0.35],
        "preprocessing__draw_rate__discretizer__high_threshold": [0.35, 0.38, 0.4, 0.45, 0.5, 0.53, 0.55, 0.6],
        "preprocessing__evenness__discretizer__low_threshold": [20.0, 25.0, 30.0, 34.0, 40.0, 50.0, 60.0],
        "preprocessing__evenness__discretizer__high_threshold": [100.0, 120.0, 130.0, 150.0, 175.0, 200.0],
        "model__min_info_gain": [
            0.0045, 0.005, 0.0055, 0.00575, 0.006, 0.00625, 0.0065,
            0.00675, 0.007, 0.00725, 0.0075, 0.008, 0.0085,
        ],
        # nuevo: minimo de ejemplos para intentar dividir un nodo. El
        # fold mas chico tiene ~1200 filas, asi que se prueba un rango
        # que va de "sin restriccion real" (2) a "bastante conservador"
        # (una fraccion importante del fold).
        "model__min_samples_leaf": [1, 5, 10, 20, 30, 50],

    }

    def muestrear_candidatos(grilla, n_iter, random_state):
        rng = random.Random(random_state)
        nombres = sorted(grilla)
        cantidad = min(n_iter, prod(len(v) for v in grilla.values()))
        vistos = set()
        candidatos = []
        while len(candidatos) < cantidad:
            valores = tuple(rng.choice(grilla[nombre]) for nombre in nombres)
            if valores not in vistos:
                vistos.add(valores)
                candidatos.append({nombre: [valor] for nombre, valor in zip(nombres, valores)})
        return candidatos, cantidad

    pipeline = create_model_pipeline(DecisionTreeClassifier(), include_evenness=True)
    candidatos, cantidad = muestrear_candidatos(param_grid, n_iter=500, random_state=42)

    print(f"\nBuscando hiperparametros (arbol + min_samples_leaf): "
          f"{cantidad} candidatos x {len(temporal_splits)} folds...")
    t0 = time.time()
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=candidatos,
        n_iter=cantidad,
        scoring=scoring,
        refit="f1_macro",
        cv=temporal_splits,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Busqueda terminada en {time.time()-t0:.1f}s")

    best_idx = search.best_index_
    mean_f1 = search.cv_results_["mean_test_f1_macro"][best_idx]
    std_f1 = search.cv_results_["std_test_f1_macro"][best_idx]
    se_f1 = std_f1 / np.sqrt(len(temporal_splits))
    print("\nMejores hiperparametros:")
    print(search.best_params_)
    print(f"F1 macro medio anual: {mean_f1:.4%} (std={std_f1:.4f}, SE={se_f1:.4f})")
    print(f"Accuracy media anual: {search.cv_results_['mean_test_accuracy'][best_idx]:.4%}")
    print(f"Balanced accuracy media anual: {search.cv_results_['mean_test_balanced_accuracy'][best_idx]:.4%}")

    results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
    top10 = results_df[[
        "mean_test_f1_macro", "std_test_f1_macro", "rank_test_f1_macro",
        "param_model__min_samples_leaf", "param_model__min_info_gain",
    ]].head(10)
    print("\nTop 10 candidatos:")
    print(top10.to_string(index=False))

    # Reentrenar el ganador en cada fold y juntar predicciones (reporte
    # agrupado de validacion, igual que entrega.ipynb).
    real_all, pred_all = [], []
    for fit_idx, val_idx in temporal_splits:
        fold_model = clone(search.best_estimator_)
        fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        pred_all.extend(fold_model.predict(X_train.iloc[val_idx]))
        real_all.extend(y_train.iloc[val_idx])
    real_all = np.asarray(real_all)
    pred_all = np.asarray(pred_all)

    print("\nReporte de validacion agrupada (19 folds, ganador con min_samples_leaf):")
    print(classification_report(real_all, pred_all, labels=labels, target_names=target_names, zero_division=0))

    resultados = {
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "cv_mean_annual": {
            "f1_macro": float(mean_f1),
            "accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
            "balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][best_idx]),
        },
        "cv_std_f1_macro": float(std_f1),
        "cv_se_f1_macro": float(se_f1),
        "top10": top10.to_dict(orient="records"),
    }
    out_path = task_dir / "experimentos" / "resultados" / "experimento_min_samples_leaf_resultados_actual.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
