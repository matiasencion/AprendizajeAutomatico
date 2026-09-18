"""
Prueba si agregar `min_samples_split` al arbol ID3 propio (nueva
regularizacion: no seguir dividiendo un nodo con pocos ejemplos, ademas
de min_info_gain) mejora su estabilidad.

Metodologia (misma que entrega.ipynb):
- Se elige la configuracion por validacion cruzada temporal (18 folds,
  2005-2022, ventana de entrenamiento de 5 anios), buscando junto con los
  margenes de discretizacion y min_info_gain, 500 candidatos.
- Recien despues de fijar la configuracion por CV, se evalua UNA VEZ
  sobre el test 2024-2025 (reentrenando con la misma ventana de 5 anios
  final, igual que en la Seccion 8 de entrega.ipynb). No se vuelve a
  tocar nada en base al resultado de test.
"""
import json
import os
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
    project_dataset_dir = Path(
        r"C:\Users\agust\Desktop\Agus\Facultad\2026\2do-semestre\AA\AprendizajeAutomatico\Tarea1\dataset"
    )
    project_root = project_dataset_dir.parent
    sys.path.insert(0, str(project_dataset_dir))
    sys.path.insert(0, str(project_root))
    os.chdir(project_dataset_dir)

    from load import load_dataset
    from pipeline import create_model_pipeline, pipeline_input_attributes
    from decisionTree.classifier import Classifier as DecisionTreeClassifier

    test_start = pd.Timestamp("2024-01-01")

    print("Cargando dataset...")
    t0 = time.time()
    dataset = load_dataset("futbol_uruguayo.csv")
    print(f"Cargado en {time.time()-t0:.1f}s")

    train = dataset[dataset["date"] < test_start].copy()
    test = dataset[dataset["date"] >= test_start].copy()
    print(f"Partidos en train: {len(train)}; test reservado: {len(test)}")

    X_train = train[pipeline_input_attributes].copy()
    y_train = train["result"].copy()
    X_test = test[pipeline_input_attributes].copy()
    y_test = test["result"].copy()

    validation_years = list(range(2005, 2023))
    window_years = 5
    train_years = train["date"].dt.year.to_numpy()
    temporal_splits = []
    for validation_year in validation_years:
        fit_idx = np.flatnonzero(
            (train_years < validation_year) & (train_years >= validation_year - window_years)
        )
        val_idx = np.flatnonzero(train_years == validation_year)
        temporal_splits.append((fit_idx, val_idx))
    print(f"Folds: {len(temporal_splits)} (2005-2022, ventana de {window_years} anios)")

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
        "model__min_samples_split": [2, 5, 10, 20, 30, 50, 80, 120, 160, 200, 300, 400],
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

    print(f"\nBuscando hiperparametros (arbol + min_samples_split): "
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
        "param_model__min_samples_split", "param_model__min_info_gain",
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

    print("\nReporte de validacion agrupada (18 folds, ganador con min_samples_split):")
    print(classification_report(real_all, pred_all, labels=labels, target_names=target_names, zero_division=0))

    # ---- evaluacion final, UNA SOLA VEZ, sobre test ----
    print("\n" + "=" * 70)
    print("EVALUACION FINAL SOBRE TEST (2024-2025) - una sola vez")
    print("=" * 70)
    anio_max_train = int(train_years.max())
    fit_idx_final = np.flatnonzero(train_years >= (anio_max_train + 1 - window_years))
    print(f"Reentrenamiento final: {len(fit_idx_final)} partidos "
          f"({anio_max_train + 1 - window_years}-{anio_max_train}), misma ventana que cada fold")

    modelo_final = clone(search.best_estimator_)
    modelo_final.fit(X_train.iloc[fit_idx_final], y_train.iloc[fit_idx_final])
    pred_test = modelo_final.predict(X_test)

    valores, cuentas = np.unique(pred_test, return_counts=True)
    print("Predicciones por clase:", {str(v): int(c) for v, c in zip(valores, cuentas)})

    reporte_test = classification_report(
        y_test, pred_test, labels=labels, target_names=target_names, output_dict=True, zero_division=0,
    )
    print(classification_report(y_test, pred_test, labels=labels, target_names=target_names, zero_division=0))

    resumen_test = {
        "f1_macro": f1_score(y_test, pred_test, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_test, pred_test),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
        "f1_empate": reporte_test["Empate"]["f1-score"],
        "recall_empate": reporte_test["Empate"]["recall"],
    }
    print("\nResumen test (arbol con min_samples_split):", json.dumps(resumen_test, indent=2))

    print("\nPara comparar, el arbol SIN min_samples_split (default=2) ya habia dado en test:")
    print("  accuracy=0.39, f1_macro=0.37, recall_empate=0.27, f1_empate=0.26")

    resultados = {
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "cv_mean_annual": {
            "f1_macro": float(mean_f1),
            "accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
            "balanced_accuracy": float(search.cv_results_["mean_test_balanced_accuracy"][best_idx]),
        },
        "cv_std_f1_macro": float(std_f1),
        "cv_se_f1_macro": float(se_f1),
        "test": resumen_test,
        "test_report_text": classification_report(
            y_test, pred_test, labels=labels, target_names=target_names, zero_division=0,
        ),
        "top10": top10.to_dict(orient="records"),
    }
    out_path = project_dataset_dir / "experimentos" / "resultados" / "experimento_min_samples_split_resultados.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
