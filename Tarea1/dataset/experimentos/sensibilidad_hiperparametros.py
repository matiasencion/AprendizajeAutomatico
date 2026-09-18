"""
Corre las busquedas de hiperparametros del arbol (con min_samples_split,
version final) y de Bayes, guardando la tabla COMPLETA de cv_results_
(los 500 candidatos, no solo el top-10) para poder graficar como varia
F1 macro con cada hiperparametro.

Mismo protocolo de siempre: 18 folds temporales (2005-2022, ventana de
5 anios), sin tocar el test.
"""
import os
import random
import sys
import time
from math import prod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


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
    from naiveBayes.bayes import M_Estimator as BayesClassifier

    test_start = pd.Timestamp("2024-01-01")

    print("Cargando dataset...")
    t0 = time.time()
    dataset = load_dataset("futbol_uruguayo.csv")
    print(f"Cargado en {time.time()-t0:.1f}s")

    train = dataset[dataset["date"] < test_start].copy()
    X_train = train[pipeline_input_attributes].copy()
    y_train = train["result"].copy()

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
    print(f"Folds: {len(temporal_splits)}")

    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}

    discretizer_grid = {
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
    }

    # ---- ARBOL (con min_samples_split, arquitectura final) ----
    grilla_arbol = {
        **discretizer_grid,
        "model__min_info_gain": [
            0.0045, 0.005, 0.0055, 0.00575, 0.006, 0.00625, 0.0065,
            0.00675, 0.007, 0.00725, 0.0075, 0.008, 0.0085,
        ],
        "model__min_samples_split": [2, 5, 10, 20, 30, 50, 80, 120, 160, 200, 300, 400],
    }
    pipeline_arbol = create_model_pipeline(DecisionTreeClassifier(), include_evenness=True)
    candidatos, cantidad = muestrear_candidatos(grilla_arbol, 500, 42)

    print(f"\nBuscando arbol: {cantidad} candidatos x {len(temporal_splits)} folds...")
    t0 = time.time()
    search_arbol = RandomizedSearchCV(
        estimator=pipeline_arbol, param_distributions=candidatos, n_iter=cantidad,
        scoring=scoring, refit="f1_macro", cv=temporal_splits, n_jobs=-1,
        random_state=42, verbose=1,
    )
    search_arbol.fit(X_train, y_train)
    print(f"Arbol terminado en {time.time()-t0:.1f}s")

    df_arbol = pd.DataFrame(search_arbol.cv_results_)
    df_arbol.to_csv(project_dataset_dir / "experimentos" / "resultados" / "cv_results_arbol.csv", index=False)
    print("Guardado cv_results_arbol.csv:", df_arbol.shape)

    # ---- BAYES ----
    grilla_bayes = {
        **discretizer_grid,
        "model__m": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
        "model__fit_prior": [False, True],
    }
    pipeline_bayes = create_model_pipeline(BayesClassifier(), include_evenness=True)
    candidatos_b, cantidad_b = muestrear_candidatos(grilla_bayes, 500, 42)

    print(f"\nBuscando bayes: {cantidad_b} candidatos x {len(temporal_splits)} folds...")
    t0 = time.time()
    search_bayes = RandomizedSearchCV(
        estimator=pipeline_bayes, param_distributions=candidatos_b, n_iter=cantidad_b,
        scoring=scoring, refit="f1_macro", cv=temporal_splits, n_jobs=-1,
        random_state=42, verbose=1,
    )
    search_bayes.fit(X_train, y_train)
    print(f"Bayes terminado en {time.time()-t0:.1f}s")

    df_bayes = pd.DataFrame(search_bayes.cv_results_)
    df_bayes.to_csv(project_dataset_dir / "experimentos" / "resultados" / "cv_results_bayes.csv", index=False)
    print("Guardado cv_results_bayes.csv:", df_bayes.shape)

    print("\nListo.")


if __name__ == "__main__":
    main()
