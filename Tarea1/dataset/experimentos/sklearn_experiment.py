"""
Compara los clasificadores propios (ID3, M-estimador Naive Bayes) contra
las implementaciones de scikit-learn: RandomForestClassifier y
CategoricalNB, usando EXACTAMENTE el mismo protocolo de evaluacion que ya
se usa en bayesNotebook.ipynb / treeNotebook.ipynb:

- Solo se usa `train` (partidos anteriores a 2024-01-01). El test
  2024-2025 no se toca en ningun momento.
- Validacion cruzada temporal: validar cada anio 2005-2022 (18 folds),
  entrenando con los 5 anios anteriores (window_years=5) -- el mismo
  esquema ya adoptado tras el experimento de ventanas.
- Busqueda aleatoria de hiperparametros (500 candidatos, semilla 42),
  incluyendo los margenes de discretizacion (mismos rangos que ya usan
  los notebooks) MAS los hiperparametros propios de cada modelo nuevo,
  MAS los umbrales de match_evenness (que en la corrida anterior habian
  quedado sin buscar).
- Se reporta F1 macro medio anual (criterio de seleccion), accuracy y
  balanced accuracy medias anuales, y el reporte por clase de la
  validacion agrupada (reentrenando el ganador en cada fold), igual que
  hacen los notebooks.
"""

import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
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
    from pipeline import create_model_pipeline, pipeline_input_attributes, model_attributes

    test_start = pd.Timestamp("2024-01-01")

    print("Cargando dataset (years_limit=1, matches_limit=5, igual que los notebooks)...")
    t0 = time.time()
    dataset = load_dataset("futbol_uruguayo.csv")
    print(f"Cargado en {time.time()-t0:.1f}s")

    train = dataset[dataset["date"] < test_start].copy()
    print(f"Partidos en train: {len(train)}; reservados 2024+ (no se usan): "
          f"{(dataset['date'] >= test_start).sum()}")

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
    print(f"Folds temporales: {len(temporal_splits)} (2005-2022, ventana de 5 anios)")

    labels = ["L", "E", "V"]
    target_names = ["Local", "Empate", "Visitante"]
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}
    selection_metric = "f1_macro"

    # cardinalidad de cada columna, en el orden que arma create_preprocessing
    # (differences x8 -> draw_rate x1 -> evenness x1 -> numeric x3)
    min_categories = [3] * 8 + [3] + [3] + [4, 4, 2]
    assert len(min_categories) == len(model_attributes), (len(min_categories), len(model_attributes))

    # Rangos de discretizacion: union de los ya usados en bayesNotebook y
    # treeNotebook, para no favorecer a un modelo con el rango pensado
    # para el otro.
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

    def run_search(model, model_grid, name, n_iter=500, random_state=42):
        param_grid = {**discretizer_grid, **model_grid}
        pipeline = create_model_pipeline(model, include_evenness=True)

        # La grilla completa (producto de todas las combinaciones) supera
        # el limite de un entero de 32 bits que usa internamente
        # RandomizedSearchCV cuando recibe un dict; se muestrea a mano sin
        # reemplazo, igual que ya hacen bayesNotebook.ipynb/treeNotebook.ipynb.
        import random
        from math import prod

        rng = random.Random(random_state)
        parameter_names_search = sorted(param_grid)
        candidate_count = min(n_iter, prod(len(v) for v in param_grid.values()))
        seen = set()
        sampled_parameters = []
        while len(sampled_parameters) < candidate_count:
            values = tuple(rng.choice(param_grid[name_]) for name_ in parameter_names_search)
            if values not in seen:
                seen.add(values)
                sampled_parameters.append({
                    name_: [value] for name_, value in zip(parameter_names_search, values)
                })

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=sampled_parameters,
            n_iter=candidate_count,
            scoring=scoring,
            refit=selection_metric,
            cv=temporal_splits,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

        print(f"\n{'='*70}\n{name}: iniciando busqueda ({n_iter} candidatos x {len(temporal_splits)} folds)\n{'='*70}")
        t0 = time.time()
        search.fit(X_train, y_train)
        print(f"{name}: busqueda terminada en {time.time()-t0:.1f}s")

        print(f"Mejores hiperparametros ({name}):")
        print(search.best_params_)
        best_idx = search.best_index_
        mean_f1 = search.cv_results_["mean_test_f1_macro"][best_idx]
        std_f1 = search.cv_results_["std_test_f1_macro"][best_idx]
        mean_acc = search.cv_results_["mean_test_accuracy"][best_idx]
        mean_bal = search.cv_results_["mean_test_balanced_accuracy"][best_idx]
        se_f1 = std_f1 / np.sqrt(len(temporal_splits))
        print(f"F1 macro medio anual: {mean_f1:.4%}  (std={std_f1:.4f}, SE={se_f1:.4f})")
        print(f"Accuracy media anual: {mean_acc:.4%}")
        print(f"Balanced accuracy media anual: {mean_bal:.4%}")

        # top-10 para chequear "1 SE"
        results_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
        top10 = results_df[["mean_test_f1_macro", "std_test_f1_macro", "rank_test_f1_macro"]].head(10)
        print("Top 10 candidatos (f1_macro medio, std, rank):")
        print(top10.to_string(index=False))

        # Reentrenar el ganador en cada fold y juntar predicciones (igual
        # que la celda de matriz de confusion de los notebooks).
        real_all, pred_all = [], []
        for fit_idx, val_idx in temporal_splits:
            fold_model = clone(search.best_estimator_)
            fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            preds = fold_model.predict(X_train.iloc[val_idx])
            real_all.extend(y_train.iloc[val_idx])
            pred_all.extend(preds)
        real_all = np.asarray(real_all)
        pred_all = np.asarray(pred_all)

        grouped = {
            "accuracy": accuracy_score(real_all, pred_all),
            "balanced_accuracy": balanced_accuracy_score(real_all, pred_all),
            "f1_macro": f1_score(real_all, pred_all, average="macro", zero_division=0),
        }
        report_text = classification_report(
            real_all, pred_all, labels=labels, target_names=target_names, zero_division=0,
        )
        print("\nMetricas de validacion agrupada (reentrenando el ganador en cada fold):")
        print(json.dumps(grouped, indent=2))
        print(report_text)

        return {
            "name": name,
            "best_params": {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                             for k, v in search.best_params_.items()},
            "mean_annual": {"f1_macro": float(mean_f1), "accuracy": float(mean_acc), "balanced_accuracy": float(mean_bal)},
            "std_f1_macro": float(std_f1),
            "se_f1_macro": float(se_f1),
            "grouped": grouped,
            "report_text": report_text,
            "top10": top10.to_dict(orient="records"),
        }

    results = {}

    # ============================== RANDOM FOREST ==============================
    rf_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 5, 8, 10, 15, 20],
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__max_features": ["sqrt", "log2", None],
        "model__criterion": ["gini", "entropy"],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }
    rf_model = RandomForestClassifier(random_state=42, n_jobs=1)
    results["random_forest"] = run_search(rf_model, rf_grid, "RandomForestClassifier (sklearn)")

    # ============================== CATEGORICAL NB ==============================
    nb_grid = {
        "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "model__fit_prior": [False, True],
    }
    nb_model = CategoricalNB(min_categories=min_categories)
    results["categorical_nb"] = run_search(nb_model, nb_grid, "CategoricalNB (sklearn)")

    out_path = project_dataset_dir / "experimentos" / "resultados" / "experimento_sklearn_resultados.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados guardados en {out_path}")


if __name__ == "__main__":
    main()
