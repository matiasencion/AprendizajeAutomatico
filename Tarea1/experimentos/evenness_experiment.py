"""
Experimento controlado: efecto del nuevo atributo de "paridad"
(match_evenness, basado en abs(elo_difference)) sobre Naive Bayes y el
arbol de decision ID3.

Metodologia del experimento:
- Se usa exclusivamente `train` (partidos anteriores a 2024-01-01).
- El conjunto de test 2024-2025 NO se toca en ningun momento.
- Se reutilizan los mismos folds temporales que ya usan los notebooks
  (validar cada anio 2005-2023 entrenando con los 5 anios anteriores).
- Para aislar el efecto del atributo nuevo, se fijan todos los demas
  hiperparametros en los mejores valores ya encontrados por cada notebook
  (registrados en las salidas de bayesNotebook.ipynb y en
  experimentos_tree.csv) y solo se busca sobre los umbrales del nuevo
  atributo. Asi la comparacion baseline vs. con-paridad no se contamina
  con "busque mas combinaciones" en el resto del pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)



def main():
    # La estructura del proyecto es fija: experimentos/, datos/ y src/ son hermanos.
    task_dir = Path(__file__).resolve().parents[1]
    dataset_path = task_dir / "datos" / "futbol_uruguayo.csv"
    sys.path.insert(0, str(task_dir / "src"))

    from load import load_dataset
    from pipeline import create_model_pipeline, pipeline_input_attributes
    from naiveBayes.bayes import M_Estimator as BayesClassifier
    from decisionTree.classifier import Classifier as DecisionTreeClassifier

    print("Cargando dataset y calculando atributos (incluye elo_closeness)...")
    dataset = load_dataset(dataset_path)

    test_start = pd.Timestamp("2024-01-01")
    train = dataset[dataset["date"] < test_start].copy()
    # El test NO se carga como X_test/y_test en ningun momento de este script.
    print(f"Partidos en train (< 2024): {len(train)}")
    print(f"Partidos reservados como test 2024+ (no se usan): "
          f"{(dataset['date'] >= test_start).sum()}")

    X_train = train[pipeline_input_attributes].copy()
    y_train = train["result"].copy()

    # --- folds temporales identicos a los notebooks ---
    validation_years = list(range(2005, 2024))
    train_years = train["date"].dt.year.to_numpy()
    window_years = 5
    temporal_splits = []
    for validation_year in validation_years:
        fit_indices = np.flatnonzero(
            (train_years < validation_year) & (train_years >= validation_year - window_years)
        )
        validation_indices = np.flatnonzero(train_years == validation_year)
        temporal_splits.append((fit_indices, validation_indices))

    labels = ["L", "E", "V"]
    target_names = ["Local", "Empate", "Visitante"]

    def evaluate(pipeline_template, splits, X, y):
        """Entrena y valida en cada fold temporal. Devuelve metricas anuales
        (una por fold) y las predicciones agrupadas de todos los folds."""
        per_fold = {"accuracy": [], "balanced_accuracy": [], "f1_macro": []}
        real_all, pred_all = [], []

        for fit_idx, val_idx in splits:
            model = clone(pipeline_template)
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            preds = model.predict(X.iloc[val_idx])

            y_val = y.iloc[val_idx]
            per_fold["accuracy"].append(accuracy_score(y_val, preds))
            per_fold["balanced_accuracy"].append(balanced_accuracy_score(y_val, preds))
            per_fold["f1_macro"].append(f1_score(y_val, preds, average="macro", zero_division=0))

            real_all.extend(y_val)
            pred_all.extend(preds)

        real_all = np.asarray(real_all)
        pred_all = np.asarray(pred_all)

        summary = {
            "mean_annual": {k: float(np.mean(v)) for k, v in per_fold.items()},
            "std_annual": {k: float(np.std(v)) for k, v in per_fold.items()},
            "grouped": {
                "accuracy": accuracy_score(real_all, pred_all),
                "balanced_accuracy": balanced_accuracy_score(real_all, pred_all),
                "f1_macro": f1_score(real_all, pred_all, average="macro", zero_division=0),
            },
            "report": classification_report(
                real_all, pred_all, labels=labels, target_names=target_names,
                zero_division=0, output_dict=True,
            ),
            "report_text": classification_report(
                real_all, pred_all, labels=labels, target_names=target_names,
                zero_division=0,
            ),
        }
        return summary

    results = {}

    # ============================== BAYES ==============================
    print("\n" + "=" * 70)
    print("NAIVE BAYES (M-estimador)")
    print("=" * 70)

    bayes_best_params = dict(
        record_margin=0.06,
        last_matches_margin=0.03,
        goal_difference_margin=0.25,
        attack_margin=0.05,
        defense_margin=0.3,
        elo_margin=75.0,
        rest_days_margin=2.0,
        h2h_margin=0.3,
        draw_low_threshold=0.3,
        draw_high_threshold=0.6,
    )

    def make_bayes_pipeline(include_evenness, evenness_low=40.0, evenness_high=150.0):
        model = BayesClassifier(m=2.0, fit_prior=False)
        return create_model_pipeline(
            model,
            include_evenness=include_evenness,
            evenness_low_threshold=evenness_low,
            evenness_high_threshold=evenness_high,
            **bayes_best_params,
        )

    print("\n-- Baseline (sin match_evenness), hiperparametros ya elegidos por CV --")
    bayes_baseline = evaluate(make_bayes_pipeline(False), temporal_splits, X_train, y_train)
    print(json.dumps(bayes_baseline["mean_annual"], indent=2))
    print(bayes_baseline["report_text"])

    print("\n-- Busqueda de umbrales solo para match_evenness (resto fijo) --")
    evenness_grid = [
        (20.0, 100.0), (25.0, 120.0), (30.0, 130.0), (34.0, 150.0),
        (40.0, 150.0), (40.0, 175.0), (50.0, 150.0), (50.0, 180.0),
        (60.0, 200.0), (76.0, 152.0),
    ]
    bayes_search_rows = []
    for low, high in evenness_grid:
        summary = evaluate(
            make_bayes_pipeline(True, low, high), temporal_splits, X_train, y_train
        )
        bayes_search_rows.append({
            "low": low, "high": high,
            **{f"mean_{k}": v for k, v in summary["mean_annual"].items()},
        })
        print(f"low={low:>6} high={high:>6}  "
              f"f1_macro={summary['mean_annual']['f1_macro']:.4f}  "
              f"accuracy={summary['mean_annual']['accuracy']:.4f}  "
              f"bal_acc={summary['mean_annual']['balanced_accuracy']:.4f}")

    best_bayes_row = max(bayes_search_rows, key=lambda r: r["mean_f1_macro"])
    print(f"\nMejor umbral de paridad para Bayes: low={best_bayes_row['low']}, "
          f"high={best_bayes_row['high']}")

    bayes_with_evenness = evaluate(
        make_bayes_pipeline(True, best_bayes_row["low"], best_bayes_row["high"]),
        temporal_splits, X_train, y_train,
    )
    print("\n-- Con match_evenness (mejor umbral) --")
    print(json.dumps(bayes_with_evenness["mean_annual"], indent=2))
    print(bayes_with_evenness["report_text"])

    results["bayes"] = {
        "baseline": bayes_baseline,
        "with_evenness": bayes_with_evenness,
        "best_thresholds": best_bayes_row,
        "search_rows": bayes_search_rows,
    }

    # ============================== TREE ==============================
    print("\n" + "=" * 70)
    print("ARBOL DE DECISION (ID3)")
    print("=" * 70)

    tree_best_params = dict(
        record_margin=0.065,
        last_matches_margin=0.005,
        goal_difference_margin=0.725,
        attack_margin=0.22,
        defense_margin=0.015,
        elo_margin=50.0,
        rest_days_margin=5.0,
        h2h_margin=0.05,
        draw_low_threshold=0.1675,
        draw_high_threshold=0.41,
    )

    def make_tree_pipeline(include_evenness, evenness_low=40.0, evenness_high=150.0):
        model = DecisionTreeClassifier(min_info_gain=0.0075)
        return create_model_pipeline(
            model,
            include_evenness=include_evenness,
            evenness_low_threshold=evenness_low,
            evenness_high_threshold=evenness_high,
            **tree_best_params,
        )

    print("\n-- Baseline (sin match_evenness), hiperparametros ya elegidos por CV --")
    tree_baseline = evaluate(make_tree_pipeline(False), temporal_splits, X_train, y_train)
    print(json.dumps(tree_baseline["mean_annual"], indent=2))
    print(tree_baseline["report_text"])

    print("\n-- Busqueda de umbrales solo para match_evenness (resto fijo) --")
    tree_search_rows = []
    for low, high in evenness_grid:
        summary = evaluate(
            make_tree_pipeline(True, low, high), temporal_splits, X_train, y_train
        )
        tree_search_rows.append({
            "low": low, "high": high,
            **{f"mean_{k}": v for k, v in summary["mean_annual"].items()},
        })
        print(f"low={low:>6} high={high:>6}  "
              f"f1_macro={summary['mean_annual']['f1_macro']:.4f}  "
              f"accuracy={summary['mean_annual']['accuracy']:.4f}  "
              f"bal_acc={summary['mean_annual']['balanced_accuracy']:.4f}")

    best_tree_row = max(tree_search_rows, key=lambda r: r["mean_f1_macro"])
    print(f"\nMejor umbral de paridad para el arbol: low={best_tree_row['low']}, "
          f"high={best_tree_row['high']}")

    tree_with_evenness = evaluate(
        make_tree_pipeline(True, best_tree_row["low"], best_tree_row["high"]),
        temporal_splits, X_train, y_train,
    )
    print("\n-- Con match_evenness (mejor umbral) --")
    print(json.dumps(tree_with_evenness["mean_annual"], indent=2))
    print(tree_with_evenness["report_text"])

    results["tree"] = {
        "baseline": tree_baseline,
        "with_evenness": tree_with_evenness,
        "best_thresholds": best_tree_row,
        "search_rows": tree_search_rows,
    }

    # ---- ver si el arbol elige match_evenness como atributo relevante ----
    print("\n" + "=" * 70)
    print("Arbol entrenado en TODO train con match_evenness: primeros niveles")
    print("=" * 70)
    full_tree_model = make_tree_pipeline(True, best_tree_row["low"], best_tree_row["high"])
    full_tree_model.fit(X_train, y_train)
    fitted_tree = full_tree_model.named_steps["model"].tree_

    def print_shallow(node, level=0, max_level=2):
        if level > max_level:
            return
        indent = "  " * level
        print(f"{indent}{node.value}")
        for branch_value, child in node.children.items():
            print(f"{indent}  [{branch_value}]")
            print_shallow(child, level + 2, max_level)

    print_shallow(fitted_tree)

    out_path = task_dir / "experimentos" / "resultados" / "experimento_evenness_resultados_actual.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados detallados guardados en {out_path}")


if __name__ == "__main__":
    main()
