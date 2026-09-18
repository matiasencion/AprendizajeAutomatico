"""
Experimento controlado: sensibilidad a
  (a) window_years   -> tamanio de la ventana de ENTRENAMIENTO de cada fold
      de la validacion cruzada temporal (definido en los notebooks).
  (b) years_limit / matches_limit -> tamanio de la ventana usada DENTRO de
      load_attributes para calcular record_difference, last_matches_difference,
      goal_difference_value, attack_difference, defense_difference,
      draw_rate_average, local/away_experience y record_enough.

Estos son dos conceptos distintos que ambos se llaman "ventana de anios" en
la conversacion, por eso se prueban los dos por separado.

Reglas del experimento (pedidas por el usuario):
- Solo se usa `train` (partidos anteriores a 2024-01-01). El test 2024-2025
  NO se toca en ningun momento.
- Se extiende la validacion cruzada de 2013-2022 (10 anios) a 2005-2022
  (18 anios), aplicando la recomendacion de usar mas anios de validacion
  para reducir el riesgo de sobreajustar la eleccion a pocos anios. La
  decada 2000s ya tiene una distribucion de resultados parecida a la
  actual (ver EXPERIMENTO_PARIDAD.md), asi que no reintroduce el problema
  de "otro futbol".
- Se reporta, ademas de la media, el error estandar entre folds
  (std / sqrt(n_folds)) y se aplica el criterio de "1 error estandar":
  se marcan como equivalentes al mejor candidato todos los que caen dentro
  de 1 SE de la media del mejor, en vez de quedarse ciegamente con el
  maximo (que puede ser ruido).
- Todos los demas hiperparametros quedan fijos en los mejores ya
  encontrados (incluyendo el atributo de paridad con sus mejores umbrales
  del experimento anterior), para aislar el efecto de lo que se esta
  probando ahora.

Actualizacion (2026-09-18): se repitio este experimento despues de
agregar `min_samples_split` al arbol propio (ver
EXPERIMENTO_MIN_SAMPLES_SPLIT.md), usando el ganador de esa busqueda como
`tree_best_params`/hiperparametros del modelo. Bayes no cambio, se
recalcula igual para mantener el mismo formato de resultados que ya lee
entrega.ipynb.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report


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
    from naiveBayes.bayes import M_Estimator as BayesClassifier
    from decisionTree.classifier import Classifier as DecisionTreeClassifier

    test_start = pd.Timestamp("2024-01-01")

    # Mejores hiperparametros ya encontrados (margenes, modelo, y umbrales de
    # match_evenness del experimento de paridad). Se mantienen fijos.
    bayes_best_params = dict(
        record_margin=0.06, last_matches_margin=0.03, goal_difference_margin=0.25,
        attack_margin=0.05, defense_margin=0.3, elo_margin=75.0,
        rest_days_margin=2.0, h2h_margin=0.3,
        draw_low_threshold=0.3, draw_high_threshold=0.6,
        evenness_low_threshold=50.0, evenness_high_threshold=150.0,
    )
    # Ganador de experimentos/EXPERIMENTO_MIN_SAMPLES_SPLIT.md: se agrego
    # min_samples_split al arbol propio, y este es el ganador de esa
    # busqueda (elegida por CV, no por el resultado de test). Se repite
    # el experimento de ventanas con esta configuracion porque
    # min_samples_split interactua con cuanta informacion hay disponible
    # por nodo, que es justamente lo que estas ventanas controlan.
    tree_best_params = dict(
        record_margin=0.06, last_matches_margin=0.0, goal_difference_margin=0.15,
        attack_margin=0.2, defense_margin=0.01, elo_margin=75.0,
        rest_days_margin=7.0, h2h_margin=0.2,
        draw_low_threshold=0.35, draw_high_threshold=0.45,
        evenness_low_threshold=50.0, evenness_high_threshold=120.0,
    )

    labels = ["L", "E", "V"]
    target_names = ["Local", "Empate", "Visitante"]

    def make_bayes_pipeline():
        return create_model_pipeline(BayesClassifier(m=2.0, fit_prior=False), **bayes_best_params)

    def make_tree_pipeline():
        return create_model_pipeline(
            DecisionTreeClassifier(min_info_gain=0.00625, min_samples_split=10),
            **tree_best_params,
        )

    def build_temporal_splits(train, validation_years, window_years):
        train_years = train["date"].dt.year.to_numpy()
        splits = []
        for validation_year in validation_years:
            fit_idx = np.flatnonzero(
                (train_years < validation_year) & (train_years >= validation_year - window_years)
            )
            val_idx = np.flatnonzero(train_years == validation_year)
            if len(fit_idx) == 0 or len(val_idx) == 0:
                continue
            splits.append((fit_idx, val_idx))
        return splits

    def evaluate(pipeline_template, splits, X, y):
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
        n = len(per_fold["f1_macro"])
        mean = {k: float(np.mean(v)) for k, v in per_fold.items()}
        std = {k: float(np.std(v, ddof=1)) if n > 1 else 0.0 for k, v in per_fold.items()}
        se = {k: std[k] / np.sqrt(n) for k in std}
        return {
            "n_folds": n,
            "mean": mean,
            "std": std,
            "se": se,
            "grouped": {
                "accuracy": accuracy_score(real_all, pred_all),
                "balanced_accuracy": balanced_accuracy_score(real_all, pred_all),
                "f1_macro": f1_score(real_all, pred_all, average="macro", zero_division=0),
            },
            "report_text": classification_report(
                real_all, pred_all, labels=labels, target_names=target_names, zero_division=0,
            ),
        }

    def report_one_se(rows, key="mean_f1_macro", se_key="se_f1_macro", value_label="valor"):
        best = max(rows, key=lambda r: r[key])
        threshold = best[key] - best[se_key]
        within = [r for r in rows if r[key] >= threshold]
        print(f"Mejor {value_label}: {best['label']} (f1_macro medio={best[key]:.4f}, "
              f"SE={best[se_key]:.4f}, umbral 1-SE={threshold:.4f})")
        print("Candidatos dentro de 1 SE del mejor (estadisticamente equivalentes):")
        for r in sorted(within, key=lambda r: -r[key]):
            print(f"  {r['label']:>20}  f1_macro={r[key]:.4f}  (SE={r[se_key]:.4f})")
        return best, within

    results = {}

    # =========================================================
    # PARTE A: sensibilidad a window_years (ventana de ENTRENAMIENTO de la CV)
    # Se usa el dataset con la configuracion actual de load_attributes
    # (years_limit=1, matches_limit=5) y se extiende la validacion a 2005-2022.
    # =========================================================
    print("=" * 70)
    print("PARTE A: window_years (ventana de entrenamiento de cada fold)")
    print("=" * 70)

    t0 = time.time()
    dataset_default = load_dataset("futbol_uruguayo.csv", years_limit=1, matches_limit=5)
    print(f"Dataset (years_limit=1, matches_limit=5) cargado en {time.time()-t0:.1f}s")

    train_default = dataset_default[dataset_default["date"] < test_start].copy()
    print(f"Partidos en train: {len(train_default)}; "
          f"reservados 2024+: {(dataset_default['date'] >= test_start).sum()}")

    X_train_default = train_default[pipeline_input_attributes].copy()
    y_train_default = train_default["result"].copy()

    extended_validation_years = list(range(2005, 2023))  # 18 anios

    window_values = [3, 5, 7, 10]
    part_a = {"bayes": [], "tree": []}
    for window_years in window_values:
        splits = build_temporal_splits(train_default, extended_validation_years, window_years)
        n_folds = len(splits)

        bayes_summary = evaluate(make_bayes_pipeline(), splits, X_train_default, y_train_default)
        tree_summary = evaluate(make_tree_pipeline(), splits, X_train_default, y_train_default)

        print(f"\nwindow_years={window_years}  (folds validos={n_folds})")
        print(f"  Bayes: f1_macro={bayes_summary['mean']['f1_macro']:.4f} "
              f"(SE={bayes_summary['se']['f1_macro']:.4f})  "
              f"accuracy={bayes_summary['mean']['accuracy']:.4f}  "
              f"bal_acc={bayes_summary['mean']['balanced_accuracy']:.4f}")
        print(f"  Tree:  f1_macro={tree_summary['mean']['f1_macro']:.4f} "
              f"(SE={tree_summary['se']['f1_macro']:.4f})  "
              f"accuracy={tree_summary['mean']['accuracy']:.4f}  "
              f"bal_acc={tree_summary['mean']['balanced_accuracy']:.4f}")

        part_a["bayes"].append({
            "label": f"window_years={window_years}", "window_years": window_years,
            "n_folds": n_folds,
            "mean_f1_macro": bayes_summary["mean"]["f1_macro"],
            "se_f1_macro": bayes_summary["se"]["f1_macro"],
            "summary": bayes_summary,
        })
        part_a["tree"].append({
            "label": f"window_years={window_years}", "window_years": window_years,
            "n_folds": n_folds,
            "mean_f1_macro": tree_summary["mean"]["f1_macro"],
            "se_f1_macro": tree_summary["se"]["f1_macro"],
            "summary": tree_summary,
        })

    print("\n--- Seleccion con criterio 1-SE (Bayes, window_years) ---")
    report_one_se(part_a["bayes"], value_label="window_years")
    print("\n--- Seleccion con criterio 1-SE (Tree, window_years) ---")
    report_one_se(part_a["tree"], value_label="window_years")

    results["window_years"] = part_a

    # =========================================================
    # PARTE B: sensibilidad a years_limit (ventana de load_attributes),
    # matches_limit fijo en 5. window_years fijo en 5 (valor actual de los
    # notebooks) para aislar el efecto.
    # =========================================================
    print("\n" + "=" * 70)
    print("PARTE B: years_limit (ventana de calculo de atributos historicos)")
    print("=" * 70)

    years_limit_values = [1, 2, 3]
    part_b = {"bayes": [], "tree": []}
    datasets_cache = {(1, 5): dataset_default}

    for years_limit in years_limit_values:
        key = (years_limit, 5)
        if key not in datasets_cache:
            t0 = time.time()
            datasets_cache[key] = load_dataset(
                "futbol_uruguayo.csv", years_limit=years_limit, matches_limit=5
            )
            print(f"Dataset (years_limit={years_limit}, matches_limit=5) "
                  f"cargado en {time.time()-t0:.1f}s")
        dataset = datasets_cache[key]
        train = dataset[dataset["date"] < test_start].copy()
        X_train = train[pipeline_input_attributes].copy()
        y_train = train["result"].copy()

        splits = build_temporal_splits(train, extended_validation_years, window_years=5)
        n_folds = len(splits)

        bayes_summary = evaluate(make_bayes_pipeline(), splits, X_train, y_train)
        tree_summary = evaluate(make_tree_pipeline(), splits, X_train, y_train)

        print(f"\nyears_limit={years_limit}  (folds validos={n_folds})")
        print(f"  Bayes: f1_macro={bayes_summary['mean']['f1_macro']:.4f} "
              f"(SE={bayes_summary['se']['f1_macro']:.4f})")
        print(f"  Tree:  f1_macro={tree_summary['mean']['f1_macro']:.4f} "
              f"(SE={tree_summary['se']['f1_macro']:.4f})")

        part_b["bayes"].append({
            "label": f"years_limit={years_limit}", "years_limit": years_limit,
            "n_folds": n_folds,
            "mean_f1_macro": bayes_summary["mean"]["f1_macro"],
            "se_f1_macro": bayes_summary["se"]["f1_macro"],
            "summary": bayes_summary,
        })
        part_b["tree"].append({
            "label": f"years_limit={years_limit}", "years_limit": years_limit,
            "n_folds": n_folds,
            "mean_f1_macro": tree_summary["mean"]["f1_macro"],
            "se_f1_macro": tree_summary["se"]["f1_macro"],
            "summary": tree_summary,
        })

    print("\n--- Seleccion con criterio 1-SE (Bayes, years_limit) ---")
    report_one_se(part_b["bayes"], value_label="years_limit")
    print("\n--- Seleccion con criterio 1-SE (Tree, years_limit) ---")
    report_one_se(part_b["tree"], value_label="years_limit")

    results["years_limit"] = part_b

    # =========================================================
    # PARTE C: sensibilidad a matches_limit (ventana de "forma reciente"
    # dentro de load_attributes), years_limit fijo en 1. window_years fijo
    # en 5.
    # =========================================================
    print("\n" + "=" * 70)
    print("PARTE C: matches_limit (ventana de forma reciente)")
    print("=" * 70)

    matches_limit_values = [3, 5, 8]
    part_c = {"bayes": [], "tree": []}

    for matches_limit in matches_limit_values:
        key = (1, matches_limit)
        if key not in datasets_cache:
            t0 = time.time()
            datasets_cache[key] = load_dataset(
                "futbol_uruguayo.csv", years_limit=1, matches_limit=matches_limit
            )
            print(f"Dataset (years_limit=1, matches_limit={matches_limit}) "
                  f"cargado en {time.time()-t0:.1f}s")
        dataset = datasets_cache[key]
        train = dataset[dataset["date"] < test_start].copy()
        X_train = train[pipeline_input_attributes].copy()
        y_train = train["result"].copy()

        splits = build_temporal_splits(train, extended_validation_years, window_years=5)
        n_folds = len(splits)

        bayes_summary = evaluate(make_bayes_pipeline(), splits, X_train, y_train)
        tree_summary = evaluate(make_tree_pipeline(), splits, X_train, y_train)

        print(f"\nmatches_limit={matches_limit}  (folds validos={n_folds})")
        print(f"  Bayes: f1_macro={bayes_summary['mean']['f1_macro']:.4f} "
              f"(SE={bayes_summary['se']['f1_macro']:.4f})")
        print(f"  Tree:  f1_macro={tree_summary['mean']['f1_macro']:.4f} "
              f"(SE={tree_summary['se']['f1_macro']:.4f})")

        part_c["bayes"].append({
            "label": f"matches_limit={matches_limit}", "matches_limit": matches_limit,
            "n_folds": n_folds,
            "mean_f1_macro": bayes_summary["mean"]["f1_macro"],
            "se_f1_macro": bayes_summary["se"]["f1_macro"],
            "summary": bayes_summary,
        })
        part_c["tree"].append({
            "label": f"matches_limit={matches_limit}", "matches_limit": matches_limit,
            "n_folds": n_folds,
            "mean_f1_macro": tree_summary["mean"]["f1_macro"],
            "se_f1_macro": tree_summary["se"]["f1_macro"],
            "summary": tree_summary,
        })

    print("\n--- Seleccion con criterio 1-SE (Bayes, matches_limit) ---")
    report_one_se(part_c["bayes"], value_label="matches_limit")
    print("\n--- Seleccion con criterio 1-SE (Tree, matches_limit) ---")
    report_one_se(part_c["tree"], value_label="matches_limit")

    results["matches_limit"] = part_c

    out_path = project_dataset_dir / "experimentos" / "resultados" / "experimento_ventanas_resultados.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados detallados guardados en {out_path}")


if __name__ == "__main__":
    main()
