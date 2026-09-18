import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import CategoricalNB

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
from decisionTree.classifier import Classifier
from naiveBayes.bayes import M_Estimator
from evaluacion import construir_folds, ajustar_final, evaluar_ventanas
from pipeline import create_model_pipeline


class RevisionTests(unittest.TestCase):
    def test_leaf_rejects_small_branch_and_uses_valid_alternative(self):
        X = pd.DataFrame({"perfect_but_small": [0]*9+[1], "supported": [0]*5+[1]*5})
        y = pd.Series(["L"]*9+["E"])
        model = Classifier(min_info_gain=0, min_samples_leaf=2).fit(X, y)
        self.assertEqual(model.tree_.value, "supported")
        self.assertTrue(all(not child.children for child in model.tree_.children.values()))

    def test_leaf_stops_when_no_split_is_valid(self):
        model = Classifier(min_info_gain=0, min_samples_leaf=2).fit(
            pd.DataFrame({"x": [0]*9+[1]}), ["L"]*9+["E"])
        self.assertFalse(model.tree_.children)

    def test_root_smaller_than_leaf_limit_stays_leaf(self):
        model = Classifier(min_samples_leaf=10).fit(pd.DataFrame({"x": [0, 1]}), ["L", "E"])
        self.assertFalse(model.tree_.children)

    def test_leaf_one_preserves_single_sample_branch(self):
        model = Classifier(min_info_gain=0, min_samples_leaf=1).fit(
            pd.DataFrame({"x": [0]*9+[1]}), ["L"]*9+["E"])
        self.assertEqual(model.tree_.children[1].value, "E")

    def test_invalid_leaf_and_clone(self):
        for value in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                Classifier(min_samples_leaf=value).fit(pd.DataFrame({"x": [1, 2]}), ["L", "E"])
        self.assertEqual(clone(Classifier(min_samples_leaf=5)).min_samples_leaf, 5)
        self.assertNotIn("min_samples_split", Classifier().get_params())



    def test_temporal_folds_include_2023_without_future_rows(self):
        train = pd.DataFrame({"date": pd.to_datetime([f"{y}-06-01" for y in range(2000, 2024)])})
        splits = construir_folds(train)
        self.assertEqual(len(splits), 19)
        fit, val = splits[-1]
        self.assertEqual(train.iloc[val].date.dt.year.tolist(), [2023])
        self.assertEqual(train.iloc[fit].date.dt.year.tolist(), list(range(2018, 2023)))
        for fit, val in splits:
            self.assertLess(train.iloc[fit].date.max(), train.iloc[val].date.min())
        with self.assertRaises(ValueError):
            construir_folds(pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])}))

    def test_final_fit_uses_five_years_and_excludes_test(self):
        data = pd.DataFrame({"date": pd.to_datetime([f"{y}-06-01" for y in range(2017, 2026)]),
                             "x": range(9), "result": ["L"]*9})
        fitted = ajustar_final(CategoricalNB(), data, ["x"])
        self.assertEqual(fitted.class_count_.sum(), 5)
        self.assertEqual(np.flatnonzero(fitted.category_count_[0].sum(axis=0)).tolist(), [2,3,4,5,6])

    def test_bayes_classifier_protocol_and_not_fitted(self):
        self.assertTrue(is_classifier(M_Estimator()))
        self.assertTrue(is_classifier(create_model_pipeline(M_Estimator())))
        with self.assertRaises(NotFittedError):
            M_Estimator().predict(pd.DataFrame({"x": [0]}))

    def test_bayes_smoothing_matches_reference(self):
        X = pd.DataFrame({"x": [0, 1, 2, 0, 1, 2]})
        y = ["L", "L", "E", "V", "V", "V"]
        own = M_Estimator(m=3).fit(X, y)
        ref = CategoricalNB(alpha=1).fit(X, y)
        for i, label in enumerate(ref.classes_):
            np.testing.assert_allclose([own.model[label]["x"][v] for v in range(3)], ref.feature_log_prob_[0][i])

    def test_prediction_schema_validation(self):
        for cls in [M_Estimator, Classifier]:
            fitted = cls().fit(pd.DataFrame({"a": [0, 1], "b": [1, 0]}), ["L", "E"])
            with self.assertRaises(ValueError):
                fitted.predict(pd.DataFrame({"b": [1], "a": [0]}))

    def test_window_analysis_rebuilds_only_development(self):
        # Controlo el limite de datos y el flujo, sin repetir todos los ajustes.
        columns = {"home": ["A"]*11, "away": ["B"]*11, "gh": [1]*11, "ga": [0]*11,
                   "result": ["L"]*11, "date": pd.to_datetime([f"{y}-06-01" for y in range(2014,2025)])}
        data = pd.DataFrame(columns)
        from pipeline import pipeline_input_attributes
        for name in pipeline_input_attributes:
            data[name] = 0
        def rebuild(raw, **kwargs):
            self.assertLess(raw.date.max(), pd.Timestamp("2024-01-01"))
            for name in pipeline_input_attributes:
                raw[name] = 0
            return raw
        with patch("load.load_attributes", side_effect=rebuild) as loader:
            results = evaluar_ventanas(data, {"bayes": create_model_pipeline(M_Estimator())},
                                      validation_years=[2023])
        self.assertEqual(loader.call_count, 4)
        self.assertEqual(len(results["window_years"]["bayes"]), 4)


if __name__ == "__main__":
    unittest.main()
