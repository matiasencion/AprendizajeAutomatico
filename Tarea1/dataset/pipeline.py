import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder



#atributos comparativos que deben ser codificados antes de usarlos en los modelos
comparative_attributes = [
    "record",
    "last_matches",
    "goal_difference",
    "attack",
    "defense",
]

#atributo que representa la tendencia conjunta de los equipos al empate
draw_attributes = [
    "draw_tendency",
]

categorical_attributes = comparative_attributes + draw_attributes

# Diferencias numericas calculadas cronologicamente por load_attributes.
# Estas son las columnas que recibe el discretizador antes del encoder.
difference_attributes = [
    "record_difference",
    "last_matches_difference",
    "goal_difference_value",
    "attack_difference",
    "defense_difference",
]

draw_rate_attributes = [
    "draw_rate_average",
]

#atributos que ya son numericos y no necesitan ser codificados
numeric_attributes = [
    "local_experience",
    "away_experience",
    "record_enough",
]

#lista completa de atributos que van a recibir los clasificadores
model_attributes = categorical_attributes + numeric_attributes

#lista de atributos de entrada que recibira el pipeline completo
pipeline_input_attributes = (
    difference_attributes
    + draw_rate_attributes
    + numeric_attributes
)


# Convierte diferencias numericas en ventaja local, visitante o empate.
# Una diferencia positiva representa ventaja del equipo local y una
# diferencia negativa representa ventaja del visitante. Los valores cuyo
# valor absoluto no supera el margen se consideran paridad.
class MarginDiscretizer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        record_margin=0.05,
        last_matches_margin=0.10,
        goal_difference_margin=0.25,
        attack_margin=0.25,
        defense_margin=0.25,
    ):
        # Los parametros se guardan sin modificarlos para que sklearn pueda
        # encontrarlos y variarlos con GridSearchCV.
        self.record_margin = record_margin
        self.last_matches_margin = last_matches_margin
        self.goal_difference_margin = goal_difference_margin
        self.attack_margin = attack_margin
        self.defense_margin = defense_margin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Este transformador recibe solamente las diferencias seleccionadas
        # por el ColumnTransformer y devuelve sus versiones categoricas.
        return pd.DataFrame(
            {
                "record": self._discretize(
                    X["record_difference"],
                    self.record_margin,
                ),
                "last_matches": self._discretize(
                    X["last_matches_difference"],
                    self.last_matches_margin,
                ),
                "goal_difference": self._discretize(
                    X["goal_difference_value"],
                    self.goal_difference_margin,
                ),
                "attack": self._discretize(
                    X["attack_difference"],
                    self.attack_margin,
                ),
                "defense": self._discretize(
                    X["defense_difference"],
                    self.defense_margin,
                ),
            },
            index=X.index,
        )

    @staticmethod
    def _discretize(values, margin):
        return np.select(
            [values > margin, values < -margin],
            ["home", "away"],
            default="balanced",
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(comparative_attributes, dtype=object)


# Convierte la tasa promedio de empates en una categoria baja, media o alta.
class DrawRateDiscretizer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        low_threshold=0.20,
        high_threshold=0.35,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = X["draw_rate_average"]

        return pd.DataFrame(
            {
                "draw_tendency": np.select(
                    [
                        values < self.low_threshold,
                        values > self.high_threshold,
                    ],
                    ["low", "high"],
                    default="medium",
                )
            },
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(draw_attributes, dtype=object)


#Pipeline para discretizar y codificar las diferencias
def create_difference_pipeline(
    record_margin=0.05,
    last_matches_margin=0.10,
    goal_difference_margin=0.25,
    attack_margin=0.25,
    defense_margin=0.25,
):

    return Pipeline(
        [
            (
                "discretizer",
                MarginDiscretizer(
                    record_margin=record_margin,
                    last_matches_margin=last_matches_margin,
                    goal_difference_margin=goal_difference_margin,
                    attack_margin=attack_margin,
                    defense_margin=defense_margin,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    # away = ventaja visitante, balanced = paridad y
                    # home = ventaja local.
                    categories=[
                        ["away", "balanced", "home"],
                        ["away", "balanced", "home"],
                        ["away", "balanced", "home"],
                        ["away", "balanced", "home"],
                        ["away", "balanced", "home"],
                    ],
                    dtype=int,
                ),
            ),
        ]
    )


#Pipeline para discretizar y codificar la tendencia al empate
def create_draw_pipeline(
    low_threshold=0.20,
    high_threshold=0.35,
):

    return Pipeline(
        [
            (
                "discretizer",
                DrawRateDiscretizer(
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["low", "medium", "high"],
                    ],
                    dtype=int,
                ),
            ),
        ]
    )

#Aplica un pipeline a las diferencias y deja pasar los numericos
def create_preprocessing(
    record_margin=0.05,
    last_matches_margin=0.10,
    goal_difference_margin=0.25,
    attack_margin=0.25,
    defense_margin=0.25,
    draw_low_threshold=0.20,
    draw_high_threshold=0.35,
):

    difference_pipeline = create_difference_pipeline(
        record_margin=record_margin,
        last_matches_margin=last_matches_margin,
        goal_difference_margin=goal_difference_margin,
        attack_margin=attack_margin,
        defense_margin=defense_margin,
    )

    draw_pipeline = create_draw_pipeline(
        low_threshold=draw_low_threshold,
        high_threshold=draw_high_threshold,
    )

    preprocessing = ColumnTransformer(
        [
            (
                "differences",
                difference_pipeline,
                difference_attributes,
            ),
            (
                "draw_rate",
                draw_pipeline,
                draw_rate_attributes,
            ),
            (
                "numeric",
                "passthrough",
                numeric_attributes,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # El ID3 usa los nombres de las columnas para construir sus nodos.
    preprocessing.set_output(transform="pandas")

    return preprocessing

#Pipeline completo: preprocesamiento y modelo
def create_model_pipeline(
    model,
    record_margin=0.05,
    last_matches_margin=0.10,
    goal_difference_margin=0.25,
    attack_margin=0.25,
    defense_margin=0.25,
    draw_low_threshold=0.20,
    draw_high_threshold=0.35,
):

    preprocessing = create_preprocessing(
        record_margin=record_margin,
        last_matches_margin=last_matches_margin,
        goal_difference_margin=goal_difference_margin,
        attack_margin=attack_margin,
        defense_margin=defense_margin,
        draw_low_threshold=draw_low_threshold,
        draw_high_threshold=draw_high_threshold,
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
