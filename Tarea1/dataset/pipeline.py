from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder



#atributos categoricos que deben ser codificados antes de usarlos en los modelos
categorical_attributes = [
    "record",
    "last_matches",
    "goal_difference",
]

#atributos que ya son numericos y no necesitan ser codificados
numeric_attributes = [
    "local_experience",
    "away_experience",
    "record_enough",
]

#lista completa de atributos que van a recibir los clasificadores
model_attributes = categorical_attributes + numeric_attributes

def create_encoder_pipeline():
    #definimos de antemano el orden de las categorias comparativas
    #V significa ventaja visitante, E significa paridad y L ventaja local
    encoder = OrdinalEncoder(
        categories=[
            ["V", "E", "L"],
            ["V", "E", "L"],
            ["V", "E", "L"],
        ],
        dtype=int,
    )

    #el encoder se aplica solo a los atributos categoricos, mientras que los
    #atributos numericos pasan sin modificaciones
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "encoder",
                encoder,
                categorical_attributes,
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

    #mantenemos el resultado como dataframe para conservar los nombres
    #de las columnas luego de aplicar la transformacion
    preprocessor.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("encode_attributes", preprocessor),
        ]
    )
