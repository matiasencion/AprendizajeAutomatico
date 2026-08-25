from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from atributos import CustomAttributeCreator


def create_pipeline():

    #columnas de la tabla que se van a procesar
    cols_to_process=["home", "away", "record", "last_matches", "goal_difference", "local_experience", "away_experience", "record_enough", "result"]

    #se define el encoder (igual que antes, y forzando a que los valores sean enteros)
    enc = OrdinalEncoder(dtype=int)  

    #se define el preprocesador, que aplica el encoder a las columnas que se van a procesar
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', enc, cols_to_process)
        ],
        verbose_feature_names_out=False #esto es para evitar que se agregue el prefijo "encoder__" a los nombres de las columnas procesadas
    )

    preprocessor.set_output(transform="pandas") #esto es para hacer que se devuelva un dataframe de las mismas caracteristicas que el que estamos usando hasta ahora

    #ahora definimos el pipeline
    pipeline_completo= Pipeline(
        steps=[('new_attributes',CustomAttributeCreator(years_limit=10)),
               ('encode_table', preprocessor)]
    )

    return pipeline_completo
