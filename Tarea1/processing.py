import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import OrdinalEncoder
DATASET_FILE="./futbol_uruguayo.csv"

dataset = pd.read_csv(DATASET_FILE, sep=";", header=None).add_prefix("c")
dataset = dataset.drop(columns=["date", "full_time", "competition", "home_ident", "away_ident", "home_country", "away_country", "home_code", "away_code", "home_continent", "away_continent", "continent", "level"])
#TODO cambiar el tratamiento de la fecha (ahora la borramos)
#quiero recorrer y comparar gh y ga para determinar el ganador y crear otra columna
dataset["winner"] = None

#crear columna result
dataset["result"] = 0 #default empates

dataset.loc[dataset["gh"] > dataset["ga"], "result"] = 1 #gana local
dataset.loc[dataset["gh"] < dataset["ga"], "result"] = -1 #gana visitante

dataset = dataset.drop(columns=["gh", "ga"])

# Seleccionas las columnas categóricas que quieres transformar
cols_categoricas = ["home", "away", "result"]

enc = OrdinalEncoder()

# fit_transform aprende las categorías y transforma todo el bloque
dataset[cols_categoricas] = enc.fit_transform(dataset[cols_categoricas])

