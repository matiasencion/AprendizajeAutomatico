import pandas as pd
from sklearn import preprocessing, model_selection

DATASET_FILE="./futbol_uruguayo.csv"

dataset = pd.read_csv(DATASET_FILE, sep=";", header=None).add_prefix("c")