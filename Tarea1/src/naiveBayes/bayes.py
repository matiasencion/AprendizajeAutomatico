"""Naive Bayes categórico con suavizado mediante un M-estimador."""

import math
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

class M_Estimator(ClassifierMixin, BaseEstimator):
    """Se acumulan log-probabilidades para evitar productos numéricamente pequeños."""

    def __init__(self, m=2, fit_prior=True):
        self.m = m #hiperparametro
        self.fit_prior = fit_prior
        # El constructor guarda parámetros; fit crea todo el estado aprendido.

    #Los logaritmos se calculan durante fit para evitar su repetición al predecir
    #siendo X el dataframe con los atributos e Y la columna con los resultados
    def fit(self, X, Y):
        if not np.isfinite(self.m) or self.m < 0:
            raise ValueError("m debe ser finito y no negativo")
        # Se admiten DataFrame, Series y arreglos de NumPy
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
        if X.empty or len(X) != len(Y) or X.isna().any().any() or Y.isna().any():
            raise ValueError("X e Y deben tener igual longitud, ser no vacios y no contener faltantes")
        self.n_features_in_ = X.shape[1]

        # Se conservan los nombres de columnas para predict()
        self.feature_names_in_ = X.columns

        # Se reinician los índices para conservar la alineación entre atributos y etiquetas
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        self.model = {}
        self.clases = []
        self.prob_clases = {}

        #Se identifican las clases observadas
        self.clases = Y.unique()
        self.classes_ = self.clases # scikit-learn espera el atributo classes_
        #El total de ejemplos permite calcular las probabilidades a priori
        total_clases=len(Y)

        for clase in self.clases:
            if self.fit_prior:
                self.prob_clases[clase]= math.log(len(Y[Y==clase])/total_clases) #Se calcula log(P(clase))
            else:
                self.prob_clases[clase]= math.log(1 / len(self.clases)) # P(clase) uniforme

            self.model[clase]={}

            for attribute in X.columns:
                self.model[clase][attribute]={}

                possibleValues= X[attribute].unique()
                p_c=1/len(possibleValues) #probabilidad de cada valor del atributo

                for value in possibleValues:
                    #Se calcula P(atributo=valor|clase) mediante el M-estimador
                    prob = (len(X[(X[attribute]==value) & (Y==clase)]) + self.m*p_c) / (len(Y[Y==clase]) + self.m)
                    # Se aplica un piso numérico para evitar log(0), por ejemplo cuando m=0
                    self.model[clase][attribute][value] = math.log(max(prob, 1e-15))

                # Se conserva una probabilidad de respaldo para valores no observados
                prob_default = (0 + self.m*p_c) / (len(Y[Y==clase]) + self.m)
                self.model[clase][attribute]["__default__"] = math.log(max(prob_default, 1e-15))
        return self


    #esta funcion va a predecir el resultado de una sola fila
    def predictRow(self, row):
        prob_per_class={}

        for clase in self.clases:
            prob_per_class[clase]=self.prob_clases[clase] #Se inicializa con log(P(clase))

            for attribute in row.index:
                value=row[attribute]
                # Para valores no observados se utiliza la probabilidad de respaldo calculada en fit
                prob_value = self.model[clase][attribute].get(value, self.model[clase][attribute]["__default__"])
                prob_per_class[clase] += prob_value #Se acumula log(P(atributo=valor|clase))

        #Se devuelve la clase con mayor probabilidad
        return max(prob_per_class, key=prob_per_class.get)

    def predict(self, X):
        check_is_fitted(self, "classes_")

        if not isinstance(X, pd.DataFrame):
            # Se utilizan las columnas registradas durante fit()
            X = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        if list(X.columns) != list(self.feature_names_in_) or X.isna().any().any():
            raise ValueError("X debe conservar las columnas de entrenamiento y no contener faltantes")

        predictions=[]

        for _, row in X.iterrows():
            predictions.append(self.predictRow(row))

        # GridSearchCV y otras herramientas de sklearn generalmente esperan que predict() retorne un numpy array
        return np.array(predictions)

