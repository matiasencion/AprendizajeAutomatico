import math
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class M_Estimator(BaseEstimator, ClassifierMixin):
    def __init__(self, m=2):
        self.m = m #hiperparametro
        # La inicialización de model, clases y prob_clases se mueve a fit() para cumplir con la API de sklearn
        
    #por temas de eficiencia, calculamos los logs directamente en el fit asi no tenemos que hacerlo en cada prediccion
    #siendo X el dataframe con los atributos e Y la columna con los resultados
    def fit(self, X, Y):
        # Convertimos a DataFrame y Series por si GridSearchCV pasa arrays de numpy
        if not isinstance(X, pd.DataFrame):
            # Si X fue entrenado previamente, intentamos reusar las columnas, sino usamos las por defecto
            cols = getattr(self, "feature_names_in_", None)
            X = pd.DataFrame(X, columns=cols)
        if not isinstance(Y, pd.Series):
            Y = pd.Series(Y)
            
        # Guardamos las columnas para usarlas en predict()
        self.feature_names_in_ = X.columns
        
        # Reseteamos los índices para evitar problemas de alineación al hacer los splits de CV
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        self.model = {}
        self.clases = []
        self.prob_clases = {}

        #obtenemos las clases
        self.clases = Y.unique()
        self.classes_ = self.clases # scikit-learn espera el atributo classes_
        #ya obtenemos el total para poder sacar las probabilidades de cada clase
        total_clases=len(Y)

        for clase in self.clases:
            self.prob_clases[clase]= math.log(len(Y[Y==clase])/total_clases) #calculamos log(P(clase))
            self.model[clase]={}

            for attribute in X.columns:
                self.model[clase][attribute]={}

                possibleValues= X[attribute].unique()
                p_c=1/len(possibleValues) #probabilidad de cada valor del atributo

                for value in possibleValues:
                    #calculamos P(atributo=valor|clase) con la formula de m-estimacion
                    self.model[clase][attribute][value]= math.log((len(X[(X[attribute]==value) & (Y==clase)]) + self.m*p_c)   /   (len(Y[Y==clase]) + self.m))

                # Guardamos la probabilidad por defecto para valores nunca vistos en el entrenamiento (frecuencia = 0)
                self.model[clase][attribute]["__default__"] = math.log((0 + self.m*p_c) / (len(Y[Y==clase]) + self.m))
        return self


    #esta funcion va a predecir el resultado de una sola fila
    def predictRow(self, row):
        prob_per_class={}

        for clase in self.clases:
            prob_per_class[clase]=self.prob_clases[clase] #inicializamos con log(P(clase))

            for attribute in row.index:
                value=row[attribute]
                # Obtenemos la probabilidad con .get(). Si 'value' no existe, usamos la probabilidad '__default__' calculada en fit
                prob_value = self.model[clase][attribute].get(value, self.model[clase][attribute]["__default__"])
                prob_per_class[clase] += prob_value #sumamos log(P(atributo=valor|clase))

        #retornamos la clase con mayor probabilidad
        return max(prob_per_class, key=prob_per_class.get)

    def predict(self, X):
        import pandas as pd
        import numpy as np
        
        if not isinstance(X, pd.DataFrame):
            # Usamos las columnas guardadas durante el fit()
            X = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))

        predictions=[]

        for _, row in X.iterrows():
            predictions.append(self.predictRow(row))

        # GridSearchCV y otras herramientas de sklearn generalmente esperan que predict() retorne un numpy array
        return np.array(predictions)
        