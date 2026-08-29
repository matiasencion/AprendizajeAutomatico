import math

import pandas as pd

class M_Estimator:
    def __init__(self, m=2):
        self.m = m #hiperparametro
        self.model={} #este modelo es el diccionario que va a contener las probabilidades de la siguiente forma modelo[clase][atributo][valor]= probabilidad, siendo P(atributo=valor|clase)
        self.clases = [] #lista de clases que vamos a tener en el dataset
        self.prob_clases= {} #aca se almacenan las P(clase)
        

    #siendo X el dataframe con los atributos e Y la columna con los resultados
    def fit(self, X, Y):

        #obtenemos las clases
        self.clases = Y.unique()
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
        predictions=[]

        for _, row in X.iterrows():
            predictions.append(self.predictRow(row))

        return pd.Series(predictions, index=X.index)
        