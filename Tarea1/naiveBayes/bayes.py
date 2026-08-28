import pandas as pd

class M_Estimator:
    def __init__(self, m=2):
        self.m = m #hiperparametro
        self.model={} #este modelo es el diccionario que va a contener las probabilidades de la siguiente forma modelo[clase][atributo][valor]= probabilidad, siendo P(atributo=valor|clase)
        self.prob_clases= {}

    #siendo X el dataframe con los atributos e Y la columna con los resultados
    def fit(self, X, Y):

        #obtenemos las clases
        clases = Y.unique()
        #ya obtenemos el total para poder sacar las probabilidades de cada clase
        total_clases=Y.len()

        for clase in clases:
            self.prob_clases[clase]= len(Y[Y==clase])/total_clases #calculamos P(clase)
            self.model[clase]={}

            for attribute in X.columns:
                self.model[clase][attribute]={}

                possibleValues= X[attribute].unique()
                p_c=1/len(possibleValues) #probabilidad de cada valor del atributo

                for value in possibleValues:
                    #calculamos P(atributo=valor|clase) con la formula de m-estimacion
                    self.model[clase][attribute][value]= (len(X[(X[attribute]==value) & (Y==clase)]) + self.m*p_c)   /   (len(Y[Y==clase]) + self.m)


    #TODO: acá me queda la duda de que hacer, si devolver una tabla en la que lo unico que tenegamos que hacer sea poner los valores de la tupla
    # y que nos de una probabilidad de cada clase (resultado del partido) posible, o si la idea es que nos den una tupla y devolvamos solo las probabilidades
    # correspondientes a esta tupla 
    #def predict(self,X):