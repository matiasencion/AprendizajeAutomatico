import math
import numpy as np
import pandas as pd

from . import tree

class Classifier:
    def __init__(self, min_info_gain=0.9):
        self.tree = None
        self.min_info_gain = min_info_gain

    #entropia de los resultados Y
    def entropy(self, Y):
        entropy = 0

        possibleResults = Y.unique()
        
        for result in possibleResults:
            aux = (Y == result).sum() / Y.shape[0]
            aux = aux * math.log2(aux)
            entropy += aux

        return entropy * (-1)
    
    #ganancia de un atributo
    def infoGain(self, X, Y, attribute):
        ent = self.entropy(Y)
        uniqueValues = X[attribute].unique()
        gain = 0

        for value in uniqueValues:
            rowsWithValue = X[attribute] == value
            aux = rowsWithValue.sum() / X.shape[0]

            #nos quedamos con los resultados de las filas que tienen este valor
            YAux = Y.loc[rowsWithValue]
            aux = aux * self.entropy(YAux)
            gain += aux

        gain = ent - gain
        return gain

    #esta funcion retorna el atributo con mayor ganancia de informacion
    def maxGainAttribute(self, X, Y, attributes):
        bestAttribute = None
        bestGain = 0

        for attribute in attributes:
            gain = self.infoGain(X, Y, attribute)

            if gain >= bestGain:
                bestAttribute = attribute
                bestGain = gain

        return bestAttribute,bestGain

    #implementacion del algoritmo ID3 básico (por ahora)
    def getTree(self, X, Y, attributes, min_info_gain):

        #si todos los resultados son iguales, retornar el resultado
        if self.entropy(Y) == 0:
            return tree.Tree(Y.unique()[0],{})  
        
        #si atributos es vacio
        if len(attributes) == 0:
            return tree.Tree(Y.mode()[0],{})
        
        #si pasamos de acá, es que ya estamos en el else general
        #elegir atributo con mayor ganancia
        bestAttribute, bestGain = self.maxGainAttribute(X, Y, attributes)

        if bestAttribute is None:
            return tree.Tree(Y.mode()[0], {})
    
        #si ningun atributo supera la ganancia minima, se corta la recursión
        if bestGain < min_info_gain:
            return tree.Tree(Y.mode()[0],{})



        children = {}

        possibleValues = X[bestAttribute].unique()

        for value in possibleValues:
            rowsWithValue = X[bestAttribute] == value
            XAux = X.loc[rowsWithValue]
            YAux = Y.loc[rowsWithValue]

            remaining_attributes = [
                attribute for attribute in attributes
                if attribute != bestAttribute
            ]

            children[value] = self.getTree(
                XAux,
                YAux,
                remaining_attributes,
                min_info_gain
            )
        
        return tree.Tree(bestAttribute, children)

    #se usa para entrenar el clasificador, se pasan los atributos en X y los resultados en Y
    def fit(self, X, Y):
        if len(X) != len(Y):
            raise ValueError("X e Y deben tener la misma cantidad de filas")

        #hacemos copias para no modificar los datos recibidos desde el notebook
        X = X.copy()
        Y = pd.Series(
            list(Y),
            index=X.index,
            name="Y"
        )

        #todas las columnas originales de X son posibles atributos del arbol
        attributes = list(X.columns)

        self.tree = self.getTree(
            X,
            Y,
            attributes,
            self.min_info_gain
        )

        return self

    #se usa internamente para predecir el resultado de una sola fila
    def predictRow(self, row):
        if self.tree is None:
            raise ValueError("El clasificador debe entrenarse antes de predecir")

        current_node = self.tree

        while current_node.children:
            attribute_value = row[current_node.value]

            if attribute_value not in current_node.children:
                # Si el valor del atributo no está en los hijos, retornar None.
                return None

            current_node = current_node.children[attribute_value]

        return current_node.value

    #se usa para predecir todas las filas de X, igual que en los modelos de scikit-learn
    def predict(self, X):
        predictions = []

        for _, row in X.iterrows():
            predictions.append(
                self.predictRow(row)
            )

        return np.array(predictions)
