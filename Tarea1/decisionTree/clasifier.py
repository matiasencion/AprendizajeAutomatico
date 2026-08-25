import math
from . import tree

class clasifier:
    def __init__(self, min_info_gain=0.9):
        self.tree = None
        self.min_info_gain = min_info_gain

    #entropia de la tabla
    def entropy(self, table):
        entropy = 0

        possibleResults = table["result"].unique()
        
        for result in possibleResults:
            aux = (table["result"] == result).sum() / table.shape[0]
            aux = aux * math.log2(aux)
            entropy += aux

        return entropy * (-1)

    #ganancia de un atributo
    def infoGain(self, table, attribute):
        ent = self.entropy(table)
        uniqueValues = table[attribute].unique()
        gain = 0
        for value in uniqueValues:
            aux = (table[attribute] == value).sum() / table[attribute].shape[0]
            tableAux = table[table[attribute] == value] # TODO: checkear que si funciona asi
            aux = aux * self.entropy(tableAux)
            gain += aux
        gain = ent - gain
        return gain

    #esta funcion retorna el atributo con mayor ganancia de informacion
    def maxGainAttribute(self, table, attributes):
        bestAttribute = None
        bestGain = 0
        for attribute in attributes:
            gain = self.infoGain(table, attribute)
            if gain >= bestGain:
                bestAttribute = attribute
                bestGain = gain
        return bestAttribute,bestGain

    #implementacion del algoritmo ID3 básico (por ahora)
    def getTree(self, attributes, table, min_info_gain):

        #si todos los resultados son iguales, retornar el resultado
        if self.entropy(table) == 0:
            return tree.Tree(table["result"].unique()[0],{})  
        
        #si atributos es vacio
        if len(attributes) == 0:
            return tree.Tree(table["result"].mode()[0],{})
        
        #si pasamos de acá, es que ya estamos en el else general
        #elegir atributo con mayor ganancia
        bestAttribute, bestGain = self.maxGainAttribute(table, attributes)

        if bestAttribute is None:
            return tree.Tree(table["result"].mode()[0], {})
    
        #si ningun atributo supera la ganancia minima, se corta la recursión
        if bestGain < min_info_gain:
            return tree.Tree(table["result"].mode()[0],{})



        children = {}

        possibleResults = table[bestAttribute].unique()
        for result in possibleResults:
            tableAux = table[table[bestAttribute] == result]
            if tableAux.shape[0] == 0:
                return tree.Tree(table["result"].mode()[0],{})    
            else:
                remaining_attributes = [
                    attribute for attribute in attributes
                    if attribute != bestAttribute
                ]
                children[result] = self.getTree(
                    remaining_attributes, tableAux, min_info_gain
                )
        
        return tree.Tree(bestAttribute, children)

    #se usa para entrenar el clasificador, se le pasa la tabla de entrenamiento y los atributos a usar
    def fit(self, attributes, table):
        self.tree = self.getTree(attributes, table, self.min_info_gain)

    #se usa para predecir el resultado de una fila de la tabla, se le pasa la fila y retorna el resultado
    def predict(self, row):
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
