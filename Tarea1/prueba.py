import math
import tree

class clasifierTree:
    def __init__(self):
        self.tree = None

#entropia de la tabla
def entropy(table):
    entropy = 0

    possibleResults = table["result"].unique()
    
    for result in possibleResults:
        aux = (table["result"] == result).sum() / table.shape[0]
        aux = aux * math.log2(aux)
        entropy += aux

    return entropy * (-1)


#ganancia al particionar por un atributo
def infoGain(table, attribute):
    ent = entropy(table)
    uniqueValues = table[attribute].unique()
    gain = 0
    for value in uniqueValues:
        aux = (table[attribute] == value).sum() / table[attribute].shape[0]
        tableAux = table[table[attribute] == value] # TODO: checkear que si funciona asi
        aux = aux * entropy(tableAux)
        gain += aux
    gain = ent - gain
    return gain

#id3

# (arbol_padre)->[(etiqueta1, arbol_hijo1), (etiqueta2, arbol_hijo2), ...]

def getTree(attributes, table, min_info_gain):
    #si todos los resultados son iguales, retornar el resultado
    if entropy(table) == 0:
        return tree.Tree(table["result"].unique()[0],{})  
    
    #si atributos es vacio
    if len(attributes) == 0:
        return tree.Tree(table["result"].mode()[0],{})
    
    #si pasamos de acá, es que ya estamos en el else general

    #elegir atributo con mayor ganancia
    bestAttribute = None
    bestGain = 0
    for attribute in attributes:
        gain = infoGain(table, attribute)
        if gain > min_info_gain:
            if gain > bestGain:
                bestAttribute = attribute
                bestGain = gain

    children = {}

    possibleResults = table[bestAttribute].unique()
    for result in possibleResults:
        tableAux = table[table[bestAttribute] == result]
        if tableAux.shape[0] == 0:
            return tree.Tree(table["result"].mode()[0],{})    
        else:
            atributes = atributes.remove(bestAttribute) 
            children[result] = getTree(atributes, tableAux, min_info_gain)
    
    return tree.Tree(bestAttribute, children)

