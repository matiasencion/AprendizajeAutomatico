class Tree:

    #constructor del arbol
    def __init__(self, value, children):
        self.value = value
        self.children = children
    
    #imprimir el arbol de manera recursiva
    def print_tree(self, level=0):
        print(' ' * level * 2 + str(self.value))
        for child in self.children:
            child.print_tree(level + 1)