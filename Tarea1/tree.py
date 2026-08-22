class Tree:

    #constructor del arbol
    def __init__(self, value, children):
        self.value = value
        self.children = children
    
    #imprimir el arbol de manera recursiva
    def print_tree(self, level=0):
        print(' ' * level * 2 + str(self.value))
        for branch_value, child in self.children.items():
            print(' ' * (level + 1) * 2 + f'[{branch_value}]')
            child.print_tree(level + 2)
