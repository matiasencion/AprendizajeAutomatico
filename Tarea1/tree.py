sclass Tree:
    def __init__(self, value, children):
        self.value = value
        self.children = children

    def print_tree(self, level=0):
        indent = "  " * level
        print(indent + str(self.value))

        for branch_value, child in self.children.items():
            print("  " * (level + 1) + f"[{branch_value}]")
            child.print_tree(level + 2)
