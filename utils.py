import pickle

class VisitCounter:
    def __init__(self):
        self.visited = 0

    def save_visited_trees(self, trees, folder_path):
        trees_to_save = [tree for tree in trees if tree.root.N > 0]

        with open(f"{folder_path}/tree-{self.visited}.pkl", 'wb') as f:
            pickle.dump(trees_to_save, f)

        self.visited += 1