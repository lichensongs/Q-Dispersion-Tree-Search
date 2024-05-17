import pickle
import numpy as np

class VisitCounter:
    def __init__(self):
        self.visited = 0

    def save_visited_trees(self, trees, folder_path):
        trees_to_save = [tree for tree in trees if tree.root.N > 0]

        with open(f"{folder_path}/tree-{self.visited}.pkl", 'wb') as f:
            pickle.dump(trees_to_save, f)

        self.visited += 1

def perturb_prob_simplex(intervals: np.ndarray, probs: np.ndarray, eps=0.01):
    """
    p: a probability simplex of shape (N,)
    intervals: a matrix of shape (N, 2) where each row is the lower and upper bound for the corresponding element to p
    epsilon: the perturbation magnitude

    return: an interval of (2,) that is the union of the weighted sum of intervals by probability perturbed by epsilon
    """
    lower_index = 0
    upper_index = 1

    union_interval = np.zeros(2)
    for bound_index in [lower_index, upper_index]:
        grads_to_weights = intervals[:, bound_index]
        index_order = np.argsort(grads_to_weights)
        weights = probs.copy()
        # in order to lower the lower bound, we need to increase the weight of the smallest element in lower bound
        # in order to increase the upper bound, we need to increase the weight of the largest element in upper bound
        increase_eps_direction = 1 if bound_index == lower_index else -1

        for direction in (-1, 1):
            # when increasing weights, we can at most increase the weight to 1
            # when decreasing weights, we can at most decrease the weight to 0
            eps_limit = 1 - weights if direction == increase_eps_direction else weights
            remaining_eps = eps
            for i in index_order[::direction]:
                eps_to_use = min(remaining_eps, eps_limit[i])
                weights[i] += direction * increase_eps_direction * eps_to_use
                remaining_eps -= eps_to_use
                if remaining_eps <= 0:
                    break
        union_interval[bound_index] = np.dot(weights, intervals[:, bound_index])

    return union_interval