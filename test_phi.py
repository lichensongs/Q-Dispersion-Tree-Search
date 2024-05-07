from ISMCTS import SamplingNode

import itertools
import numpy as np

def add_eps(array: np.ndarray, perm: np.ndarray, eps: float) -> np.ndarray:
    eps_budget = eps
    eps_to_add = np.zeros_like(array)
    for i in perm:
        eps_to_add[i] = min(eps_budget, 1 - array[i])
        eps_budget -= eps_to_add[i]
        if eps_budget <= 0:
            break
    return array + eps_to_add

def substract_eps(array: np.ndarray, perm: np.ndarray, eps: float) -> np.ndarray:
    eps_budget = eps
    eps_to_sub = np.zeros_like(array)
    for i in perm:
        eps_to_sub[i] = min(eps_budget, array[i])
        eps_budget -= eps_to_sub[i]
        if eps_budget <= 0:
            break
    return array - eps_to_sub

def perturb_by_eps(array: np.ndarray, eps: float) -> np.ndarray:
    permumations = np.array(list(itertools.permutations(range(len(array)))))

    increased_arrays = []
    for perm in permumations:
        increased_arrays.append(add_eps(array, perm, eps))

    increased_arrays = np.array(increased_arrays)
    increased_arrays = np.unique(increased_arrays, axis=0)

    perturbed = []
    for increased in increased_arrays:
        for perm in permumations:
            perturbed.append(substract_eps(increased, perm, eps))

    perturbed = np.array(perturbed)
    perturbed = np.unique(perturbed, axis=0)
    return perturbed

def eval_weighted_interval(intervals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    upper_bound_array = np.where(weights > 0, intervals[:, 1], intervals[:, 0])
    lower_bound_array = np.where(weights > 0, intervals[:, 0], intervals[:, 1])
    upper_bound = np.dot(weights, upper_bound_array)
    lower_bound = np.dot(weights, lower_bound_array)
    return np.array([lower_bound, upper_bound])

def perturb_intervals(intervals: np.ndarray, weights: np.ndarray, eps: float, c: int) -> np.ndarray:
    perturbed_weights = perturb_by_eps(weights, eps)
    perturbed_weights = - perturbed_weights
    perturbed_weights[:, c] += 1
    
    weighted_intervals = []
    for weights in perturbed_weights:
        weighted_intervals.append(eval_weighted_interval(intervals, weights))

    weighted_intervals = np.array(weighted_intervals)
    max_upper_bound = np.max(weighted_intervals[:, 1])
    min_lower_bound = np.min(weighted_intervals[:, 0])
    return np.array([min_lower_bound, max_upper_bound])


def test_sample_phi(c, eps, intervals, weights):
    sample_interval = SamplingNode.Phi(c, eps, intervals, weights)
    perturbed_interval = perturb_intervals(intervals, weights, eps, c)
    print(f"intervals:\n{intervals}, c: {c}, eps: {eps},  weights: {weights}")
    assert np.max(np.abs(sample_interval - perturbed_interval)) <= 1e-8, f"sample_interval: {sample_interval}, perturbed_interval: {perturbed_interval}"

if __name__ == '__main__':

    c = 0
    eps = 0.1
    intervals = np.array([[0, 6], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_sample_phi(c, eps, intervals, weights)

    c = 0
    eps = 0.5
    intervals = np.array([[0, 6], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_sample_phi(c, eps, intervals, weights)
