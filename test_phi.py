from utils import perturb_prob_simplex

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

def perturb_weighted_sum_intervals(intervals: np.ndarray, weights: np.ndarray, eps: float) -> np.ndarray:
    perturbed_weights = perturb_by_eps(weights, eps)

    weighted_intervals = []
    for weights in perturbed_weights:
        weighted_intervals.append(eval_weighted_interval(intervals, weights))

    weighted_intervals = np.array(weighted_intervals)
    max_upper_bound = np.max(weighted_intervals[:, 1])
    min_lower_bound = np.min(weighted_intervals[:, 0])
    return np.array([min_lower_bound, max_upper_bound])

def test_func(f1, f2, params):
    f1_res = f1(*params)
    f2_res = f2(*params)
    print(f"\n========TEST=========\n params: {params}")
    assert np.max(np.abs(f1_res - f2_res)) <= 1e-8, f"f1_res: {f1_res}, f2_res: {f2_res}"
    print(f'========PASSED========{f1_res}\n')


if __name__ == '__main__':

    eps = 0.1
    intervals = np.array([[0, 6], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))

    eps = 0.3
    intervals = np.array([[1, 2], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))

    eps = 0.5
    intervals = np.array([[0, 6], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))

    eps = 0.5
    intervals = np.array([[1, 2], [3, 4]])
    weights = np.array([0.3, 0.7])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))

    eps = 0.05
    intervals = np.array([[1, 1], [-2, -2]])
    weights = np.array([2/3, 1/3])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))

    eps = 0.05
    intervals = np.array([[1, 1], [-2, -2]])
    weights = np.array([0.0, 1.0])

    test_func(perturb_weighted_sum_intervals,
              perturb_prob_simplex,
              (intervals, weights, eps))
