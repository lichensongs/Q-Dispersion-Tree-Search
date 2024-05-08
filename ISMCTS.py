from KuhnPoker import InfoSet, Action, HiddenValue, Interval, IntervalLike, KuhnPokerInfoSet, Model, Card, KuhnPokerModel

import numpy as np
import abc
from typing import Dict, List, Optional, Tuple


c_PUCT = 1.0
DEBUG = True

def to_interval(i: IntervalLike) -> Interval:
    if isinstance(i, Interval):
        return i
    return np.array([i, i])

class Node(abc.ABC):
    def __init__(self, info_set: InfoSet, Q: IntervalLike=0):
        self.info_set = info_set
        self.cp = info_set.get_current_player()
        self.game_outcome = info_set.get_game_outcome()
        self.Q: Optional[Interval] = to_interval(Q)
        self.N = 0

    def terminal(self) -> bool:
        return self.game_outcome is not None

    @abc.abstractmethod
    def visit(self, model: Model):
        pass


class ActionNode(Node):
    def __init__(self, info_set: InfoSet, initQ: IntervalLike=0):
        super().__init__(info_set, Q=initQ)

        self.actions = info_set.get_actions()
        self.children: Dict[Action, Node] = {}
        self.P = None
        self.V = None  #self.game_outcome if self.terminal() else None
        self.Vc = None

        if self.terminal():
            self.Q = to_interval(self.game_outcome[self.cp])

        self.PURE = np.zeros(len(self.actions))
        self.MIXED = np.zeros(len(self.actions))
        self.n_mixed = 0
        self.n_pure = 0

    def expand(self, model: Model):
        self.P, self.V, self.Vc = model.action_eval(self.info_set)
        for a in self.actions:
            info_set = self.info_set.apply(a)
            if self.cp != info_set.get_current_player() and info_set.has_hidden_info():
                self.children[a] = SamplingNode(info_set, self.Vc[a])
            else:
                self.children[a] = ActionNode(info_set, self.Vc[a])

        self.Q = to_interval(self.V[self.cp])

    def computePUCT(self):
        c = len(self.children)
        actions = np.zeros(c, dtype=int)
        Q = np.zeros((c, 2))  # mins and maxes
        P = self.P
        N = np.zeros(c)
        for i, (a, child) in enumerate(self.children.items()):
            actions[i] = a
            Q[i] = child.Q
            N[i] = child.N

        PUCT = Q + c_PUCT * P * np.sqrt(np.sum(N)) / (N + 1)

        # check for pure case
        max_lower_bound_index = np.argmax(PUCT[:, 0])
        max_lower_bound = PUCT[max_lower_bound_index, 0]
        return Q, np.where(PUCT[:, 1] >= max_lower_bound)[0]

    def get_mixing_distribution(self, action_indices):
        mask = np.zeros_like(P)
        mask[action_indices] = 1
        P = self.P * mask

        s = np.sum(P)
        assert s > 0
        return P / s

    def visit(self, model: Model):
        self.N += 1

        if self.terminal():
            return

        if self.P is None:
            self.expand(model)
            return

        Qc, action_indices = self.computePUCT()
        if len(action_indices) == 1:  # pure case
            self.n_pure += 1
            action_index = action_indices[0]
            pure_distr = np.zeros(len(self.P))
            pure_distr[action_index] = 1
            self.PURE = (self.PURE * (self.n_pure-1) + pure_distr) / self.n_pure
        else:  # mixed case
            self.n_mixed += 1
            mixing_distr = self.get_mixing_distribution(action_indices)
            action_index = np.random.choice(len(self.P), p=mixing_distr)
            self.MIXED = (self.MIXED * (self.n_mixed-1) + mixing_distr) / self.n_mixed

        na = np.newaxis
        E_mixed = np.sum(Qc * self.MIXED[:, na], axis=0)
        E_pure = np.sum(Qc * self.PURE[:, na], axis=0)
        self.Q = (self.n_mixed * E_mixed + self.n_pure * E_pure) / (self.n_mixed + self.n_pure)
        assert self.Q.shape == (2, )

        action = self.actions[action_index]
        self.children[action].visit(model)


class SamplingNode(Node):
    def __init__(self, info_set: InfoSet, initQ: IntervalLike=0):
        super().__init__(info_set, Q=initQ)
        self.H = None
        self.V = None
        self.Vc = None
        self.H_mask = info_set.get_H_mask()
        assert np.any(self.H_mask)
        self.children: Dict[HiddenValue, Node] = {}
    
    def apply_H_mask(self):
        self.H *= self.H_mask

        H_sum = np.sum(self.H)
        if H_sum < 1e-6:
            self.H = self.H_mask / np.sum(self.H_mask)
        else:
            self.H /= H_sum

    @staticmethod
    def Phi(c: HiddenValue, eps: float, Q: np.ndarray, H: np.ndarray, verbose=False) -> Interval:
        """
        H: shape of (n, )
        Q: shape of (n, 2)

        H is a hidden state probability distribution.
        Q represents a utility-belief-interval for each hidden state.
        c is an index sampled from H.

        Computes:

        Phi(H) = union_{H' in N_epsilon(H)} phi(H')
        
        where 
        
        N_epsilon(H) = {H' | ||H - H'||_1 <= epsilon}
        Q_left = Q[:, 0]
        Q_right = Q[:, 1]
        phi(H') = union_{Q_left <= q <= Q_right} (q[c] - sum_i H'[i] * q[i])

        Returns the interval Phi(H) as an Interval.
        """
        one_c = np.zeros_like(H)
        one_c[c] = 1

        output = np.zeros(2)
        for index in (0, 1):
            index_sign = 1 - 2 * index
            H_prime = H.copy()
            phi_partial_extreme = -Q[:, 1 - index]
            phi_partial_extreme[c] = -Q[c, index]
            index_ordering = np.argsort(phi_partial_extreme)

            for direction in (-1, 1):
                eps_limit = (1 - H_prime) if direction == index_sign else H_prime
                remaining_eps = eps
                for i in index_ordering[::direction]:
                    eps_to_use = min(remaining_eps, eps_limit[i])
                    H_prime[i] += index_sign * direction * eps_to_use
                    remaining_eps -= eps_to_use
                    if remaining_eps <= 0:
                        break

            assert np.isclose(np.sum(H_prime), 1), H_prime

            q = Q[:, 1 - index].copy()
            q[c] = Q[c, index]
            output[index] = np.dot(one_c - H_prime, q)

        if verbose:
            print('*****')
            print('Phi computation:')
            print('H = %s' % H)
            print('Q = %s' % Q)
            print('c = %s' % c)
            print('eps = %s' % eps)
            print('Phi = %s' % output)
        return output

    def expand(self, model: Model):
        self.H, self.V, self.Vc = model.hidden_eval(self.info_set)
        self.apply_H_mask()

        for h in np.where(self.H_mask)[0]:
            info_set = self.info_set.instantiate_hidden_state(h)
            if info_set.has_hidden_info():
                assert self.cp == info_set.get_current_player()
                self.children[h] = SamplingNode(info_set, self.Vc[h])
            else:
                self.children[h] = ActionNode(info_set, self.Vc[h])

        self.Q = to_interval(self.V[self.cp])

    def visit(self, model: Model):
        self.N += 1

        if self.H is None:
            self.expand(model)

        h = np.random.choice(len(self.H), p=self.H)

        self.children[h].visit(model)

        
class Tree:
    def __init__(self, model: Model, root: ActionNode):
        self.model = model
        self.root = root

    def get_visit_distribution(self, n: int) -> Dict[Action, float]:
        while self.root.N <= n:
            if DEBUG:
                print(f'=============== # visit: {self.root.N}  ===============')
            self.root.visit(self.model)
            if self.root.N == 1:
                continue

        n_total = self.root.N
        return {action: node.N / n_total for action, node in self.root.children.items()}

    def visit(self):
        self.root.visit()


if __name__ == '__main__':
    info_set = KuhnPokerInfoSet([Action.PASS, Action.ADD_CHIP], [Card.QUEEN, None])
    model = KuhnPokerModel(1/3, 1/3)
    root = ActionNode(info_set)
    mcts = Tree(model, root)
    mcts.get_visit_distribution(5)