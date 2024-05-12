from basic_types import Action, ActionDistribution, HiddenValue, Interval, IntervalLike
from info_set import InfoSet
from model import Model

import numpy as np

import abc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

EPS = 0.05
c_PUCT = 1.0

CHEAT = True  # evaluate model for child nodes immediately, so we don't need Vc


def to_interval(i: IntervalLike) -> Interval:
    if isinstance(i, Interval):
        return i
    return np.array([i, i])


@dataclass
class VisitResult:
    Q: Interval
    action_distr: Optional[ActionDistribution] = None  # if action was mixed


class Node(abc.ABC):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, Q: IntervalLike=0):
        self.info_set = info_set
        self.cp = info_set.get_current_player()
        self.game_outcome = info_set.get_game_outcome()
        self.Q: Optional[Interval] = to_interval(Q)
        self.N = 0
        self.tree_owner = tree_owner

    def terminal(self) -> bool:
        return self.game_outcome is not None

    @abc.abstractmethod
    def visit(self, model: Model) -> VisitResult:
        pass

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

class ActionNode(Node):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, initQ: IntervalLike=0, spawned_tree=None, eps=EPS):
        super().__init__(info_set, tree_owner=tree_owner, Q=initQ)

        self.actions = info_set.get_actions()
        self.children: Dict[Action, Node] = {}
        self.P = None
        self.V = None  #self.game_outcome if self.terminal() else None
        self.Vc = None
        self.spawned_tree = spawned_tree
        self.eps = eps
        # self.recent_action = None

        if self.terminal():
            self.Q = to_interval(self.game_outcome[self.tree_owner])

        self.PURE = np.zeros(len(self.actions))
        self.MIXED = np.zeros(len(self.actions))
        self.n_mixed = 0
        self.n_pure = 0

        self._expanded = False

    def __str__(self):
        return f'Action({self.info_set}, tree_owner={self.tree_owner}, N={self.N}, Q={self.Q}), V={self.V}'

    def eval_model(self, model: Model):
        if self.P is not None or self.terminal():
            # already evaluated
            return

        self.P, self.V, self.Vc = model.action_eval(self.tree_owner, self.info_set)
        self.Q = to_interval(self.V)

    def expand(self, model: Model):
        self._expanded = True
        if self.terminal():
            return
        self.eval_model(model)
        for a in self.actions:
            info_set = self.info_set.apply(a)
            if self.cp != info_set.get_current_player() and info_set.has_hidden_info():
                self.children[a] = SamplingNode(info_set, tree_owner=self.tree_owner, eps=self.eps, initQ=self.Vc[a])
            else:
                self.children[a] = ActionNode(info_set, tree_owner=self.tree_owner, eps=self.eps, initQ=self.Vc[a])
            if CHEAT:
                self.children[a].eval_model(model)

        self.Q = to_interval(self.V)

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

        PUCT = c_PUCT * P * np.sqrt(np.sum(N)) / (N + 1)
        PUCT = Q + PUCT[:, np.newaxis]

        # check for pure case
        max_lower_bound_index = np.argmax(PUCT[:, 0])
        max_lower_bound = PUCT[max_lower_bound_index, 0]

        logging.debug(f'-- PUCT:')
        for q, n, puct in zip(Q, N, PUCT):
            logging.debug(f'Q: {q}, N: {n}, PUCT: {puct}')

        return Q, np.where(PUCT[:, 1] >= max_lower_bound - 1e-8)[0]

    def get_mixing_distribution(self, action_indices):
        mask = np.zeros_like(self.P)
        mask[action_indices] = 1
        P = self.P * mask

        s = np.sum(P)
        assert s > 0, (self.P, mask)
        return P / s

    def visit(self, model: Model) -> VisitResult:

        logging.debug(f'= Visiting {self}:')

        self.N += 1

        if self.terminal():
            logging.debug(f'= end visit {self} hit terminal, return Q: {self.Q}')
            return VisitResult(Q=self.Q)

        if not self._expanded:
            self.expand(model)

            logging.debug(f'- expanding {self}')
            for k, child in self.children.items():
                logging.debug(f'  - {k}: {child}')
            logging.debug(f'= end visit {self} expand, return self.Q: {self.Q}')

            return VisitResult(Q=self.Q)

        old_Q = self.Q
        if self.spawned_tree is not None:
            action_distr = self.spawned_tree.get_spawned_tree_action()
            action = np.random.choice(len(self.P), p=action_distr)

            logging.debug(f'======= spawned tree action: {action}, root: {self.spawned_tree.root}')

            result = self.children[action].visit(model)
            child_Q = result.Q

            luck_adjusted_Q = self.calc_luck_adjusted_Q(child_Q, action, action_distr)

            old_Q = self.Q
            self.Q = self.Q * (self.N - 1) / self.N + luck_adjusted_Q / self.N

            logging.debug(f'- child_Q: {child_Q}, luck_adjusted: {luck_adjusted_Q}')
            logging.debug(f'- update Q to {self.Q} from {old_Q}')
            logging.debug(f'= end visit {self}')

            return VisitResult(Q=self.Q)

        else:
            mixing_distr = None
            Qc, action_indices = self.computePUCT()
            if len(action_indices) == 1:  # pure case
                self.n_pure += 1
                action_index = action_indices[0]
                pure_distr = np.zeros(len(self.P))
                pure_distr[action_index] = 1
                self.PURE = (self.PURE * (self.n_pure-1) + pure_distr) / self.n_pure

                logging.debug(f'- pure action {self.PURE}, action_index: {action_index}')

            else:  # mixed case
                self.n_mixed += 1
                mixing_distr = self.get_mixing_distribution(action_indices)
                action_index = np.random.choice(len(self.P), p=mixing_distr)
                self.MIXED = (self.MIXED * (self.n_mixed-1) + mixing_distr) / self.n_mixed

                logging.debug(f'- mixed action {self.MIXED}, action_index: {action_index}')

            na = np.newaxis
            E_mixed = np.sum(Qc * self.MIXED[:, na], axis=0)
            E_pure = np.sum(Qc * self.PURE[:, na], axis=0)
            self.Q = (self.n_mixed * E_mixed + self.n_pure * E_pure) / (self.n_mixed + self.n_pure)
            assert self.Q.shape == (2, )

            action = self.actions[action_index]
            self.recent_action = action
            self.children[action].visit(model)

            logging.debug(f'- E_mixed: {E_mixed}, E_pure: {E_pure}, n_mixed: {self.n_mixed}, n_pure: {self.n_pure}, cQc[0]: {Qc[0]}, Qc[1]: {Qc[1]}')
            logging.debug(f'- update Q to {self.Q} from {old_Q}')
            logging.debug(f'= end visit {self}')

            return VisitResult(Q=self.Q, action_distr=mixing_distr)

    def calc_luck_adjusted_Q(self, Q_h: Interval, h: int, weights) -> Interval:
        child_Qs = np.zeros((len(self.children), 2))
        for i, (_, child) in enumerate(self.children.items()):
            child_Qs[i] = child.Q
        child_Qs = np.array(child_Qs)

        ix = list(self.children.keys()).index(h)

        # this is the only difference between ActionNode and SamplingNode. ActionNode us
        luck_adjustment = self.Phi(ix, self.eps, child_Qs, weights)

        luck_adjusted_Q = np.array([Q_h[0] - luck_adjustment[1], Q_h[1] - luck_adjustment[0]])

        logging.debug(f'-- calc luck adjusted Q:')
        for q in child_Qs:
            logging.debug(f'-- child_Qs: {q}')
        logging.debug(f'-- luck_adjustment: {luck_adjustment}')

        return luck_adjusted_Q


class SamplingNode(Node):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, initQ: IntervalLike=0, eps=EPS):
        super().__init__(info_set, tree_owner, Q=initQ)
        self.H = None
        self.V = None
        self.Vc = None
        self.H_mask = info_set.get_H_mask()
        assert np.any(self.H_mask)
        self.children: Dict[HiddenValue, Node] = {}
        self._expanded = False
        self.eps = eps

    def __str__(self):
        return f'Hidden({self.info_set}, tree_owner={self.tree_owner}, N={self.N}, Q={self.Q}), V={self.V}'

    def apply_H_mask(self):
        self.H *= self.H_mask

        H_sum = np.sum(self.H)
        if H_sum < 1e-6:
            self.H = self.H_mask / np.sum(self.H_mask)
        else:
            self.H /= H_sum

    def eval_model(self, model: Model):
        if self.H is not None:
            # already evaluated
            return
        self.H, self.V, self.Vc = model.hidden_eval(self.tree_owner, self.info_set)
        self.apply_H_mask()
        self.Q = to_interval(self.V)

    def expand(self, model: Model):
        self.eval_model(model)

        for h in np.where(self.H_mask)[0]:
            info_set = self.info_set.instantiate_hidden_state(h)
            if info_set.has_hidden_info():
                assert self.cp == info_set.get_current_player()
                self.children[h] = SamplingNode(info_set, tree_owner=self.tree_owner, eps=self.eps, initQ=self.Vc[h])
            else:
                self.children[h] = ActionNode(info_set, tree_owner=self.tree_owner, eps=self.eps, initQ=self.Vc[h])
                self.children[h].spawned_tree = self.create_spawned_tree(info_set, model)
            if CHEAT:
                self.children[h].eval_model(model)

            logging.debug(f'  - {h}: {self.children[h]}')
            if self.children[h].spawned_tree is not None:
                logging.debug(f'  - spawned tree: {self.children[h].spawned_tree}')

        self._expanded = True

    def create_spawned_tree(self, info_set: InfoSet, model: Model):
        info_set = info_set.clone()
        cp = info_set.get_current_player()
        for i in range(len(info_set.cards)):
            if i != cp:
                info_set.cards[i] = None

        root = ActionNode(info_set)
        spawned_tree = Tree(model, root, self.eps)
        return spawned_tree

    def calc_luck_adjusted_Q(self, Q_h: Interval, h) -> Interval:
        child_Qs = np.zeros((len(self.children), 2))
        for i, (_, child) in enumerate(self.children.items()):
            child_Qs[i] = child.Q
        child_Qs = np.array(child_Qs)

        ix = list(self.children.keys()).index(h)
        luck_adjustment = self.Phi(ix, self.eps, child_Qs, self.H[self.H_mask != 0])
        luck_adjusted_Q = np.array([Q_h[0] - luck_adjustment[1], Q_h[1] - luck_adjustment[0]])

        logging.debug(f'-- calc luck adjusted Q:')
        for q in child_Qs:
            logging.debug(f'-- child_Qs: {q}')
        logging.debug(f'-- luck_adjustment: {luck_adjustment}')

        return luck_adjusted_Q

    def visit(self, model: Model) -> VisitResult:
        logging.debug(f'= Visiting {self}:')

        self.N += 1

        if not self._expanded:
            logging.debug(f'- expanding {self}')

            self.expand(model)

        h = np.random.choice(len(self.H), p=self.H)

        logging.debug(f'- sampling hidden state {h} from {self.H}')

        result = self.children[h].visit(model)
        Q_h = result.Q
        luck_adjusted_Q = self.calc_luck_adjusted_Q(Q_h, h)

        old_Q = self.Q
        self.Q = self.Q * (self.N - 1) / self.N + luck_adjusted_Q / self.N

        logging.debug(f'- Q_h: {Q_h}, luck_adjusted: {luck_adjusted_Q}')
        logging.debug(f'- update Q to {self.Q} from {old_Q}')
        logging.debug(f'= end visit {self}')
        return VisitResult(Q=self.Q)


class Tree:
    next_id = 0

    def __init__(self, model: Model, root: ActionNode, eps=EPS):
        self.model = model
        self.root = root
        self.tree_owner = root.info_set.get_current_player()
        self.root.tree_owner = self.tree_owner
        self.eps = eps
        self.root.eps = eps

        self.tree_id = Tree.next_id
        Tree.next_id += 1

    def __str__(self):
        return f'Tree(id={self.tree_id}, owner={self.tree_owner}, root={self.root})'

    def get_visit_distribution(self, n: int) -> Dict[Action, float]:
        while self.root.N <= n:
            logging.debug(f'\n======= visit tree: {self}')
            self.root.visit(self.model)

        n_total = self.root.N - 1
        return {action: node.N / n_total for action, node in self.root.children.items()}

    def get_spawned_tree_action(self) -> ActionDistribution:
        logging.debug(f'\n======= get action from spawn tree: {self}')

        while self.root.N < 1:
            self.root.visit(self.model)

        result = self.root.visit(self.model)
        return result.action_distr
