from basic_types import InfoSet, VisitCounter, Action, Interval, IntervalLike
from model import Model
from utils import perturb_prob_simplex

import numpy as np

import abc
from dataclasses import dataclass
from typing import Dict, Optional
import logging


class Constants:
    EPS = 0.0
    c_PUCT = 0.1
    Dirichlet_ALPHA = 1.0

CHEAT = True  # evaluate model for child nodes immediately, so we don't need Vc

def to_interval(i: IntervalLike) -> Interval:
    if isinstance(i, Interval):
        return i
    return np.array([i, i])

class Node(abc.ABC):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, Q: IntervalLike=0):
        self.info_set = info_set
        self.cp = info_set.get_current_player()
        self.game_outcome = info_set.get_game_outcome()
        self.Q: Optional[Interval] = to_interval(Q)
        self.N = 0
        self.tree_owner = tree_owner
        self.children: Dict[int, Edge] = {}
        self.residual_Q_to_V = 0
        self.EV = None

    def add_child(self, key: int, node: 'Node'):
        self.children[key] = Edge(len(self.children), node)
        logging.debug(f'  - {key}: {node}')

    def get_Qc(self) -> np.ndarray:
        return np.array([edge.node.Q for edge in self.children.values()])

    def get_Vc(self) -> np.ndarray:
        return np.array([edge.node.V for edge in self.children.values()])

    def terminal(self) -> bool:
        return self.game_outcome is not None

    @abc.abstractmethod
    def visit(self, model: Model):
        pass

    def calc_union_interval(self, probs, eps=Constants.EPS) -> Interval:
        Vc = self.get_Vc()
        Vc_intervals = np.tile(Vc[:, np.newaxis], (1, 2))
        child_keys = np.array(list(self.children.keys()))
        union_interval = perturb_prob_simplex(Vc_intervals, probs[child_keys], eps=eps)
        return union_interval

    def take_child_key_update(self, key: int, model: Model, prob):
        logging.debug(f'= Taking action {key} from {self}')
        edge = self.children[key]
        child = edge.node
        child.visit(model)

        if self.EV is None:
            self.EV = self.calc_union_interval(prob, eps=Constants.EPS)

        child_Q_residuals = np.stack([to_interval(v.node.residual_Q_to_V) for k, v in self.children.items()], axis=0)
        N = np.array([v.node.N for k, v in self.children.items()])
        N = N / np.sum(N)
        residual = (N[np.newaxis, :] @ child_Q_residuals)[0]
        self.Q = self.EV + residual
        self.residual_Q_to_V = (self.residual_Q_to_V * (self.N - 2) + self.Q - self.V) / (self.N - 1)

@dataclass
class Edge:
    index: int
    node: Node

class ActionNode(Node):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, initQ: IntervalLike=0):
        super().__init__(info_set, tree_owner=tree_owner, Q=initQ)

        self.actions = info_set.get_actions()
        self.P = None
        self.V = None  #self.game_outcome if self.terminal() else None
        self.Vc = None
        self.spawned_tree: Optional[Tree] = None
        self.dirichlet_draw = None

        if self.terminal():
            self.V = self.game_outcome[self.tree_owner]
            self.Q = to_interval(self.V)

        self._expanded = False

    def __str__(self):
        return f'Action({self.info_set}, tree_owner={self.tree_owner}, N={self.N}, Q={self.Q}, V={self.V})'

    def eval_model(self, model: Model):
        if self.P is not None or self.terminal():
            # already evaluated
            return

        # self.P, self.V, self.Vc = model.action_eval(self.tree_owner, self.info_set)
        self.P = model.eval_P(self)
        self.V, self.Vc = model.eval_V(self)
        if np.isnan(self.V):
            self.V = 0.0
        self.Q = to_interval(self.V)

    def expand(self, model: Model):
        logging.debug(f'- expanding {self}')

        self._expanded = True
        if self.terminal():
            return
        self.eval_model(model)
        for a in self.actions:
            info_set = self.info_set.apply(a)
            if self.cp != info_set.get_current_player() and info_set.has_hidden_info():
                node = SamplingNode(info_set, tree_owner=self.tree_owner, initQ=self.Vc[a])
            else:
                node = ActionNode(info_set, tree_owner=self.tree_owner, initQ=self.Vc[a])

            self.add_child(a, node)
            if CHEAT:
                node.eval_model(model)

        self.Q = to_interval(self.V)

    def computePUCT(self, dirichlet: bool=False):
        c = len(self.children)
        actions = np.zeros(c, dtype=int)
        Q = np.zeros((c, 2))  # mins and maxes

        if dirichlet:
            P = 0.75 * self.P + 0.25 * self.dirichlet_draw
        else:
            P = self.P

        N = np.zeros(c)
        for a, edge in self.children.items():
            i = edge.index
            child = edge.node
            actions[i] = a
            Q[i] = child.Q
            N[i] = child.N

        PUCT_N_adjustment = Constants.c_PUCT * P * np.sqrt(np.sum(N)) / (N + 1)
        PUCT_N_adjustment_raw = Constants.c_PUCT * self.P * np.sqrt(np.sum(N)) / (N + 1)
        PUCT = Q + PUCT_N_adjustment[:, np.newaxis]

        # check for pure case
        max_lower_bound_index = np.argmax(PUCT[:, 0])
        max_lower_bound = PUCT[max_lower_bound_index, 0]

        logging.debug(f'-- PUCT:')
        for q, n, puct in zip(Q, N, PUCT):
            logging.debug(f'Q: {q}, N: {n}, PUCT: {puct}')

        overlapping_action_indices = np.where(PUCT[:, 1] >= max_lower_bound - 1e-8)[0]
        action_index = np.argmax(PUCT_N_adjustment_raw[overlapping_action_indices])
        action = actions[overlapping_action_indices[action_index]]
        return action

    def visit(self, model: Model, dirichlet: bool=False):
        logging.debug(f'= Visiting {self}:')
        self.N += 1

        if self.terminal():
            logging.debug(f'= end visit {self} hit terminal, return Q: {self.Q}')
            return

        if not self._expanded:
            self.expand(model)
            logging.debug(f'= end visit {self} expand, return self.Q: {self.Q}')
            if dirichlet:
                self.dirichlet_draw = np.random.dirichlet([Constants.Dirichlet_ALPHA] * len(self.P))
            return

        if self.spawned_tree is not None:
            logging.debug(f'======= get action distr from spawn tree: {self.spawned_tree}')
            if self.spawned_tree.root.N == 0:
                self.spawned_tree.root.visit(model)
            action = self.spawned_tree.root.visit(model)
            self.take_child_key_update(action, model, self.P)
        else:
            action = self.computePUCT(dirichlet)
            self.take_child_key_update(action, model, self.P)
            return action

class SamplingNode(Node):
    def __init__(self, info_set: InfoSet, tree_owner: Optional[int] = None, initQ: IntervalLike=0):
        super().__init__(info_set, tree_owner, Q=initQ)
        self.H = None
        self.V = None
        self.Vc = None
        self.H_mask = info_set.get_H_mask()
        assert np.any(self.H_mask)
        self._expanded = False

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

        self.H = model.eval_H(self)
        self.V, self.Vc = model.eval_V(self)
        if np.isnan(self.V):
            self.V = 0.0
        self.apply_H_mask()
        self.Q = to_interval(self.V)

    def expand(self, model: Model):
        self.eval_model(model)

        for h in np.where(self.H_mask)[0]:
            info_set = self.info_set.instantiate_hidden_state(h)
            if info_set.has_hidden_info():
                assert self.cp == info_set.get_current_player()
                node = SamplingNode(info_set, tree_owner=self.tree_owner, initQ=self.Vc[h])
            else:
                node = ActionNode(info_set, tree_owner=self.tree_owner, initQ=self.Vc[h])
                node.spawned_tree = self.create_spawned_tree(info_set, model)

            self.add_child(h, node)
            if CHEAT:
                node.eval_model(model)

            logging.debug(f'  - {h}: {node}')
            if node.spawned_tree is not None:
                logging.debug(f'  - spawned tree: {node.spawned_tree}')

        self._expanded = True

    def create_spawned_tree(self, info_set: InfoSet, model: Model):
        info_set = info_set.clone()
        cp = info_set.get_current_player()
        for i in range(len(info_set.cards)):
            if i != cp:
                info_set.cards[i] = None

        root = ActionNode(info_set)
        spawned_tree = Tree(model, root)
        return spawned_tree

    def visit(self, model: Model):
        logging.debug(f'= Visiting {self}:')
        self.N += 1

        if not self._expanded:
            logging.debug(f'- expanding {self}')
            self.expand(model)
            return

        PUCT_N = {k: self.H[k] / (v.node.N + 1) for k, v in self.children.items()}
        h = max(PUCT_N, key=PUCT_N.get)
        logging.debug(f'- sampling hidden state {h} from {self.H}')

        self.take_child_key_update(h, model, self.H)


class Tree:
    next_id = 0
    visit_counter: Optional[VisitCounter] = None

    def __init__(self, model: Model, root: ActionNode):
        self.model = model
        self.root = root
        self.tree_owner = root.info_set.get_current_player()
        self.root.tree_owner = self.tree_owner

        self.tree_id = Tree.next_id
        Tree.next_id += 1
        if Tree.visit_counter is not None:
            Tree.visit_counter.add_data(self)

    def __str__(self):
        return f'Tree(id={self.tree_id}, owner={self.tree_owner}, root={self.root})'

    def get_visit_distribution(self, n: int, dirichlet: bool=False) -> Dict[Action, float]:
        while self.root.N <= n:
            logging.debug(f'======= visit tree: {self}')
            self.root.visit(self.model, dirichlet=dirichlet)
            if Tree.visit_counter is not None:
                Tree.visit_counter.take_data_snapshot()

        n_total = self.root.N - 1
        return {action: edge.node.N / n_total for action, edge in self.root.children.items()}
