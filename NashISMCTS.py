import numpy as np

import abc
from typing import Dict, List, Optional


c_PUCT = 1.0

Action = int
Interval = np.ndarray  # shape of (2,)
IntervalLike = Interval | float


def to_interval(i: IntervalLike) -> Optional[Interval]:
    if i is None:
        return None
    if isinstance(i, Interval):
        return i
    return np.array([i, i])


class InfoSet:
    pass


class Model:
    pass


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
    def __init__(self, info_set: InfoSet, initQ: Optional[IntervalLike]=None):
        super().__init__(info_set, Q=initQ)

        self.actions = info_set.get_actions()
        self.children: Dict[Action, Node] = {}
        self.P = None
        self.V = None  #self.game_outcome if self.terminal() else None
        self.Vc = None

        if self.terminal():
            self.Q = to_interval(self.game_outcome[self.cp])

        self.PN = np.zeros(len(self.actions))
        self.MIX = np.zeros(len(self.actions))
        self.n_mixed = 0
        self.n_pure = 0

    def expand(self, model: Model):
        self.P, self.V, self.Vc = model()
        for a in self.actions:
            info_set = self.info_set.apply(a)
            if self.cp == info_set.get_current_player():
                self.children[a] = ActionNode(info_set, self.Vc[a])
            else:
                self.children[a] = SamplingNode(info_set, self.Vc[a])

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
            self.PN = (self.PN * (self.n_pure-1) + pure_distr) / self.n_pure
        else:  # mixed case
            self.n_mixed += 1
            mixing_distr = self.get_mixing_distribution(action_indices)
            action_index = np.random.choice(len(self.P), p=mixing_distr)
            self.MIX = (self.MIX * (self.n_mixed-1) + mixing_distr) / self.n_mixed

        E_mixed = np.sum(Qc * self.MIX, axis=0)
        E_pure = np.sum(Qc * self.PN, axis=0)
        self.Q = (self.n_mixed * E_mixed + self.n_pure * E_pure) / (self.n_mixed + self.n_pure)
        assert self.Q.shape == (2, )

        action = self.actions[action_index]
        self.children[action].visit(model)


class SamplingNode(Node):
    pass


class Tree:
    def __init__(self, model: Model, root: ActionNode):
        self.model = model
        self.root = root

    def visit(self):
        self.root.visit()
