from basic_types import HiddenArray, PolicyArray, Value, ValueChildArray

import numpy as np


import abc
from typing import Tuple


class Model(abc.ABC):
    # add abstract methods
    @abc.abstractmethod
    def eval_P(self, node):
        pass

    @abc.abstractmethod
    def eval_V(self, node):
        pass

    @abc.abstractmethod
    def eval_H(self, node):
        pass

    # @abc.abstractmethod
    # def action_eval(self, info_set: InfoSet) -> Tuple[PolicyArray, Value, ValueChildArray]:
    #     pass

    # @abc.abstractmethod
    # def hidden_eval(self, info_set: InfoSet) -> Tuple[HiddenArray, Value, ValueChildArray]:
    #     pass