from basic_types import HiddenArray, PolicyArray, Value, ValueChildArray

import numpy as np


import abc
from typing import Tuple


class Model(abc.ABC):
    # add abstract methods
    @abc.abstractmethod
    def eval_P(self, node) -> PolicyArray:
        pass

    @abc.abstractmethod
    def eval_V(self, node) -> Tuple[Value, ValueChildArray]:
        pass

    @abc.abstractmethod
    def eval_H(self, node) -> HiddenArray:
        pass