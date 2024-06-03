import numpy as np
import abc
from typing import List, Optional, Dict

PolicyArray = np.ndarray  # shape of (|A|,)
ValueChildArray = np.ndarray
HiddenArray = np.ndarray  # shape of (|H|,)

Value = float
Interval = np.ndarray  # shape of (2,)
IntervalLike = Interval | float
HiddenValue = int
Action = int
ActionDistribution = Dict[int, float]

class InfoSet(abc.ABC):
    @abc.abstractmethod
    def has_hidden_info(self) -> bool:
        pass

    @abc.abstractmethod
    def clone(self) -> 'InfoSet':
        pass

    @abc.abstractmethod
    def get_current_player(self) -> int:
        pass

    @abc.abstractmethod
    def get_game_outcome(self) -> Optional[np.array]:
        pass

    @abc.abstractmethod
    def get_actions(self) -> List[Action]:
        pass

    @abc.abstractmethod
    def get_H_mask(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def apply(self, action: Action) -> 'InfoSet':
        pass

    @abc.abstractmethod
    def instantiate_hidden_state(self, h: HiddenValue) -> 'InfoSet':
        pass

class VisitCounter(abc.ABC):
    @abc.abstractmethod
    def add_data(self, data):
        pass

    @abc.abstractmethod
    def take_data_snapshot(self):
        pass

    @abc.abstractmethod
    def save_snapshots(self, folder_path):
        pass