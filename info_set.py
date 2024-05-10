from basic_types import Action, HiddenValue

import numpy as np

import abc
from typing import List, Optional


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
    def get_game_outcome(self) -> Optional[int]:
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
