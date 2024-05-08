import abc
from typing import Dict, List, Optional, Tuple
import numpy as np
from enum import Enum

class Action(Enum):
    PASS = 0
    ADD_CHIP = 1


class Player(Enum):
    ALICE = 0
    BOB = 1


class Card(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2

Value = float
Interval = np.ndarray  # shape of (2,)
IntervalLike = Interval | float
HiddenValue = int

class InfoSet(abc.ABC):
    @abc.abstractmethod
    def has_hidden_info(self) -> bool:
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

class KuhnPokerInfoSet(InfoSet):
    def __init__(self, action_history: List[Action], cards: List[Optional[Card]]=[None, None]):
        self.action_history = action_history
        self.cards = cards
    
    def __str__(self):
        hist_str = ''.join(map(str, [a.value for a in self.action_history]))
        card_str = ''.join(['?' if c is None else c.name[0] for c in self.cards])
        return f'history=[{hist_str}], cards={card_str}'

    def __repr__(self):
        return str(self)
    
    def clone(self) -> 'KuhnPokerInfoSet':
        return KuhnPokerInfoSet(list(self.action_history), list(self.cards))
    
    def get_current_player(self) -> Player:
        return Player(len(self.action_history) % 2)
    
    def get_game_outcome(self) -> Optional[Value]:
        """
        None if game not terminal.
        """
        if None in self.cards:
            return None

        if len(self.action_history) < 2:
            return None

        if tuple(self.action_history[-2:]) == (Action.PASS, Action.ADD_CHIP):
            return None

        if tuple(self.action_history[-2:]) == (Action.ADD_CHIP, Action.PASS):
            winner = self.get_current_player()
        elif self.cards[Player.ALICE.value].value > self.cards[Player.BOB.value].value:
            winner = Player.ALICE
        else:
            winner = Player.BOB

        loser = Player(1 - winner.value)
        loser_pot_contribution = self._get_pot_contribution(loser)

        outcome = np.zeros(2)
        outcome[winner.value] = loser_pot_contribution
        outcome[loser.value] = -loser_pot_contribution
        return outcome
    
    def _get_pot_contribution(self, player: Player):
        action_values = [a.value for i, a in enumerate(self.action_history) if i % 2 == player.value]
        return 1 + sum(action_values)
    
    def apply(self, action: Action) -> 'InfoSet':
        action_history = self.action_history + [action]
        cards = list(self.cards)
        return InfoSet(action_history, cards)
    
    def has_hidden_info(self) -> bool:
        return None in self.cards
    
    def get_actions(self) -> List[Action]:
        return [a for a in Action]

    def get_H_mask(self) -> np.ndarray:
        H_mask = np.zeros(len(Card))
        if not self.has_hidden_info():
            return H_mask
        ix = [c.value for c in Card if c not in self.cards]
        H_mask[ix] = 1
        return H_mask
    
    def instantiate_hidden_state(self, h: HiddenValue) -> 'InfoSet':
        cp = self.get_current_player()
        cards = list(self.cards)
        cards[cp.value] = Card(h)
        return KuhnPokerInfoSet(list(self.action_history), cards)

class Model(abc.ABC):
    # add abstract methods
    @abc.abstractmethod
    def action_eval(self, info_set: InfoSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @abc.abstractmethod
    def hidden_eval(self, info_set: InfoSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

class KuhnPokerModel(Model):
    def __init__(self, p, q):
        self.p = p
        self.q = q

        self._P_tensor = np.zeros((2, 3, 2))

        J = Card.JACK.value
        Q = Card.QUEEN.value
        K = Card.KING.value

        self._P_tensor[:, K, 1] = 1  # always add chip with King
        self._P_tensor[0, Q, 0] = 1  # never bet with a Queen
        self._P_tensor[1, J, 0] = 1  # never call with a Jack

        self._P_tensor[0, J] = np.array([1-p, p])  # bluff with a Jack with prob p
        self._P_tensor[1, Q] = np.array([1-q, q])  # call with a Queen with prob q
   
    def action_eval(self, info_set: InfoSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = info_set.get_current_player()
        card = info_set.cards[cp.value]
        assert card is not None
        x = info_set.action_history[-1].value
        y = card.value
        P = self._P_tensor[x][y]
        V = 0
        Vc = np.zeros(len(Action))

        return P, V, Vc
    
    def hidden_eval(self, info_set: InfoSet) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp = info_set.get_current_player()
        H = np.ones(3)
        for k in range(len(info_set.action_history) - 1):
            if k % 2 == cp.value:
                continue
            x = info_set.action_history[k].value
            y = info_set.action_history[k+1].value
            H *= self._P_tensor[x, :, y]

        card = info_set.cards[1 - cp.value]
        assert card is not None
        H[card.value] = 0
        H /= np.sum(H)

        V = 0
        Vc = np.zeros(len(Action))
        return H, V, Vc

if __name__ == '__main__':
    info_set = KuhnPokerInfoSet([Action.PASS, Action.ADD_CHIP], [None, Card.JACK])
    print(info_set)
    print(info_set.get_current_player())
    print('get_game_outcome: ', info_set.get_game_outcome())
    print('get_actions: ', info_set.get_actions())
    print('get_H_mask: ', info_set.get_H_mask())

    model = KuhnPokerModel(1/3, 1/3)
    # print('model action_eval: ', model.action_eval(info_set))
    print('model hidden_eval: ', model.hidden_eval(info_set))