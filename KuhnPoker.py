from basic_types import Action, HiddenArray, HiddenValue, PolicyArray, Value, ValueChildArray
from ISMCTS import ActionNode, Constants, Tree, visit_counter
from info_set import InfoSet
from model import Model
from utils import VisitCounter

import numpy as np

from enum import Enum
from typing import List, Optional, Tuple
import argparse
import logging

PASS = 0
ADD_CHIP = 1


class Player(Enum):
    ALICE = 0
    BOB = 1


class Card(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2


class KuhnPokerInfoSet(InfoSet):
    def __init__(self, action_history: List[Action], cards: List[Optional[Card]]=[None, None]):
        self.action_history = action_history
        self.cards = cards

    def __str__(self):
        hist_str = ''.join(map(str, [a for a in self.action_history]))
        card_str = ''.join(['?' if c is None else c.name[0] for c in self.cards])
        return f'history=[{hist_str}], cards={card_str}'

    def __repr__(self):
        return str(self)

    def clone(self) -> 'KuhnPokerInfoSet':
        return KuhnPokerInfoSet(list(self.action_history), list(self.cards))

    def get_current_player(self) -> Player:
        return Player(len(self.action_history) % 2).value

    def get_game_outcome(self) -> Optional[Value]:
        """
        None if game not terminal.
        """
        if None in self.cards:
            return None

        if len(self.action_history) < 2:
            return None

        if tuple(self.action_history[-2:]) == (PASS, ADD_CHIP):
            return None

        if tuple(self.action_history[-2:]) == (ADD_CHIP, PASS):
            winner = self.get_current_player()
        elif self.cards[Player.ALICE.value].value > self.cards[Player.BOB.value].value:
            winner = 0 #Player.ALICE.value``
        else:
            winner = 1 #Player.BOB.value

        loser = Player(1 - winner).value
        loser_pot_contribution = self._get_pot_contribution(loser)

        outcome = np.zeros(2)
        outcome[winner] = loser_pot_contribution
        outcome[loser] = -loser_pot_contribution
        return outcome

    def _get_pot_contribution(self, player: Player):
        action_values = [a for i, a in enumerate(self.action_history) if i % 2 == player]
        return 1 + sum(action_values)

    def apply(self, action: Action) -> 'InfoSet':
        action_history = self.action_history + [action]
        cards = list(self.cards)
        return KuhnPokerInfoSet(action_history, cards)

    def has_hidden_info(self) -> bool:
        return None in self.cards

    def get_actions(self) -> List[Action]:
        return [PASS, ADD_CHIP]

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
        cards[cp] = Card(h)
        return KuhnPokerInfoSet(list(self.action_history), cards)


class  KuhnPokerModel(Model):
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

        self._V_tensor = np.zeros((2, 2, 3))  # owner, prev_action, card
        # self._V_hidden_tensor = np.zeros((2, 2, 3))  # owner, prev_action, card

        self._V_tensor[0, 0, J] = -1
        self._V_tensor[0, 0, Q] = -1 # Alice's tree, [010], Q?
        self._V_tensor[0, 0, K] = -1 # Alice's tree, [010], K?

        self._V_tensor[0, 1, J] = -1
        self._V_tensor[0, 1, Q] = 2 * (p - 1) / (1 + p) # Alice's tree, [011], Q?
        self._V_tensor[0, 1, K] = +2

        self._V_tensor[1, 0, J] = -1 + p * (1 - 3*q) / 2
        self._V_tensor[1, 0, Q] = 0
        self._V_tensor[1, 0, K] = 1 + q / 2

        self._V_tensor[1, 1, J] = -0.5 - 1.5 * q # Bob's tree [01] ?J
        self._V_tensor[1, 1, Q] = 1 - 3 * q # Bob's tree, [01], QJ
        self._V_tensor[1, 1, K] = -2 # Bob's tree, [01], KJ


    def action_eval(self, tree_owner: int, info_set: InfoSet) -> Tuple[PolicyArray, Value, ValueChildArray]:
        cp = info_set.get_current_player()
        card = info_set.cards[cp]
        assert card is not None
        x = info_set.action_history[-1]
        y = card.value
        P = self._P_tensor[x][y]
        V = self._V_tensor[tree_owner][x][y]
        Vc = np.zeros(2)

        return P, V, Vc

    def hidden_eval(self, tree_owner: int, info_set: InfoSet) -> Tuple[HiddenArray, Value, ValueChildArray]:
        cp = info_set.get_current_player()
        H = np.ones(3)
        for k in range(len(info_set.action_history) - 1):
            if k % 2 == cp:
                continue
            x = info_set.action_history[k]
            y = info_set.action_history[k+1]
            H *= self._P_tensor[x, :, y]

        card = info_set.cards[1 - cp]
        assert card is not None
        H[card.value] = 0
        H /= np.sum(H)

        x = info_set.action_history[-1]
        y = card.value

        V = self._V_tensor[tree_owner][x][y]
        Vc = np.zeros(len(Card))
        return H, V, Vc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    parser.add_argument("--player", type=str, help="Alice or Bob")
    parser.add_argument("--iter", type=int, help="Number of iterations")
    parser.add_argument("--eps", type=float, help="Range parameter for Q-value uncertainty")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--savetrees", action='store_true', help="Save visited trees")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s",
        filename="kuhn_poker.log",
        filemode='w'
    )

    if args.eps is not None:
        Constants.EPS = args.eps

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.savetrees:
        visit_counter = VisitCounter()

    if args.player == 'Alice':
        info_set = KuhnPokerInfoSet([PASS, ADD_CHIP], [Card.QUEEN, None])
    elif args.player == 'Bob':
        info_set = KuhnPokerInfoSet([PASS], [None, Card.JACK])
    else:
        raise ValueError(f"Invalid player: {args.player}")

    model = KuhnPokerModel(1/3, 1/3)
    root = ActionNode(info_set)
    mcts = Tree(model, root)
    visit_dist = mcts.get_visit_distribution(args.iter)
    print(visit_dist)
    if visit_counter is not None:
        visit_counter.save_visited_trees('debug')
