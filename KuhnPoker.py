from basic_types import Action, HiddenArray, HiddenValue, PolicyArray, Value, ValueChildArray, InfoSet
from ISMCTS import ActionNode, Constants, Tree, Node, SamplingNode
from model import Model
from AlphaZero import AlphaZero, NNModel
from utils import TreeVisitCounter

import numpy as np
import torch
import torch.nn as nn
import random
import pickle
from torch.utils.data import Dataset, DataLoader

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
        return f'[{hist_str}], cards={card_str}'

    def __repr__(self):
        return str(self)

    def clone(self) -> 'KuhnPokerInfoSet':
        return KuhnPokerInfoSet(list(self.action_history), list(self.cards))

    def to_tensor(self) -> torch.Tensor:
        hist = -1 * torch.ones(3, dtype=torch.float32)
        hist[:len(self.action_history)] = torch.tensor(self.action_history, dtype=torch.float32)
        cards = torch.tensor([-1 if c is None else c.value for c in self.cards], dtype=torch.float32)
        return torch.cat([hist, cards])

    def to_action_tensor(self) -> torch.Tensor:
        return torch.concat([torch.tensor([0], dtype=torch.float32), self.to_action_info_set().to_tensor()])

    def to_sampling_tensor(self) -> torch.Tensor:
        return torch.concat([torch.tensor([1], dtype=torch.float32), self.to_sampling_info_set().to_tensor()])

    def to_spawned_tensor(self) -> torch.Tensor:
        return torch.concat([torch.tensor([2], dtype=torch.float32), self.to_tensor()])

    def to_sampling_info_set(self) -> 'KuhnPokerInfoSet':
        info_set = self.clone()
        info_set.cards[info_set.get_current_player()] = None
        return info_set

    def to_action_info_set(self) -> 'KuhnPokerInfoSet':
        info_set = self.clone()
        info_set.cards[1 - info_set.get_current_player()] = None
        return info_set

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

        # node type: 0=Action, 1=Hidden, 2=Action with Spawned, prev action, card (cp card if action or owner's card if hidden)
        self._V_tensor = np.random.normal(size=(3, 2, 3))

        self._V_tensor[0, 0, J] = -1 + p * (1 - 3*q) / 2 # Action, Bob's turn, [0], ?J
        self._V_tensor[0, 0, Q] = 0 # Action, Bob's turn, [0], ?Q
        self._V_tensor[0, 0, K] = 1 + q / 2 # Action, Bob's turn, [0], ?K

        self._V_tensor[0, 1, J] = -1 # Action, Alice's turn, [01], J?
        self._V_tensor[0, 1, Q] = 2 * (p - 1) / (1 + p) # Action, Alice's turn, [01], Q?
        self._V_tensor[0, 1, K] = 2 # Action, Alice's turn, [01], K?

        self._V_tensor[1, 0, :] = -1

        self._V_tensor[1, 1, J] = -0.5 - 1.5 * q # Hidden, Bob's tree, [01], ?J
        self._V_tensor[1, 1, Q] = 2 * (p - 1) / (1 + p) # Hidden, Alice's tree, [011], Q?
        self._V_tensor[1, 1, K] = 2 # Hidden, Alice's tree, [011], K?

        self._V_tensor[2, 1, Q] = 1 - 3 * q # Action, Bob's tree, [01], QJ
        self._V_tensor[2, 1, K] = -2 # Action, Bob's tree, [01], KJ

    def eval_V(self, node: Node) -> Tuple[Value, ValueChildArray]:
        if isinstance(node, SamplingNode):
            node_type = 1
            card = node.info_set.cards[node.tree_owner]
            num_of_children = 3
        elif node.spawned_tree is None:
            node_type = 0
            card = node.info_set.cards[node.info_set.get_current_player()]
            num_of_children = 2
        else:
            node_type = 2
            card = node.info_set.cards[1 - node.tree_owner]
            num_of_children = 2

        assert card is not None
        x = node.info_set.action_history[-1]
        y = card.value
        V = self._V_tensor[node_type][x][y]
        Vc = np.zeros(num_of_children)

        return V, Vc

    def eval_P(self, node: Node) -> PolicyArray:
        if isinstance(node, SamplingNode):
            card = node.info_set.cards[node.tree_owner]
        else:
            card = node.info_set.cards[node.info_set.get_current_player()]

        assert card is not None
        x = node.info_set.action_history[-1]
        y = card.value
        P = self._P_tensor[x][y]

        return P

    def eval_H(self, node: Node) -> HiddenArray:
        if node.info_set.action_history == [PASS, ADD_CHIP, ADD_CHIP] and node.info_set.cards == [Card.KING, None]:
            pass

        info_set = node.info_set
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
        return H

class TensorModel(Model):
    def __init__(self, vmodel: NNModel, pmodel: NNModel):
        self.vmodel = vmodel
        self.pmodel = pmodel

    def eval_V(self, node) -> Tuple[Value, ValueChildArray]:
        if isinstance(node, SamplingNode):
            x = node.info_set.to_sampling_tensor()
            num_of_children = 3
        elif node.spawned_tree is None:
            x = node.info_set.to_action_tensor()
            num_of_children = 2
        else:
            x = node.info_set.to_spawned_tensor()
            num_of_children = 2

        V = self.vmodel(x).detach().numpy()[0]
        if node.info_set.get_current_player() != node.tree_owner:
            V = -V

        Vc = np.zeros(num_of_children)

        return V, Vc

    def eval_P(self, node) -> PolicyArray:
        assert isinstance(node, ActionNode)
        cards = node.info_set.cards
        cp = node.info_set.get_current_player()
        action_history = node.info_set.action_history

        if action_history == []:
            return np.array([1.0, 0.0])
        elif action_history[-1] == PASS and cards[cp] == Card.QUEEN:
            return np.array([1.0, 0.0])
        elif action_history[-1] == ADD_CHIP and cards[cp] == Card.JACK:
            return np.array([1.0, 0.0])
        elif action_history[-1] == PASS and cards[cp] == Card.JACK:
            return np.array([2/3, 1/3])
        elif action_history[-1] == ADD_CHIP and cards[cp] == Card.QUEEN:
            return np.array([2/3, 1/3])
        elif cards[cp] == Card.KING:
            return np.array([0.0, 1.0])

        x = node.info_set.to_action_info_set().to_tensor()
        prob = self.pmodel(x).detach().numpy()[0]

        return np.array([1 - prob, prob])

    def eval_H(self, node) -> HiddenArray:
        assert isinstance(node, SamplingNode)
        cp = node.info_set.get_current_player()

        H = np.ones(3)
        for k in range(len(node.info_set.action_history) - 1):
            temp_info_set = node.info_set.clone()
            temp_info_set.action_history = node.info_set.action_history[:k]
            action = node.info_set.action_history[k]
            temp_cp = temp_info_set.get_current_player()

            if temp_cp != cp:
                continue

            for card in Card:
                temp_info_set.cards = [None, None]
                temp_info_set.cards[cp] = card
                probs = self.eval_P(ActionNode(temp_info_set))
                H[card.value] *= probs[action]

        tree_owner_card = node.info_set.cards[node.tree_owner]
        H[tree_owner_card.value] = 0
        H /= np.sum(H)
        return H

class InfoSetGenerator:
    def __init__(self):
        pass

    def __call__(self):
        info_set = KuhnPokerInfoSet([PASS])

        deck  = list(Card)
        random.shuffle(deck)
        cards = deck[:2]

        info_set.cards = cards
        return info_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    parser.add_argument("--player", type=str, help="Alice or Bob")
    parser.add_argument("--iter", type=int, help="Number of iterations")
    parser.add_argument("--eps", type=float, help="Range parameter for Q-value uncertainty")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--savetrees", action='store_true', help="Save visited trees")
    parser.add_argument("--alpha_num", nargs='+', help="run alpha zero loop with num_gen and num_games_per_gen")
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
        Tree.visit_counter = TreeVisitCounter()

    if args.player == 'Alice':
        info_set = KuhnPokerInfoSet([PASS, ADD_CHIP], [Card.QUEEN, None])
    elif args.player == 'Bob':
        info_set = KuhnPokerInfoSet([PASS], [None, Card.JACK])

    if args.alpha_num is not None:
        vmodel = NNModel(6, 64, 1)
        pmodel = NNModel(5, 64, 1, last_activation=torch.nn.Sigmoid())
        model = TensorModel(vmodel, pmodel)

        # vmodel = torch.load('model/vmodel-1023.pt')
        # pmodel = torch.load('model/pmodel-1023.pt')
        # model = TensorModel(vmodel, pmodel)

        # with open('self_play_games/self_play_games.pkl', 'rb') as f:
        #     games = pickle.load(f)

        num_gen = int(args.alpha_num[0])
        num_gen_games = int(args.alpha_num[1])

        alpha_zero = AlphaZero(model, iter=args.iter)
        alpha_zero.run(InfoSetGenerator(), num_gen, num_gen_games, gen_start_num=0, lookback=0)

    else:
        vmodel = torch.load('model/vmodel-255.pt')
        pmodel = torch.load('model/pmodel-255.pt')
        model = TensorModel(vmodel, pmodel)

        # vmodel = NNModel(6, 64, 1)
        # pmodel = NNModel(5, 64, 1, last_activation=torch.nn.Sigmoid())
        # model = TensorModel(vmodel, pmodel)

        # model = KuhnPokerModel(0.01, 0.01)

        # info_set = KuhnPokerInfoSet([PASS], [None, Card.QUEEN])
        root = ActionNode(info_set)
        mcts = Tree(model, root)
        try:
            visit_dist = mcts.get_visit_distribution(args.iter)
            print(visit_dist)
        except:
            print('Error in get_visit_distribution')

        if Tree.visit_counter is not None:
            Tree.visit_counter.save_snapshots('debug')
