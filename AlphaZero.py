from model import Model
from info_set import InfoSet
from ISMCTS import ActionNode, Tree, Node
from basic_types import ActionDistribution, Value, Action


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import numpy as np
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import pickle

@dataclass
class TurnSnapshot:
    info_set: InfoSet
    action_dist: ActionDistribution
    action: Action
    game_id: int
    gen_id: int
    outcome: Value

Games = List[TurnSnapshot]

class SelfPlayDataV(Dataset):
    def __init__(self, data: List[TurnSnapshot]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        x0 = data.info_set.to_action_tensor()
        x1 = data.info_set.to_sampling_tensor()
        x2 = data.info_set.to_spawned_tensor()
        x = torch.stack([x0, x1, x2], axis=0)
        v = torch.tensor([[data.outcome]]*3, dtype=torch.float32)
        return (x, v)

class SelfPlayDataP(Dataset):
    def __init__(self, data: List[TurnSnapshot]):
        self.data = [d for d in data if d.action_dist is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        x = data.info_set.to_action_info_set().to_tensor()
        p = torch.tensor([data.action_dist[1]], dtype=torch.float32)
        return (x, p)


class NNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, last_activation=None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        self.relu = nn.ReLU()
        self.last_activation = last_activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x

class AlphaZero:
    def __init__(self, model: Model, iter=100):
        self.model = model
        self.iter = iter
        self.self_play_games = []

    def run(self, init_info_set_generator, n_generations=32, n_games_per_gen=256, gen_start_num=0, lookback=1024):
        for gen_id in tqdm(range(gen_start_num, gen_start_num + n_generations)):
            for game_id in range(n_games_per_gen):
                self.generate_one_game(init_info_set_generator, gen_id, game_id)

            data_loader_v = DataLoader(SelfPlayDataV(self.self_play_games[-lookback:]), batch_size=128, shuffle=True)
            data_loader_p = DataLoader(SelfPlayDataP(self.self_play_games[-lookback:]), batch_size=128, shuffle=True)
            self.train(self.model.vmodel, data_loader_v, nn.MSELoss(), num_batches=32, filename=f'model/vmodel-{gen_id}.pt')
            self.train(self.model.pmodel, data_loader_p, nn.MSELoss(), num_batches=32, filename=f'model/pmodel-{gen_id}.pt')

        with open('self_play_games/self_play_games.pkl', 'wb') as f:
            pickle.dump(self.self_play_games, f)


    def generate_one_game(self, init_info_set_generator, gen_id, game_id):
        info_set = init_info_set_generator()
        # tree_owner = info_set.get_current_player()

        game = []
        while info_set.get_game_outcome() is  None:
            player_info_set = info_set.clone()
            cp = player_info_set.get_current_player()
            player_info_set.cards = [None for _ in player_info_set.cards]
            player_info_set.cards[cp] = info_set.cards[cp]

            root = ActionNode(player_info_set)
            mcts = Tree(self.model, root)
            visit_dist = mcts.get_visit_distribution(self.iter)
            probs = np.array(list(visit_dist.values()))
            action = np.random.choice(len(visit_dist), p=probs)

            game.append(TurnSnapshot(info_set, visit_dist, action, game_id, gen_id, None))

            info_set = info_set.apply(action)

        game.append(TurnSnapshot(info_set, None, None, game_id, gen_id, None))
        outcome = info_set.get_game_outcome()
        for g in game:
            g.outcome = outcome[g.info_set.get_current_player()]
            # g.outcome = outcome[tree_owner]
            self.self_play_games.append(g)

    def train(self, model: nn.Module, data_loader: DataLoader, loss_func: nn.Module, lr=1e-2, num_batches=256, filename='model/model.pt'):
        learning_rate = 1e-2
        momentum = 0.9
        weight_decay = 6e-5
        self.opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        for data in data_loader:
            num_batches -= 1
            if num_batches <= 0:
                break
            self.opt.zero_grad()
            data_x, target = data
            hat_target = model(data_x)
            loss = loss_func(hat_target.view(-1, 1), target.view(-1, 1))
            loss.backward()
            self.opt.step()

        torch.save(model, filename)