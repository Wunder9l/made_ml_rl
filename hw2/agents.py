import random
from itertools import product

import numpy as np
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import torch
from torch import nn
from torchvision import models


class IStrategy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def next_step(self, state):
        """Return action for state"""

    @abstractmethod
    def best_next_step(self, state, possible_actions):
        """Return action for state"""


class QStrategy(IStrategy):
    def __init__(self, epsilon, rows, cols):
        self.rows = rows
        self.cols = cols
        self.actions_count = rows * cols
        self.epsilon = epsilon
        mean = 1.0  # optimistic for exploration
        std = 0.1
        self.q = defaultdict(lambda: np.random.normal(mean, std))

    def pi(self, state, action):
        return self.actions_proba(state)[action]

    def actions_proba(self, state):
        actions = list(self.possible_actions())
        weights = [self.q[(state, a)] for a in actions]
        weights = (1 - self.epsilon) * np.exp(weights) / np.sum(np.exp(weights)) + self.epsilon / self.actions_count
        return {a: w for a, w in zip(actions, weights)}

    def next_step(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(self.possible_actions()))
        else:
            actions_to_weights = self.actions_proba(state)
            return random.choices(list(actions_to_weights.keys()), list(actions_to_weights.values()))[0]

    def best_next_step(self, state, possible_actions=None):
        actions_to_weights = self.actions_proba(state)
        return max(actions_to_weights, key=actions_to_weights.get)

    def copy(self):
        c = self.__class__(self.epsilon, self.rows, self.cols)
        c.q = self.q.copy()
        return c

    def make_mild(self, epsilon):
        c = self.__class__(self.epsilon, self.rows, self.cols)
        c.q = self.q
        c.epsilon = epsilon
        return c

    def get_played_states(self):
        return set(st for st, a in self.q)

    def possible_actions(self):
        return product(range(self.rows), range(self.cols))

    def update_q(self, state, action, alpha, rel, reward, gamma, next_state):
        target = reward
        if gamma > 0:
            best_next_action = self.best_next_step(next_state)
            target += gamma * self.q[(next_state, best_next_action)]
        prev_value = self.q[(state, action)]
        self.q[(state, action)] += alpha * rel * (target - self.q[(state, action)])
        return self.q[(state, action)] - prev_value


class RandomAgent(IStrategy):
    def best_next_step(self, s, possible_actions):
        return possible_actions[np.random.randint(len(possible_actions))]


class BaseDQN(nn.Module):
    def __init__(self, output, device, lr):
        super(BaseDQN, self).__init__()
        self.actions_count = output
        self.device = device
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(
            list(self.features_layer.parameters()) + list(self.value_stream.parameters()) + list(self.advantage_stream.parameters()),
            lr=lr, weight_decay=1e-5
        )

    def actions_proba(self, state, epsilon=0) -> torch.Tensor:
        q_values = self.forward(state)
        weights = torch.softmax(q_values, -1)
        return (1 - epsilon) * weights + epsilon / self.actions_count

    def update_q(self, state, action, done, reward, gamma, next_state):
        target = reward.to(self.device)
        with torch.no_grad():
            not_done = (1 - done)
            next_q = self.forward(next_state).max(dim=-1).values.unsqueeze(-1)
            next_q = next_q * not_done
            target += gamma * next_q
        action = action.to(self.device)
        q = self.forward(state).gather(-1, action)
        loss = self.loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_q_outside_target(self, states, actions, targets):
        action = actions.to(self.device)
        q = self.forward(states.to(self.device))
        q = q.gather(-1, action)
        loss = self.loss(q, targets.to(self.device))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


class DQN(BaseDQN):
    def __init__(self, output, device, lr):
        super(DQN, self).__init__( output, device, lr)
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=16*3*3, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        results = self.model(x)
        return results


class DuelingDQN(BaseDQN):
    def __init__(self, output, device, lr):
        super(DuelingDQN, self).__init__(output, device, lr)
        self.features_layer = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(8, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=16*3*3, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        ).to(device)
        self.value_stream = nn.Linear(64, 1).to(device)
        self.advantage_stream = nn.Linear(64, output).to(device)

    def forward(self, x):
        x = x.to(self.device)
        features = self.features_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean())

        return q_vals


class DQNPlayerWrapper:
    def __init__(self, env, player: DuelingDQN):
        self.env = env
        self.player = player

    def _get_predict(self):
        s = self.env.getPlayerBoard()
        s_torch = torch.tensor(s, dtype=torch.float32).reshape(1, 1, self.env.n_rows, self.env.n_cols)
        with torch.no_grad():
            return self.player.forward(s_torch).cpu().squeeze()

    def best_next_step(self, state, possible_actions):
        q_values = self._get_predict()
        action = int(q_values.argmax())
        return self.env.action_from_int(action)

    def actions_proba(self, state):
        q_values = self._get_predict()
        weights = q_values.numpy()
        # weights = torch.softmax(q_values, -1).numpy()
        return {self.env.action_from_int(a): w for a, w in enumerate(weights)}

# class DQNAgent(QStrategy):
#     def __init__(self, epsilon, rows, cols):
#         super(DQNAgent, self).__init__(epsilon, rows, cols)
#         self.model = MyNet(output=rows * cols)
#
#     def _run_model(self, state):
#         return self.model(torch.tensor(state, dtype=torch.float32))
#
#     def actions_proba(self, state):
#         actions = list(self.possible_actions())
#         weights = torch.softmax(self._run_model(state))
#         weights = (1 - self.epsilon) * np.exp(weights) / np.sum(np.exp(weights)) + self.epsilon / self.actions_count
#         return {a: w for a, w in zip(actions, weights)}


# class DuellingTrainer:
#     def __init__(self, agent, critic):