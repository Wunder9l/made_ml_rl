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
        c = self.copy()
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


class DuelingDQN(nn.Module):
    def __init__(self, output, device, lr):
        super(DuelingDQN, self).__init__()
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
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(
            list(self.features_layer.parameters()) + list(self.value_stream.parameters()) + list(self.advantage_stream.parameters()),
            lr=lr
        )

    def forward(self, x):
        features = self.feauture_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean())

        return q_vals

    def update_q(self, state, action, done, reward, gamma, next_state):
        s = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int32)
        with torch.no_grad():
            next_s = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            not_done = 1 - torch.tensor(done, dtype=torch.int32)
            next_q = self.forward(next_s).max(dim=-1).values * not_done
            target = (torch.tensor(reward, dtype=torch.float32) + gamma * next_q).to(self.device)
        q = self.forward(s).gather(-1, action)
        loss = self.loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


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