import random
import numpy as np
from abc import abstractmethod, ABCMeta
from collections import defaultdict

from blackjack_envs import Action


class IStrategy:
    __metaclass__ = ABCMeta

    @abstractmethod
    def next_step(self, state):
        """Return action for state"""


class SimpleStrategy(IStrategy):
    def next_step(self, state):
        user_sum, dealer_sum, usable_ace = state
        if user_sum in (19, 20, 21):
            return Action.STICK.value
        return Action.HIT.value


class QStrategy(IStrategy):
    def __init__(self, epsilon, actions_count):
        self.actions_count = actions_count
        self.epsilon = epsilon
        mean = 1.0  # optimistic for exploration
        # mean = (1.0 - self.epsilon) / self.actions_count
        std = 0.1
        self.q = defaultdict(lambda: np.random.normal(mean, std))

    def pi(self, state, action):
        return self.epsilon / self.actions_count + self.q[(state, action)]

    def actions_proba(self, state):
        actions = list(self.possible_actions())
        weights = []
        for a in actions:
            weights.append(self.pi(state, a))
        return actions, np.exp(weights) / np.sum(np.exp(weights))

    def next_step(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions_count - 1)
        else:
            actions, weights = self.actions_proba(state)
            return random.choices(actions, weights)[0]

    def copy(self):
        c = self.__class__(self.epsilon, self.actions_count)
        c.q = self.q.copy()
        return c

    def get_played_states(self):
        return set(st for st, a in self.q)

    def possible_actions(self):
        return range(self.actions_count)


class DiscreteQStrategy(QStrategy):
    def __init__(self, actions_count):
        super(DiscreteQStrategy, self).__init__(epsilon=0, actions_count=actions_count)

    def pi(self, state, action):
        if action == self.next_step(state):
            return 1
        else:
            return 0

    def next_step(self, state):
        best_action, best_value = -1, -1e6
        for a in range(self.actions_count):
            if self.q[(state, a)] > best_value:
                best_action = a
                best_value = self.q[(state, a)]
        return best_action

    def next_step_mild(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.actions_count - 1)
        else:
            return self.next_step(state)

    def copy(self):
        c = self.__class__(self.actions_count)
        c.q = self.q.copy()
        return c
