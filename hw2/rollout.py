from copy import deepcopy

from agents import IStrategy
from my_env import TicTacToe
from dataclasses import dataclass
import numpy as np

EPSILON = 1e-8


def norm(weights: np.array):
    if weights.min() < EPSILON:
        weights += (-weights.min() + 2 * EPSILON)
    weights = weights / weights.sum()
    return weights


@dataclass
class RolloutStat:
    games: int = 0
    wins: int = 0
    defeats: int = 0

    def __init__(self, alpha=0.0, initial_weight=0):
        self.initial_weight = initial_weight
        self.alpha = alpha

    def update(self, win, defeat, games=1):
        self.wins += win
        self.defeats += defeat
        self.games += games

    def get_ucb(self, total, c):
        games = self.games + EPSILON
        total += EPSILON
        expected_value = self.wins / games - self.defeats / games
        total_value = self.alpha * self.initial_weight + (1-self.alpha) * expected_value
        ucb = total_value + c * np.log(total / games)
        return ucb

    def clone_to(self, alpha, initial_weight):
        clone = RolloutStat(alpha, initial_weight)
        clone.update(self.wins, self.defeats, self.games)
        return clone


class Rollout:
    def __init__(self, env: TicTacToe, agent):
        self.env = env
        self.agent = agent
        self.possible_actions = env.getEmptySpaces()
        self.actions_cnt = len(self.possible_actions)
        if not self.actions_cnt:
            raise RuntimeError("Can't start rollout in terminal state")

    def make_rollouts(self, cnt, ucb_c):
        stats = [RolloutStat() for _ in range(self.actions_cnt)]
        assert cnt > self.actions_cnt
        total = 0
        for i, a in enumerate(self.possible_actions):
            win, defeat = self._simulate(a)
            stats[i].update(win, defeat)
            total += 1
        for _ in range(cnt - self.actions_cnt):
            probs = self.get_probs(stats, total=total, ucb_c=ucb_c)
            action_idx = np.random.choice(self.actions_cnt, p=probs)
            win, defeat = self._simulate(self.possible_actions[action_idx])
            stats[action_idx].update(win, defeat)
            total += 1
        return self.possible_actions, self.get_probs(stats, total=total, ucb_c=ucb_c), stats

    @staticmethod
    def get_probs(stats, total, ucb_c):
        probs = np.array([st.get_ucb(total=total, c=ucb_c) for st in stats])
        return norm(probs)

    def _simulate(self, action):
        env = deepcopy(self.env)
        turn = env.curTurn
        while True:
            (s, possible_actions, _), reward, is_finished, _ = env.step(action)
            if is_finished:
                break
            actions, weights = self.agent.get_proba(s, possible_actions)
            action_idx = np.random.choice(len(actions), p=norm(weights))
            action = actions[action_idx]
        if reward > 0:
            is_win = turn == env.curTurn
            return (1, 0) if is_win else (0, 1)
        elif reward < 0:
            is_win = turn != env.curTurn
            return (1, 0) if is_win else (0, 1)
        else:
            return (0, 0)


class RolloutPlayer(IStrategy):
    def __init__(self, env: TicTacToe, agent, rollouts_per_state, ucb_c, is_stochastic):
        self.is_stochastic = is_stochastic
        self.ucb_c = ucb_c
        self.rollouts_per_state = rollouts_per_state
        self.env = env
        self.agent = agent

    def best_next_step(self, s, actions):
        rollout = Rollout(self.env, self.agent)
        actions, probs, _ = rollout.make_rollouts(self.rollouts_per_state, self.ucb_c)
        if self.is_stochastic:
            idx = np.random.choice(rollout.actions_cnt, p=probs)
            return actions[idx]
        else:
            return actions[np.argmax(probs)]
