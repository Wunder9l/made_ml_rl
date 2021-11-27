from collections import deque

import numpy as np
import torch
from tqdm import trange

import agents
from agents import QStrategy, DuelingDQN
from my_env import TicTacToe

BUFFER_SIZE = int(1e4)

class Stats:
    def __init__(self):
        self.q_diff = 0.0
        self.wrong_actions = 0

    def update(self, q_diff: float, wrong_action: bool):
        self.q_diff += abs(q_diff)
        self.wrong_actions += int(wrong_action)


def q_learning_train_one_episode(env, player: QStrategy, epsilon, gamma, alpha, counts, stats: Stats = None):
    (s, possible_actions, turn) = env.reset()
    mild_player = player.make_mild(epsilon)
    prev_stage = None
    is_finished = False
    while not is_finished:
        a = mild_player.next_step(s)
        counts[(s, a)] += 1
        rel = player.pi(s, a) / mild_player.pi(s, a)
        (next_s, next_possible_actions, next_turn), reward, is_finished, _ = env.step(a)
        if env.is_wrong_action(reward):
            q_diff = player.update_q(s, a, alpha, rel, reward, gamma=0, next_state=next_s)
            if stats is not None:
                stats.update(q_diff, wrong_action=True)
        elif is_finished:
            q_diff = player.update_q(s, a, alpha, rel, reward, gamma=0, next_state=next_s)
            if stats is not None:
                stats.update(q_diff, wrong_action=False)
            if prev_stage:
                prev_s, prev_a, prev_rel = prev_stage
                q_diff = player.update_q(prev_s, prev_a, alpha, prev_rel, -reward, gamma=0, next_state=next_s)
                if stats is not None:
                    stats.update(q_diff, wrong_action=False)
        else:
            if prev_stage:
                prev_s, prev_a, prev_rel = prev_stage
                q_diff = player.update_q(prev_s, prev_a, alpha, prev_rel, -reward, gamma, next_state=next_s)
                if stats is not None:
                    stats.update(q_diff, wrong_action=False)
            prev_stage = s, a, rel
        s = next_s


class ReplayBuffer:
    def __init__(self, size, prioritize):
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.priorities_enabled = prioritize

    def save(self, state, action, is_finished, reward, next_state, target, q_value):
        self.buffer.append((state, action, is_finished, reward, next_state, target))
        self.priorities.append(abs((target - q_value).item()))

    def get_batch(self, batch_size):
        if self.priorities_enabled:
            weights = np.array(self.priorities)
            weights += weights.mean()
            weights /= sum(weights)
        else:
            weights = None
        indices = np.random.choice(len(self.buffer), size=batch_size, p=weights)
        states = []
        actions = []
        next_states = []
        rewards = []
        targets = []
        dones = []
        for idx in indices:
            state, action, is_finished, reward, next_state, target = self.buffer[idx]
            # print(f'reward={reward}, target={target}, is_finished={is_finished}')
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            targets.append(target)
            dones.append(is_finished)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).reshape(batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(batch_size, 1)
        targets = torch.tensor(targets, dtype=torch.float32).reshape(batch_size, 1)
        dones = torch.tensor(dones, dtype=torch.int32).reshape(batch_size, 1)
        return states, actions, dones, rewards, next_states, targets


def dqn_play_one_episode(env: TicTacToe, player: DuelingDQN, epsilon, buffer: ReplayBuffer, gamma, stats: Stats = None):
    env.reset()
    s = env.getPlayerBoard().reshape(1, env.n_rows, env.n_cols)
    prev_stage = None
    is_finished = False
    possible_actions = np.arange(player.actions_count)
    while not is_finished:
        with torch.no_grad():
            s_torch = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            q_values = player.forward(s_torch).cpu().squeeze()
            weights = torch.softmax(q_values, -1)
            weights = (1 - epsilon) * weights + epsilon / player.actions_count
            a = np.random.choice(possible_actions, p=weights.numpy())
            q_value = q_values[a]
        _, reward, is_finished, _ = env.step(env.action_from_int(a))
        next_s = env.getPlayerBoard().reshape(1, env.n_rows, env.n_cols)
        if env.is_wrong_action(reward):
            buffer.save(s, a, is_finished, reward, next_state=next_s, target=reward, q_value=q_value)
            if stats is not None:
                stats.update(reward - q_value, wrong_action=True)
        elif is_finished:
            buffer.save(s, a, is_finished, reward, next_state=next_s, target=reward, q_value=q_value)
            if prev_stage:
                prev_s, prev_a, prev_q, prev_r = prev_stage
                buffer.save(prev_s, prev_a, is_finished, -reward, next_state=next_s, target=-reward, q_value=prev_q)
                if stats is not None:
                    stats.update(-reward - prev_q, wrong_action=False)
        else:
            if prev_stage:
                prev_s, prev_a, prev_q, prev_r = prev_stage
                with torch.no_grad():
                    next_s_torch = torch.tensor(next_s, dtype=torch.float32).unsqueeze(0)
                    next_q = player.forward(next_s_torch).cpu().squeeze().max()
                buffer.save(prev_s, prev_a, is_finished, prev_r, next_s, target=prev_r + gamma * next_q, q_value=prev_q)
                if stats is not None:
                    stats.update(prev_r + gamma * next_q - prev_q, wrong_action=False)
            prev_stage = s, a, q_value, reward
        s = next_s


class DQNStats:
    def __init__(self):
        self.targets = []
        self.rewards = []
        self.dones = []
        self.wrong = []
        self.wins = []
        self.looses = []

    def update(self, states, actions, is_done, rewards, next_states, targets):
        self.targets.append(float(targets.mean()))
        self.rewards.append(float(rewards.mean()))
        self.dones.append(int(is_done.sum()))
        self.wrong.append(int((rewards < -5).sum()))
        self.wins.append(int((rewards > 0.5).sum()))
        self.looses.append(int(((rewards < -.5) & (rewards > -2)).sum()))

    def plot(self):
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame({
            'epoch': list(range(len(self.targets))),
            'reward': self.rewards,
            'target': self.targets,
            'dones': self.dones,
            'wrong': self.wrong,
            'wins': self.wins,
            'looses': self.looses,
        })
        px.line(df, x='epoch', y=list(set(df.columns) - {'epoch'})).show()


def duelling_train(epochs, episodes_per_epoch, buffer_size, env, agent: agents.BaseDQN,
                   epsilon, gamma, batch_size, trains_per_epoch, snapshots_number, snapshot_games,
                   prioritize=False, do_plot=False):
    snapshot_each = epochs // snapshots_number if snapshots_number else int(1e12)
    snapshots = {}
    stats = Stats()
    new_stats = DQNStats()
    for i in trange(epochs):
        buffer = ReplayBuffer(buffer_size, prioritize)
        for _ in range(episodes_per_epoch):
            dqn_play_one_episode(env, agent, epsilon, buffer, gamma, stats)
        for _ in range(trains_per_epoch):
            states, actions, is_done, rewards, next_states, targets = buffer.get_batch(batch_size)
            if do_plot:
                new_stats.update(states, actions, is_done, rewards, next_states, targets)
            agent.update_q(states, actions, is_done, rewards, gamma, next_states)
            # agent.update_q_outside_target(states, actions, targets)
        if i % snapshot_each == 0 and snapshots_number:
            wins = 0
            player = agents.DQNPlayerWrapper(env, agent)
            opponent = agents.RandomAgent()
            for _ in range(snapshot_games):
                winner = play_one_episode(env, player, opponent)
                wins += int(winner == player)
            snapshots[i] = {
                'q_diff': stats.q_diff,
                'wrong_actions': stats.wrong_actions,
                'wins_ratio': wins / snapshot_games,
            }
            stats = Stats()
    if do_plot:
        new_stats.plot()
    return snapshots


def play_one_episode(env, player_a, player_b):
    is_finished = False
    (s, actions, _) = env.reset()
    players = {
        -1: player_a,
        1: player_b
    }
    while not is_finished:
        player = players[env.curTurn]
        s, actions = env.getHash(), env.getEmptySpaces()
        a = player.best_next_step(s, actions)
        (s, actions, _), reward, is_finished, _ = env.step(a)
    if reward > 0:
        return player
    elif reward < 0:
        return player_a if player_b == player else player_b
    else:
        return None
