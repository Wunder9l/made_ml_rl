import collections
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

from agents import IStrategy
from my_env import get_hash, TicTacToe
from rollout import EPSILON, Rollout, RolloutStat, norm


@dataclass
class Node:
    cur_turn: int = None
    state = None
    state_as_string: str = None
    actions: List = None

    def __init__(self, state, cur_turn):
        self.cur_turn = cur_turn
        self.state = state
        self.state_as_string = get_hash(state, cur_turn)
        self.actions = []

    def get_proba(self, ucb_c):
        assert self.actions
        stats = [action.stat for action in self.actions]
        total = sum(st.games for st in stats)
        probs = Rollout.get_probs(stats, total, ucb_c)
        return self.actions, probs

    def select_action(self, ucb_c):
        actions, probs = self.get_proba(ucb_c)
        return np.random.choice(actions, p=probs)

    def merge(self, node):
        assert self.state_as_string == node.state_as_string
        action_map = {a.action: a for a in self.actions}
        for a in node.actions:
            if a.action in action_map:
                action_map[a.action].stat.update(a.stat)
            else:
                self.actions.append(a)


@dataclass
class NodeAction:
    action: Tuple[int, int] = None
    stat: RolloutStat = None
    next_state: Dict[str, Node] = None


def to_list_of_tuple(x):
    return [tuple(y) for y in x]


class MonteCarloTreeSearch(IStrategy):
    def __init__(self, env: TicTacToe, agent: IStrategy, alpha: float, opponent: IStrategy, rollouts_per_update: int, ucb_c: float):
        self.ucb_c = ucb_c
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.alpha = alpha
        self.cur_turn = env.curTurn
        self.rollouts_per_update = rollouts_per_update
        self.root = Node(self.env.board, self.cur_turn)
        self.node_bank = {self.root.state_as_string: self.root}

    def search(self, cnt):
        initial_state, initial_turn = self.env.board, self.env.curTurn
        for i in range(cnt):
            self.env.load(initial_state.copy(), initial_turn)
            is_done, win, defeat, history, node = self._select()
            if is_done:
                self._update(history, self.rollouts_per_update * win, self.rollouts_per_update * defeat, self.rollouts_per_update)
                continue
            wins, defeats = self._make_rollouts(node)
            self._update(history, wins, defeats, self.rollouts_per_update)
        self.env.load(initial_state, initial_turn)

    def get_proba(self, state, _=None):
        assert self.root.state_as_string == get_hash(state, self.cur_turn)
        actions, proba = self.root.get_proba(ucb_c=0.0)
        return [a.action for a in actions], proba

    def best_next_step(self, state, _=None):
        actions, proba = self.get_proba(state, _)
        return actions[np.argmax(proba)]

    def _select(self):
        node = self.root
        history = []
        while node.actions:
            action = node.select_action(self.ucb_c)
            history.append(action)
            is_done, win, defeat = self._simulate_player_and_opponent(action)
            if is_done:
                return is_done, win, defeat, history, None
            state_hash = self.env.getHash()
            if state_hash not in action.next_state:
                node = self.node_bank.get(state_hash, Node(self.env.board.copy(), self.env.curTurn))
                action.next_state[state_hash] = node
            else:
                node = action.next_state[state_hash]
        return False, 0, 0, history, node

    def _simulate_player_and_opponent(self, action_node: NodeAction):
        (_, possible_actions, _), reward, is_done, _ = self.env.step(action_node.action)
        if is_done:
            return is_done, reward > EPSILON, reward < -EPSILON
        actions, proba = self.opponent.get_proba(self.env.getHash(), possible_actions)
        idx = np.random.choice(len(actions), p=norm(proba))
        _, reward, is_done, _ = self.env.step(actions[idx])
        return is_done, reward < -EPSILON, reward > EPSILON

    def _update(self, history: List[NodeAction], win, defeat, games):
        for action_node in history:
            action_node.stat.update(win, defeat, games)

    def _make_rollouts(self, node):
        rollout = Rollout(self.env, self.agent)
        possible_actions, probs, stats = rollout.make_rollouts(self.rollouts_per_update, ucb_c=self.ucb_c)
        rollout_actions = to_list_of_tuple(possible_actions)
        agent_actions, agent_weights = self.agent.get_proba(self.env.getHash(), self.env.getEmptySpaces())
        actions_to_weights = {a: w for a, w in zip(to_list_of_tuple(agent_actions), agent_weights)}
        node.actions = []
        wins, defeats = 0, 0
        for a, st in zip(rollout_actions, stats):
            node.actions.append(NodeAction(action=a, stat=st.clone_to(self.alpha, actions_to_weights[a]), next_state={}))
            wins += st.wins
            defeats += st.defeats
        return wins, defeats

    def find_nodes(self, state_hash):
        return [self.node_bank[state_hash]] if state_hash in self.node_bank else []


class MCTSPlayer(IStrategy):
    def __init__(self, env: TicTacToe, agent, opponent, search_cnt_per_state, alpha, rollouts_per_update, ucb_c):
        self.search_cnt_per_state = search_cnt_per_state
        self.alpha = alpha
        self.opponent = opponent
        self.ucb_c = ucb_c
        self.rollouts_per_update = rollouts_per_update
        self.env = env
        self.agent = agent
        self.cache = {}

    def best_next_step(self, state, actions):
        mcts = self.get_or_make_mcts(self.env.board)
        return mcts.best_next_step(self.env.board, actions)

    def get_proba(self, state, possible_actions=None):
        mcts = self.get_or_make_mcts(self.env.board)
        return mcts.get_proba(self.env.board, possible_actions)

    def invalidate_cache(self):
        del self.cache
        self.cache = {}

    def get_or_make_mcts(self, s):
        state_hash = get_hash(s, self.env.curTurn)
        if state_hash not in self.cache:
            nodes = self.collect_nodes(state_hash)
            self.invalidate_cache()
            mcts = MonteCarloTreeSearch(self.env, self.agent, self.alpha, self.opponent, self.rollouts_per_update,
                                        self.ucb_c)
            for node in nodes:
                mcts.root.merge(node)
            mcts.search(self.search_cnt_per_state)
            self.cache[state_hash] = mcts
        return self.cache[state_hash]

    def collect_nodes(self, target_state_hash):
        nodes = []
        for state_hash, mcts in self.cache.items():
            nodes += mcts.find_nodes(target_state_hash)
        return nodes
