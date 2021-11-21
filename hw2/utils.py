from agents import QStrategy


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
