import numpy as np


def monte_carlo_control_on_policy_one_episode(env, strategy):
    states = [env.reset()]
    while True:
        action = strategy.next_step(states[-1])
        s, reward, is_finished = env.step(action)
        states.append(s)
        if is_finished:
            return states, reward


def update_v(v, states, reward, gamma):
    gain = reward
    for s, _, _ in reversed(states):
        prev_v, count = v[s]
        new_v = prev_v + (gain - prev_v) / (count + 1)
        v[s] = (new_v, count + 1)
        gain *= gamma


def init_v(observation_space):
    return [(v, 0) for v in np.random.normal(loc=0, scale=0.5, size=observation_space)]
