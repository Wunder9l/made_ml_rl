import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
import my_env, agents
from tqdm import trange
from importlib import reload
from collections import defaultdict
from plotly import graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import choice

import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
reload(my_env), reload(agents)
N_ROWS, N_COLS, N_WINS = 3, 3, 3
env = my_env.TicTacToe(N_ROWS, N_COLS)
counts = defaultdict(int)
# agent = agents.QStrategy(0, N_ROWS, N_COLS)
agent = agents.DuelingDQN(env.n_cols * env.n_rows, device, lr=0.001)
buffer = utils.ReplayBuffer()
utils.duelling_train(epochs=1, episodes_per_epoch=10, buffer_size=100, env=env, agent=agent, epsilon=0.1, gamma=0.95, batch_size=10, trains_per_epoch=4)
# utils.dqn_play_one_episode(env, agent, epsilon=0.1, buffer=buffer, gamma=0.95)
# samples = buffer.get_batch(10)
# print(samples)

# utils.q_learning_train_one_episode(env, agent, epsilon=0.1, gamma=0.95, alpha=0.1, counts=counts, stats=utils.Stats())
# agents.play_one_episode(env, agent, agents.RandomAgent())