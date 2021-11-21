import gym
import numpy as np
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


reload(my_env), reload(agents)
N_ROWS, N_COLS = 3, 3
env = my_env.TicTacToe(N_ROWS, N_COLS)
counts = defaultdict(int)
agent = agents.QStrategy(0, N_ROWS, N_COLS)
utils.q_learning_train_one_episode(env, agent, epsilon=0.1, gamma=0.95, alpha=0.1, counts=counts, stats=utils.Stats())
# agents.play_one_episode(env, agent, agents.RandomAgent())