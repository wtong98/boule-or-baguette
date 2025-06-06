"""Max margin heuristics"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import *
from task.graph import *

n_depth = 4
n_dist = 3

v = 2 * n_depth + 4

M1 = np.zeros((v**2, v**2))

beta = 10

# TODO: continue encoding entries <-- STOPPED HERE
bb = v * beta + beta
M1[bb, (v*beta):((beta+1)*v)] = -1
M1[bb, v * beta + beta - 2] = 1
print(M1[bb])

idx = np.argmax(M1[bb])
print(idx // v)
print(idx % v)

# %%
