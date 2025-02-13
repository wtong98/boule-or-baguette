"""Can our models learn the graph task?"""


# <codecell>
from pathlib import Path

from flax import traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.mlp import MlpConfig, MixerConfig
from model.transformer import TransformerConfig, SimpleTransformerConfig
from task.graph import GraphTiTask 

# <codecell>
n_nodes = 2**8 - 1
n_dims = 128
n_hidden = 512

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
base_lr = 10
lr = gamma0**2 * base_lr

seed = new_seed()

train_task = GraphTiTask(n_nodes, seed=seed, n_dims=n_dims, samp_adj=True)
test_task =  GraphTiTask(n_nodes, seed=seed, n_dims=n_dims, samp_adj=False)

# config = MlpConfig(mup_scale=True,
#                    n_out=1, 
#                    vocab_size=None, 
#                    n_layers=1, 
#                    n_hidden=n_hidden, 
#                    use_bias=False,
#                    act_fn='relu')


# config = MixerConfig(n_layers=2, layer_norm=True, act_fn='linear')

config = TransformerConfig(n_layers=8,
                           n_hidden=128,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=2,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           use_simple_att=False,
                           freeze_emb=False)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=20_000, 
                    # gamma=gamma,
                    # optim=optax.sgd,
                    # lr=lr,
                    seed=None)

# <codecell>
xs, ys = next(test_task)
out = state.apply_fn({'params': state.params}, xs)
preds = (out > 0).astype(bool)

print(np.mean(ys[ys>0] == preds[ys>0]))
print(np.mean(ys[ys==0] == preds[ys==0]))

print('---')
print(np.mean(ys[preds>0] == preds[preds>0]))
print(np.mean(ys[preds==0] == preds[preds==0]))

print('--')
print(np.mean(ys))
print(np.mean(preds))
