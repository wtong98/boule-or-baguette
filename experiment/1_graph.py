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
from task.graph import * 


depth = 10
n_vocab = 2**depth
n_hidden = 512

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
base_lr = 10
lr = gamma0**2 * base_lr

seed = new_seed()

# train_task = GraphTiTask(n_nodes=10)
# test_task = GraphTiTask(n_nodes=10)

# train_task = Chain([
#     BinaryTreeTiTask(order=None, depth=depth, samp_dist=1, on_branch=True),
#     BinaryTreeTiTask(order='rev', depth=depth, samp_dist=1, on_branch=True),
#     BinaryTreeTiTask(order=None, depth=depth, samp_dist=1, on_branch=False, fill_gaps=False),
#     BinaryTreeTiTask(order='rev', depth=depth, samp_dist=1, on_branch=False, fill_gaps=False),

# ], weights=None)

# test_task = Chain([
#     BinaryTreeTiTask(order=None, depth=depth, samp_dist=3, on_branch=True),
#     BinaryTreeTiTask(order='rev', depth=depth, samp_dist=3, on_branch=True),
# ])

train_task = Chain([
    BinaryTreeTiTask(order='split', depth=depth, samp_dist=(1, 2), on_branch=True),
    BinaryTreeTiTask(order='split', depth=depth, samp_dist=(1, 2), on_branch=False, fill_gaps=False),
])

test_task = Chain([
    BinaryTreeTiTask(order='split', depth=depth, samp_dist=7, on_branch=False),
])


config = MlpConfig(mup_scale=False,
                   n_out=1, 
                   n_vocab=n_vocab, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=True,
                   freeze_emb=True,
                   act_fn='relu')


# config = MixerConfig(n_layers=2, n_vocab=n_vocab, layer_norm=True)

# config = TransformerConfig(n_layers=3,
#                            n_vocab=n_vocab,
#                            n_hidden=128,
#                            pos_emb=False,
#                            n_mlp_layers=2,
#                            n_heads=2,
#                            layer_norm=True,
#                            as_rf_model=False,
#                            residual_connections=True,
#                            use_simple_att=False,
#                            freeze_emb=False)

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='bce',
                    test_every=1000,
                    train_iters=50_000, 
                    # gamma=gamma,
                    # optim=optax.sgd,
                    # lr=lr,
                    )

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
