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

# %%
xs_same = np.random.randn(2, n_dims) / np.sqrt(n_dims)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, n_dims) / np.sqrt(n_dims)

xs = np.stack([xs_same, xs_diff])

out, feats = state.apply_fn({'params': state.params}, xs, mutable='intermediates')
# out, feats = state.apply_fn({'params': state.params}, xs, capture_intermediates=True)
# feats
print(out)

np.round(feats['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['attention_weights'], decimals=3)

# %%
xs_same = np.random.randn(2, 512) / np.sqrt(512)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, 512) / np.sqrt(512)

# M = np.random.randn(32, 32) / np.sqrt(32)
b = np.random.randn(512) / np.sqrt(512) * 0
M = m[0]
# b = np.ones(32) * 1000
# b = np.ones(32) * 100

# xs_same @ M @ xs_same.T

xs_diff @ M @ xs_diff.T + (xs_diff[0] @ M @ b + xs_diff[1] @ M.T @ b)

# %%
xs_same = np.random.randn(2, n_dims) / np.sqrt(n_dims)
xs_same[0] = xs_same[1]
xs_diff = np.random.randn(2, n_dims) / np.sqrt(n_dims)

W = state.params['Dense_0']['kernel']
xs_same = xs_same @ W + state.params['Dense_0']['bias']
xs_diff = xs_diff @ W + state.params['Dense_0']['bias']

# xs_same = xs_same @ W
# xs_diff = xs_diff @ W

jax.tree.map(np.shape, state.params)
k = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['key']['kernel']
q = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['query']['kernel']

m = np.einsum('khe,qhe->hqk', k, q)

xs_diff @ m[0] @ xs_diff.T
jax.nn.softmax(xs_diff @ m[0] @ xs_diff.T)
print(xs_diff @ m[0] @ xs_diff.T)
print('--')

# plt.hist(state.params['Dense_0']['bias'])

b = state.params['Dense_0']['bias']
b = b.reshape(-1, 1)
m[1].T @ b
print(xs_same[:1] @ m[0].T @ b)
print(xs_same[:1] @ m[0] @ b)
print('--')
print(xs_same[:1] @ m[1].T @ b)
print(xs_same[:1] @ m[1] @ b)

# <codecell>
# The two heads have vectors that are perfectly anti-aligned
a = m[0].T @ b
b = m[1].T @ b

a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

a.T @ b

# TODO: confirm parallel / antiparallel direction dictates classification
# %%
xs = feats['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['inputs'][0]

xs_same = xs[1]

xs_same @ m[1] @ xs_same.T

# <codecell>
a = np.random.randn(512, 1)
M_r = np.random.randn(512, 512)

print(np.sum(m[0] @ a))
print(np.sum(m[0].T @ a))
np.trace(m[0])


# <codecell>
### COORDINATE CHECKING
all_norms = []

for n_hidden in [16, 64, 256, 1024]:
    n_points = 8
    n_dims = 128
    base_lr = 0.01
    # n_hidden = 512

    gamma0 = 1
    gamma = gamma0 * np.sqrt(n_hidden)
    lr = gamma0**2 * base_lr

    train_task = SameDifferent(n_dims=n_dims, n_symbols=n_points, seed=None, reset_rng_for_data=True)
    test_task = SameDifferent(n_dims=n_dims, n_symbols=None, seed=None, reset_rng_for_data=True, batch_size=1024)

    config = MlpConfig(mup_scale=True,
                    n_out=1, 
                    vocab_size=None, 
                    n_layers=1, 
                    n_hidden=n_hidden, 
                    use_bias=False,
                    act_fn='relu')


    state, hist = train(config,
                        data_iter=iter(train_task), 
                        test_iter=iter(test_task), 
                        loss='bce',
                        gamma=gamma,
                        test_every=1000,
                        train_iters=5, 
                        optim=optax.sgd,
                        lr=lr,
                        seed=None)

    xs, _ = next(train_task)
    xs = xs.reshape(xs.shape[0], -1)

    jax.tree.map(jnp.shape, state.params)

    W = state.params['Dense_0']['kernel']
    a = state.params['Dense_1']['kernel']
    
    act = xs @ W
    norm = np.mean(np.linalg.norm(act, axis=1))
    all_norms.append(norm)

# <codecell>
plt.plot(all_norms)
# %%
