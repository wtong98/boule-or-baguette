"""Can our models learn the graph task with CoT?"""


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
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 32

seed = new_seed()

train_task = Chain(
    # BinaryTreeTiTask(order='fwd', depth=depth, samp_dist=1, on_branch=True),
    # BinaryTreeTiTask(order='rev', depth=depth, samp_dist=1, on_branch=True),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, cot=True, unwrap=True, shuffle=False, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=False, cot=True, unwrap=True, shuffle=False, batch_size=batch_size), sub_samp=False)

test_task = BinaryTreeTiTask(depth=depth, samp_dist=8, on_branch=True, cot=True, unwrap=True, shuffle=False, batch_size=batch_size)

# config = MlpConfig(mup_scale=False,
#                    n_out=n_vocab, 
#                    n_vocab=n_vocab, 
#                    n_layers=1, 
#                    n_hidden=n_hidden, 
#                    use_bias=False,
#                    freeze_emb=True,
#                    act_fn='relu')


config = MixerConfig(n_layers=2, 
                     n_vocab=n_vocab, 
                     n_out=n_vocab,
                     layer_norm=False,
                     n_hidden=n_hidden, 
                     n_channels=32)

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
                    loss='ce',
                    test_every=1000,
                    train_iters=25_000,
                    lr=1e-3
                    # gamma=gamma,
                    # optim=optax.sgd,
                    # lr=lr,
                    )

# <codecell>
xs = np.array([[10, 126, 0, 126, 63, 31, 15, 7, -3, -3, -3, -3, -3, -3]]) + 3

logits = state.apply_fn({'params': state.params}, xs)
np.argmax(logits) - 3


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
