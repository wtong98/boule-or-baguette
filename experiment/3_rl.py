"""Experimenting with RL finetuning"""


# <codecell>
from pathlib import Path
import pickle

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
from model.transformer import TransformerConfig, generate
from task.graph import *

# <codecell>
depth = 10
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 32
unwrap = False

seed = new_seed()

train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=False, cot=True, unwrap=unwrap, batch_size=batch_size))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=8, on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size)

config = TransformerConfig(n_layers=3,
                           n_vocab=n_vocab,
                           n_out=n_vocab,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=2,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           use_simple_att=False,
                           freeze_emb=False,
                           use_bias=True,
                           return_final_logits_only=False,
                           )
# <codecell>

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=3_000,
                    lr=1e-3,
                    use_tqdm=False
                    )


# <codecell>
with open('state.pkl', 'wb') as fp:
    pickle.dump(state.params, fp)

# <codecell>
with open('state.pkl', 'rb') as fp:
    params = pickle.load(fp)

state = create_train_state(model=config.to_model(), params=params)

### RL finetune
reinforce(state, train_task, 
          action_fn=generate, 
          reward_fn=bt_rew_fn, 
          rl_loss=bt_rl_loss,
          train_iters=500,
          test_every=10,
          use_tqdm=True)


