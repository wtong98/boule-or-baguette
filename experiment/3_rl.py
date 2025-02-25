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


# @jax.jit
# def ans_acc(logits, labels):
#     ys = labels[:,2:]
#     ans_idx = jnp.sum(ys != 0, axis=-1) - 1
#     ans = ys[jnp.arange(len(ys)), ans_idx]

#     toks = logits.argmax(-1)
#     preds = toks[jnp.arange((len(ys))), ans_idx + 2]

#     return np.mean(ans == preds)


@functools.partial(jax.jit, static_argnames=('loss'))
def gen_acc_cot(state, batch, loss=None):
    xs, ys = batch
    ys = ys[:,2:]
    ans_idx = jnp.sum(ys != 0, axis=-1) - 1
    ans = ys[jnp.arange(len(ys)), ans_idx]

    traj = generate(state, xs)
    preds = traj[jnp.arange(len(ys)), ans_idx + 3]

    return {'gen_acc': jnp.mean(ans == preds)}


def print_gen(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1]["loss"]:.4f}   train_acc={hist["train"][-1]["gen_acc"]:.4f}   test_loss={hist["test"][-1]["loss"]:.4f}   test_acc={hist["test"][-1]["gen_acc"]:.4f}')


@jax.jit
def final_acc(logits, labels):
    preds = logits[:,-1,:].argmax(-1)
    return np.mean(labels == preds)


def gen_acc_rl(state, batch, loss=None):
    pass


depth = 10
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 32
unwrap = False

seed = new_seed()


train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1,3), on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1,3), on_branch=False, fill_gaps=False, cot=True, unwrap=unwrap, batch_size=batch_size))

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
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False,
                           )


state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=5_000,
                    lr=1e-3,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=_print
                    )


# <codecell>
batch = next(test_task)
xs, ys = batch

print(loss_and_acc(state, batch, loss='ce_mask'))
print(gen_acc_cot(state, batch))


# <codecell>
with open('state.pkl', 'wb') as fp:
    pickle.dump(state.params, fp)

# <codecell>
with open('state.pkl', 'rb') as fp:
    params = pickle.load(fp)

state = create_train_state(model=config.to_model(), params=params)

# <codecell>
train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=False, rl_prompt=True, unwrap=unwrap, batch_size=batch_size))

### RL finetune
reinforce(state, train_task, 
          action_fn=generate, 
          reward_fn=bt_rew_fn, 
          rl_loss=bt_rl_loss,
          train_iters=1000,
          test_every=50,
          test_iters=3,
          use_tqdm=False,
          loss=final_acc)



# <codecell>
xs, ys = next(train_task)

traj = generate(state, xs)

np.mean(traj[:,-1] == ys)


