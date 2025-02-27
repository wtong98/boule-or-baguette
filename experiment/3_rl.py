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
from model.transformer import TransformerConfig 
from task.graph import *


# @jax.jit
# def ans_acc(logits, labels):
#     ys = labels[:,2:]
#     ans_idx = jnp.sum(ys != 0, axis=-1) - 1
#     ans = ys[jnp.arange(len(ys)), ans_idx]

#     toks = logits.argmax(-1)
#     preds = toks[jnp.arange((len(ys))), ans_idx + 2]

#     return np.mean(ans == preds)


# @functools.partial(jax.jit, static_argnames=('loss'))
# def gen_acc_rl(state, batch, loss=None):
#     xs, ys = batch

#     traj = generate(state, xs)
#     preds = traj[:,-1]

#     return {'gen_acc': jnp.mean(preds == ys)}


@jax.jit
def final_acc(logits, labels):
    preds = logits[:,-1,:].argmax(-1)
    return np.mean(labels == preds)


depth = 10
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 32
unwrap = False

seed = new_seed()


train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=False, fill_gaps=False, cot=True, unwrap=unwrap, batch_size=batch_size))

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
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=10_000,
                    # lr=1e-3,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen
                    )


# <codecell>
with open('state.pkl', 'wb') as fp:
    pickle.dump(state.params, fp)

# <codecell>
with open('state.pkl', 'rb') as fp:
    params = pickle.load(fp)

state = create_train_state(
    model=config.to_model(), 
    params=params,
    optim=optax.sgd,
    lr=1e-6
    )

# <codecell>
train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=depth+1),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=False, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=depth+1))

### RL finetune
state, hist = reinforce(state, train_task, 
                        action_fn=generate, 
                        reward_fn=bt_rew_fn_with_punish, 
                        rl_loss=bt_rl_loss,
                        train_iters=20_000,
                        test_every=1000,
                        test_iters=10,
                        use_tqdm=True,
                        eval_fns=[gen_acc_rl]
                        )



# <codecell>
task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(8), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=depth+1),
    BinaryTreeTiTask(depth=depth, samp_dist=(8), on_branch=False, fill_gaps=False, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=depth+1))

batch = next(task)

gen_acc_rl(state, batch)


# <codecell>
for t in task.tasks:
    t.cot = True

xs, ys = next(task)
traj = generate(state, xs)

print(traj[:3])
print(ys[:3])
