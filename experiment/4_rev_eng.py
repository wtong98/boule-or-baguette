"""Reverse engineering the Transformer's solution"""


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


depth = 6
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 512
batch_size = 128

n_layers = 2

seed = new_seed()


train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=True, cot=True, batch_size=batch_size, use_sep=True, repeat_first=True),
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=False, cot=True, batch_size=batch_size, use_sep=True, repeat_first=True))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=4, on_branch=True, cot=True, batch_size=batch_size, use_sep=True, repeat_first=True)


config = TransformerConfig(n_layers=n_layers,
                           n_vocab=n_vocab,
                           n_out=n_vocab,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=False,
                           use_simple_att=False,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False,
                           mup_scale=False
                           )


# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=3_000,
                    # lr=1e-2,
                    optim=optax.adamw,
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
    # optim=optax.sgd,
    lr=3e-5
    )

# <codecell>
batch_size = 128

train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, rl_prompt=True, batch_size=batch_size, n_thought=None),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=True, rl_prompt=True, batch_size=batch_size, n_thought=None))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, rl_prompt=True, batch_size=batch_size, n_thought=None)

### RL finetune
state, hist = reinforce(state, train_task, 
                        test_iter=test_task,
                        action_fn=gen2, 
                        reward_fn=bt_rew_fn, 
                        rl_loss=bt_rl_loss,
                        train_iters=10_000,
                        test_every=100,
                        test_iters=10,
                        use_tqdm=True,
                        eval_fns=[gen_acc_rl]
                        )


# <codecell>
task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, rl_prompt=True, batch_size=batch_size, n_thought=None)
batch = next(task)
gen_acc_rl(state, batch)

# <codecell>
xs, ys = next(task)
traj = gen2(state, xs)

print(traj[:3])
print(ys[:3])

# <codecell>
jax.tree.map(np.shape, state.params)

# <codecell>
W = state.params['Dense_0']['kernel']
V0 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O0 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
V1 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
O1 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()

R = V0 @ O0 @ V1 @ O1 @ W

xs = np.array(state.params['Embed_freeze']['embedding'])

logits = xs @ R

out = (xs[[25]] + 0.1 * xs[[3]]) @ R

R.shape

est = np.linalg.pinv(xs @ xs.T) @ xs @ R
R_est = xs.T @ est

plt.imshow(logits, vmin=-500, vmax=500, cmap='BrBG')
plt.colorbar()

plt.xlabel('Class (output)')
plt.ylabel('Token (input)')

# plt.savefig('fig/logits.png')

# <codecell>
plt.plot(logits[3], 'o--')


# <codecell>
xs, ys = next(train_task)
logits, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

atts = [intm['intermediates'][f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze() for i in range(n_layers)]

idx = 0
fig, axs = plt.subplots(1, len(atts), figsize=(4 * len(atts), 4))

if n_layers == 1:
    axs = np.array([axs])

for ax, att in zip(axs.ravel(), atts):
    im = ax.imshow(att[idx])
    labs = list(xs[idx])
    ax.set_xticks(np.arange(len(labs)))
    ax.set_xticklabels(labs)
    ax.set_yticks(np.arange(len(labs)))
    ax.set_yticklabels(labs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    print('')

fig.tight_layout()

preds = logits.argmax(-1)
print('preds', preds[idx])
# <codecell>
readout = state.params['Dense_0']['kernel']
dot = readout.T @ readout
normd = readout / np.linalg.norm(readout, axis=0, keepdims=True)
n_dot = normd.T @ normd

# plt.imshow(dot, vmin=-30, vmax=30, cmap='BrBG')
plt.imshow(n_dot, vmin=-1, vmax=1, cmap='BrBG')
plt.colorbar()

boundaries = 2**np.arange(depth) + 2.5
boundaries[0] = 0

# for b in boundaries:
#     plt.axhline(y=b, color='red', alpha=0.3)
#     plt.axvline(x=b, color='red', alpha=0.3)

# nodes = np.arange(3, n_vocab // 2 + 1)
# plt.plot(nodes + 1, 2 * nodes, color='magenta', alpha=0.4)

# nodes = np.arange(5, n_vocab)
# plt.plot(nodes, 0.5 * nodes + 1, color='magenta', alpha=0.4)


# plt.savefig('fig/readout_sim.png')
