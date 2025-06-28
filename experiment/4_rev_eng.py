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


depth = 5
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 128

n_layers = 2

seed = new_seed()

repeat_first = True
trace_to_start = False

train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1,3), on_branch=True, cot=True, batch_size=batch_size, use_sep=True, repeat_first=repeat_first, trace_to_start=trace_to_start),
    BinaryTreeTiTask(depth=depth, samp_dist=(1,3), on_branch=False, cot=True, batch_size=batch_size, use_sep=True, repeat_first=repeat_first, trace_to_start=trace_to_start))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, cot=True, batch_size=batch_size, use_sep=True, repeat_first=repeat_first, trace_to_start=trace_to_start)


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
                           return_format=None,
                           mup_scale=False
                           )


# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=25_000,
                    # lr=1e-2,
                    # optim=optax.sgd,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen
                    )

# <codecell>
# test_task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, cot=True, batch_size=batch_size, n_thought=None)
# test_task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, cot=True, batch_size=batch_size, use_sep=True, repeat_first=True, trace_to_start=False)

# xs, ys = next(test_task)
# ys = ys[:,2:]
# ans_idx = jnp.sum(ys != 0, axis=-1) - 1
# ans = ys[jnp.arange(len(ys)), ans_idx]

# traj = gen2(state, xs)
# preds = extract_pred(traj)

# print('PR', preds)
# print('AN', ans)

# traj = gen2(state, xs)

# print(traj[:3])
# print(ys[:3])

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
task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, cot=True, batch_size=batch_size, n_thought=None)
batch = next(task)
gen_acc_cot(state, batch)

# <codecell>
task = BinaryTreeTiTask(depth=depth, samp_dist=1, on_branch=False, cot=True, batch_size=batch_size, use_sep=True, repeat_first=repeat_first, trace_to_start=trace_to_start)
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

# out = (xs[[25]] + 0.1 * xs[[3]]) @ R

R.shape

est = np.linalg.pinv(xs @ xs.T) @ xs @ R
R_est = xs.T @ est

plt.imshow(logits, cmap='BrBG', vmin=-500, vmax=500)
plt.colorbar()

plt.xlabel('Class (output)')
plt.ylabel('Token (input)')

boundaries = 2**np.arange(depth) + 2.5
boundaries[0] = 3.5

for b in boundaries:
    plt.axhline(y=b, color='red', alpha=0.3)
    plt.axvline(x=b, color='red', alpha=0.3)

nodes = np.arange(3, n_vocab // 2 + 1)
plt.plot(nodes + 1, 2 * nodes, color='magenta', alpha=0.4)

nodes = np.arange(5, n_vocab)
plt.plot(nodes, 0.5 * nodes + 1, color='magenta', alpha=0.4)

plt.savefig('fig/logits_with_guides.png')

# <codecell>
plt.plot(logits[4], 'o--')

# <codecell>
plt.plot(logits[:,1], 'o--')
plt.plot(logits[:,2], 'o--')
plt.plot(logits[:,30], 'o--')
plt.plot(logits[:,20], 'o--')


# <codecell>
xs, ys = next(train_task)
xs = np.array(xs)
# xs[0] = [6, 59,  3, 59, 31, 17, 10,  6,  4,  2]
# xs[0] = [5, 7, 3, 7, 5, 2, 0]

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

# <codecell>
xs = jnp.array([[7, 57, 0, -3, -3, -3, -3, -3, -3]]) + 3
traj = gen2(state, xs)

traj - 3


# <codecell>
### SIZE VS NODE PLOTTING
df = collate_dfs('remote/4_rev_eng/size_and_node', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    best_acc = np.max([m['gen_acc'] for m in row['hist']['test']])
    best_loss = np.min([m['loss'] for m in row['hist']['test']])

    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['config']['n_mlp_layers'],
        row['config']['layer_norm'],
        row['config']['mup_scale'],
        row['train_task'].tasks[0].depth,
        row['info']['gen_acc'],
        row['info']['loss'],
        best_acc,
        best_loss
    ], index=['name', 'n_hidden', 'n_mlp_layers', 'layer_norm', 'mup_scale', 'depth', 'gen_acc', 'loss', 'best_acc', 'best_loss'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

plot_df


# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_mlp_layers'] == 0)
    & (mdf['layer_norm'] == False)
    & (mdf['mup_scale'] == True)
    ]

g = sns.lineplot(mdf, x='n_hidden', y='best_acc', hue='depth', marker='o')

g.set_xscale('log', base=2)

# xs = np.unique(plot_df['n_hidden'])
# preds = 0.97 - 1 / (0.01 * xs)
# preds = 0.04 * np.log(xs)**(3.1/2)
# plt.plot(xs, preds)


# g.set_yscale('log')
# plt.savefig('fig/acc_scale_mlp_mup.png')

# <codecell>
plot_df[plot_df['depth'] == 10]

# <codecell>
plt.plot([m['gen_acc'] for m in df.iloc[59]['hist']['test']])