"""Experimenting with Starfish task"""


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
from model.mlp import MlpConfig
from model.transformer import TransformerConfig 
from task.graph import *


depth = 25
n_vocab = 2 * depth + 1 + CircleTask.offset
n_hidden = 128
batch_size = 128

n_layers = 2

cot = True
ttr = True

train_task = CircleTask(depth=depth, samp_dist=(1,5), batch_size=batch_size, cot=cot, trace_to_start=ttr)
test_task = CircleTask(depth=depth, samp_dist=15, batch_size=batch_size, cot=cot, trace_to_start=ttr)

# config = MlpConfig(n_vocab=n_vocab,
#                    n_layers=1,
#                    n_hidden=n_hidden,
#                    use_bias=False)

# config = TransformerConfig(n_layers=2,
#                            n_vocab=n_vocab,
#                            n_out=1,
#                            n_hidden=n_hidden,
#                            pos_emb=True,
#                            n_mlp_layers=0,
#                            n_heads=1,
#                            layer_norm=False,
#                            as_rf_model=False,
#                            residual_connections=False,
#                            use_simple_att=False,
#                            freeze_emb=True,
#                            use_bias=False,
#                            return_final_logits_only=True,
#                            mup_scale=True
#                            )

config = TransformerConfig(n_layers=n_layers,
                           n_vocab=n_vocab,
                           n_out=n_vocab,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           use_simple_att=False,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False,
                           mup_scale=True
                           )

# xs, ys = next(train_task)

# print(xs[:3])
# print(ys[:3])

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    # loss='bce',
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=20_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen,
                    # lr=1e-3
                    )


# <codecell>
xs, ys = next(train_task)
# logits = state.apply_fn({'params': state.params}, xs)
# preds = logits.argmax(-1)
preds = gen2(state, xs)

print(xs[:3])
print(preds[:3])
print(ys[:3])

gen_acc_cot(state, (xs, ys))
# <codecell>
test_task.cot = False
test_task.rl_prompt = True
xs, ys = next(test_task)

preds = gen2(state, xs)

print(xs[:3])
print(preds[:3])
print(ys[:3])

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
task = BinaryTreeTiTask(depth=depth, samp_dist=1, on_branch=False, cot=True, batch_size=batch_size, use_sep=True, repeat_first=repeat_first)
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

plt.imshow(logits, cmap='BrBG')
plt.colorbar()

plt.xlabel('Class (output)')
plt.ylabel('Token (input)')

# plt.savefig('fig/logits.png')

# <codecell>
plt.plot(logits[4], 'o--')

# <codecell>
plt.plot(logits[:,1], 'o--')
plt.plot(logits[:,2], 'o--')
plt.plot(logits[:,30], 'o--')
plt.plot(logits[:,20], 'o--')


# <codecell>
xs, ys = next(test_task)
xs = np.array(xs)
# xs[0] = [6, 59,  3, 59, 31, 17, 10,  6,  4,  2]
# xs[0] = [5, 7, 3, 7, 5, 2, 0]

logits, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

atts = [intm['intermediates'][f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze() for i in range(n_layers)]

idx = 0
fig, axs = plt.subplots(1, len(atts), figsize=(5 * len(atts), 5))

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
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/6_circle/length', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].samp_dist[1],
        row['info'],
    ], index=['name', 'n_hop', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['info'].tolist()) \
        .stack() \
        .reset_index(level=1, name='info')

plot_df = plot_df.drop('info', axis=1) \
                 .join(adf) \
                 .rename(columns={'level_1': 'test_n_hop'}) \
                 .reset_index(names='orig_index')

bdf = pd.DataFrame(plot_df['info'].tolist())
bdf.loc[~pd.isna(bdf['gen_acc']),'acc'] = bdf[~pd.isna(bdf['gen_acc'])]['gen_acc']
bdf = bdf.drop('gen_acc', axis=1)

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
# hops = [1, 3, 5, 10, 16]
hops = [5]
set_theme()

for hop in hops:
    mdf = plot_df[plot_df['n_hop'] == hop]
    g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='name', marker='o', estimator='max')
    g.axvline(x=hop, color='gray', linestyle='dashed')

    g.legend().set_title(None)
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    g.set_xlabel('Test distance')
    g.set_ylabel('Accuracy')

    plt.savefig(f'fig/circle_length_{hop}_gen.png', bbox_inches='tight')
    plt.show()

# <codecell>
mdf = plot_df.copy()
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', estimator='max', marker='o', height=2, aspect=1.2)

plt.savefig('fig/circle_length_gen.png')


# <codecell>
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/6_circle/length_sweep', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].samp_dist[1],
        row['config']['n_layers'],
        row['train_task'].trace_to_start,
        row['info'],
    ], index=['name', 'n_hop', 'n_layers', 'trace_to_start', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['info'].tolist()) \
        .stack() \
        .reset_index(level=1, name='info')

plot_df = plot_df.drop('info', axis=1) \
                 .join(adf) \
                 .rename(columns={'level_1': 'test_n_hop'}) \
                 .reset_index(names='orig_index')

bdf = pd.DataFrame(plot_df['info'].tolist())
bdf.loc[~pd.isna(bdf['gen_acc']),'acc'] = bdf[~pd.isna(bdf['gen_acc'])]['gen_acc']
bdf = bdf.drop('gen_acc', axis=1)

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
# hops = [1, 3, 5, 10, 16]
hops = [5]
use_trace_to_start = [True]

for hop, tts in itertools.product(hops, use_trace_to_start):
    mdf = plot_df.copy()
    mdf = mdf[
        (mdf['n_hop'] == hop)
        & (mdf['trace_to_start'] == tts)
        & (mdf['name'].str.contains('lnorm=False'))
        & ((mdf['n_layers'] == 2) | (mdf['n_layers'] == 1))
    ]
    g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='name', marker='o', estimator='max')
    g.axvline(x=hop, color='gray', linestyle='dashed')

    g.legend().set_title(None)
    names = ['Att', 'Att + MLP', 'Att + Resid', 'Att + MLP + Resid']
    for t, n in zip(g.legend().texts, names):
        t.set_text(n)

    sns.move_legend(g, 'upper left', bbox_to_anchor=(1,1))

    g.set_xlabel('Distance')
    g.set_ylabel('Accuracy')

    plt.savefig(f'fig/circle_length_{hop}_tts_{tts}_gen.png', bbox_inches='tight')
    plt.show()

# <codecell>
mdf = plot_df.copy()
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', errorbar=('ci', False), estimator='max', marker='o', height=2, aspect=1.2)

plt.savefig('fig/circle_length_gen.png')


# <codecell>
### RL
df = collate_dfs('remote/5_starfish/rl', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['train_args']['train_iters'],
        row['train_task'].trace_to_start,
        row['train_task'].samp_dist[1],
        row['info']['etc']['train_len_rl'],
        row['info']
    ], index=['name', 'n_hidden', 'train_iters', 'trace_to_start', 'dist_pr', 'dist_rl', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['info'].tolist()) \
        .stack() \
        .reset_index(level=1, name='info')

adf = adf[adf['level_1'] != 'etc']

plot_df = plot_df.drop('info', axis='columns').join(adf)

ldf = plot_df['level_1'].str.split('_', expand=True) \
                        .rename(columns={
                            0: 'mode',
                            1: 'test_len',
                        })

plot_df = pd.concat((plot_df, ldf), axis='columns').drop('level_1', axis='columns')
adf = pd.DataFrame(plot_df['info'].to_list())
plot_df = pd.concat((plot_df.drop('info', axis=1).reset_index(), adf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()

mdf = mdf[(mdf['n_hidden'] == 128)
        & (mdf['train_iters'] == 100_000)
        & (mdf['trace_to_start'] == True)
]

gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='dist_pr', row='dist_rl', marker='o', height=1.5, aspect=1.5, alpha=0.7, estimator='max')

# plt.savefig(f'fig/acc_rl_stable_small_max_{branch}.png')

# <codecell>
df = collate_dfs('remote/5_starfish/rl_rep', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['n_hidden'],
        row['train_args']['train_iters'],
        row['train_task'].trace_to_start,
        row['train_task'].samp_dist[1],
        row['info']['etc']['train_len_rl'],
        row['info']
    ], index=['name', 'n_hidden', 'train_iters', 'trace_to_start', 'dist_pr', 'dist_rl', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['info'].tolist()) \
        .stack() \
        .reset_index(level=1, name='info')

adf = adf[adf['level_1'] != 'etc']

plot_df = plot_df.drop('info', axis='columns').join(adf)

ldf = plot_df['level_1'].str.split('_', expand=True) \
                        .rename(columns={
                            0: 'mode',
                            1: 'test_len',
                        })

plot_df = pd.concat((plot_df, ldf), axis='columns').drop('level_1', axis='columns')
adf = pd.DataFrame(plot_df['info'].to_list())
plot_df = pd.concat((plot_df.drop('info', axis=1).reset_index(), adf), axis=1)
plot_df

# %%
mdf = plot_df.copy()

mdf = mdf[(mdf['trace_to_start'] == False)]

gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='n_hidden', row='train_iters', marker='o', height=2, aspect=2, alpha=0.7, estimator='max')

# plt.savefig(f'fig/acc_rl_stable_small_max_{branch}.png')