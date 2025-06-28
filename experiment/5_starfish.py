"""Experimenting with Starfish task"""


# <codecell>
from pathlib import Path
import pickle

from flax import linen as nn
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


def transformer_phi(X, flatten=True):
    X_curr = X.reshape(*X.shape, 1, 1)

    X = jnp.repeat(jnp.expand_dims(X, axis=1), X.shape[1], axis=1)
    X = jnp.permute_dims(X, (0, 3, 1, 2))  # B x H x L x L
    X = jnp.tril(X)
    X = jnp.permute_dims(X, (0, 2, 3, 1))  # B x L x L x H
    X = t(X) @ X

    X = jnp.expand_dims(X, axis=2)
    X = X_curr * X                            # B x L x j x k x m

    X = jnp.permute_dims(X, (0, 1, 4, 2, 3))  # B x L x m x j x k

    if flatten:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    return X


@struct.dataclass
class TrLogRegConfig:
    n_vocab: float
    n_hidden: float = 32
    flatten: bool = False

    def to_model(self):
        return TrLogReg(self)


class TrLogReg(nn.Module):
    config: TrLogRegConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs)
        x = transformer_phi(x, flatten=self.config.flatten)
        
        if not self.config.flatten:
            x = nn.Dense(1, use_bias=False)(x).squeeze()

        x = nn.Dense(self.config.n_vocab, use_bias=False)(x)
        return x


depth = 25
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 32
batch_size = 64

n_layers = 1

cot = True
ttr = False

train_task = StarfishTask(depth=depth, samp_dist=(1,10), batch_size=batch_size, cot=cot, trace_to_start=ttr)
test_task = StarfishTask(depth=depth, samp_dist=15, batch_size=batch_size, cot=cot, trace_to_start=ttr)


# config = MlpConfig(n_vocab=n_vocab,
#                    n_layers=1,
#                    n_emb=n_hidden,
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
#                            return_format='final_logit',
#                            mup_scale=True
#                            )

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
                           freeze_emb=True,
                           use_bias=False,
                           return_format=None,
                           mup_scale=True,
                           linear_att=True
                           )

config = TrLogRegConfig(n_vocab=n_vocab, n_hidden=n_hidden, flatten=True)

# xs, ys = next(train_task)

# print(xs[:3])
# print(ys[:3])

# fix_emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)

# <codecell>
# state = create_train_state(jax.random.key(new_seed()),
#                            config.to_model(),
#                            next(train_task)[0],
#                            lr=1e-3)

# state.params['Embed_freeze']['embedding'] = fix_emb

state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    # loss='bce',
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen,
                    lr=1e-3,
                    # optim=optax.sgd
                    )


# <codecell>
xs, ys = next(test_task)

# TODO: investigate with zero temperature
preds = gen2(state, xs)

# print('INPT', xs[:3])
# print('PRED', preds[:3])
# print('LABL', ys[:3])

print('INPT', xs[-3:])
print('PRED', preds[-3:])
print('LABL', ys[-3:])

# <codecell>
logits = state.apply_fn({'params': state.params}, xs)
logits[3].argmax(-1)

# <codecell>
vals = logits[3][18]
p = np.exp(vals) / np.sum(np.exp(vals))
plt.plot(p)


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
### RECONSTRUCTION AS LINEAR FUNCTION
xs, ys = next(train_task)
logits = state.apply_fn({'params': state.params}, xs)

W = state.params['Dense_0']['kernel']
emb = state.params['Embed_freeze']['embedding']
K = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['key']['kernel'].squeeze()
Q = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['query']['kernel'].squeeze()
V = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze()
O = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze()

idx = 0

W = V @ O @ W
A = Q @ K.T

X = emb[xs]
X = transformer_phi(X, flatten=False)
# A = A.T.reshape(-1, 1)

# M = jnp.kron(A, W)

# model = TrLogRegConfig(n_vocab).to_model()
# params = {'Dense_0': {'kernel': M}, 'Embed_freeze': {'embedding': emb}}
# pred_logits = model.apply({'params': params}, xs)

# pred_logits = X @ np.kron(A, W)

# pred_logits = (W.T @ X @ A.reshape(-1, 1)).squeeze()
# pred_logits = (X @ A.reshape(-1, 1)).squeeze() @ W

model = TrLogRegConfig(n_vocab, flatten=False).to_model()
params = {'Dense_0': {'kernel': A.reshape(-1, 1)}, 
          'Dense_1': {'kernel': W},
          'Embed_freeze': {'embedding': emb}}
pred_logits = model.apply({'params': params}, xs)

np.mean((logits - pred_logits)**2)

# <codecell>
state = create_train_state(jax.random.key(new_seed()), model, xs)
state.params['Dense_0']['kernel'] = M

batch = next(train_task)
gen_acc_cot(state, batch)


# <codecell>
x = np.arange(5)[:,None]
np.tril(np.repeat(x, 5, axis=-1))

# <codecell>
W = state.params['Dense_0']['kernel']
# V0 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
# O0 = state.params['TransformerBlock_0']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
# V1 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['value']['kernel'].squeeze()
# O1 = state.params['TransformerBlock_1']['MultiHeadDotProductAttention_0']['out']['kernel'].squeeze()
# R = V0 @ O0 @ V1 @ O1 @ W

V0 = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze()
O0 = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze()
V1 = state.params['TransformerBlock_1']['SimpleSelfAttention_0']['value']['kernel'].squeeze()
O1 = state.params['TransformerBlock_1']['SimpleSelfAttention_0']['out']['kernel'].squeeze()
R = V0 @ O0 @ V1 @ O1 @ W

xs = np.array(state.params['Embed_freeze']['embedding'])

logits = xs @ R

# out = (xs[[25]] + 0.1 * xs[[3]]) @ R

R.shape

est = np.linalg.pinv(xs @ xs.T) @ xs @ R
R_est = xs.T @ est

plt.imshow(logits, cmap='BrBG')
# plt.imshow(logits, cmap='BrBG', vmin=-1000, vmax=1000)
plt.colorbar()

plt.xlabel('Class (output)')
plt.ylabel('Token (input)')

# plt.savefig('fig/logits.png')

# <codecell>
plt.plot(logits[16], 'o--')

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

# atts = [intm['intermediates'][f'TransformerBlock_{i}']['MultiHeadDotProductAttention_0']['attention_weights'][0].squeeze() for i in range(n_layers)]
atts = [intm['intermediates'][f'TransformerBlock_{i}']['SimpleSelfAttention_0']['attention_weights'][0].squeeze() for i in range(n_layers)]

idx = -1
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
xs = jnp.array([[7, 57, 0, -3, -3, -3, -3, -3, -3]]) + 3
traj = gen2(state, xs)

traj - 3


# <codecell>
### SIZE VS NODE PLOTTING
df = collate_dfs('remote/5_starfish/size', show_progress=True)
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
        row['train_task'].depth,
        row['info']['gen_acc'],
        row['info']['loss'],
        best_acc,
        best_loss
    ], index=['name', 'n_hidden', 'n_mlp_layers', 'layer_norm', 'mup_scale', 'depth', 'gen_acc', 'loss', 'best_acc', 'best_loss'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \


# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_mlp_layers'] == 2)
    & (mdf['layer_norm'] == False)
    & (mdf['mup_scale'] == False)
    ]

g = sns.lineplot(mdf, x='n_hidden', y='best_acc', hue='depth', marker='o')

g.set_xscale('log', base=2)


# g.set_yscale('log')
plt.savefig('fig/acc_star_scale_mlp.png')

# <codecell>
plot_df[plot_df['depth'] == 10]

# <codecell>
plt.plot([m['gen_acc'] for m in df.iloc[59]['hist']['test']])


# <codecell>
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/5_starfish/length', show_progress=True)
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
hops = [1, 3, 5, 10, 16]
set_theme()

for hop in hops:
    mdf = plot_df[plot_df['n_hop'] == hop]
    g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='name', marker='o', estimator='max')
    g.axvline(x=hop, color='gray', linestyle='dashed')

    g.legend().set_title(None)
    sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))

    g.set_xlabel('Test distance')
    g.set_ylabel('Accuracy')

    plt.savefig(f'fig/star_length_{hop}_gen_lin.png', bbox_inches='tight')
    plt.show()

# <codecell>
mdf = plot_df.copy()
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', errorbar=('ci', False), estimator='max', marker='o', height=2, aspect=1.2)

plt.savefig('fig/star_length_gen_lin.png')


# <codecell>
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/5_starfish/length_sweep', show_progress=True)
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
hops = [1, 3, 5, 10, 16]
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

    plt.savefig(f'fig/star_length_{hop}_tts_{tts}_gen_lin.png', bbox_inches='tight')
    plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    ((mdf['n_layers'] == 2) | (mdf['n_layers'] == 1))
    ]

sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', errorbar=('ci', False), estimator='max', marker='o', height=2, aspect=1.2)

# plt.savefig('fig/star_length_gen.png')


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
df = collate_dfs('remote/5_starfish/rl_rep/set1_nonlin_good', show_progress=True)
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
trace_to_start = [False, True]

for ttr in trace_to_start:
    mdf = plot_df.copy()
    mdf = mdf[(mdf['trace_to_start'] == ttr)]
    gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='n_hidden', row='train_iters', marker='o', height=2, aspect=2, alpha=0.7, estimator='max')

    # plt.savefig(f'fig/rl_rep_star_ttr_{ttr}.png')
    plt.show()

# <codecell>
set_theme()

mdf = plot_df.copy()
mdf = mdf[mdf['n_hidden'] == 512]
mdf['test_len'] = mdf['test_len'].astype(float)

gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='trace_to_start', row='train_iters', marker='o', height=2, aspect=2, alpha=0.7, estimator='max')

gs.set_titles('CoT full = {col_name} | train = {row_name}')
gs.set_xlabels('Distance')
gs.set_ylabels('Accuracy')

for g in gs.axes.ravel():
    g.axvline(x=5, color='gray', linestyle='dashed')

plt.savefig('fig/rl_star_wide_comparison_lin.png', bbox_inches='tight')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_hidden'] == 512]
mdf = mdf[mdf['trace_to_start'] == False]

mdf['test_len'] = mdf['test_len'].astype(float)

gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='train_iters', marker='o', height=3, aspect=1.8, alpha=0.7, estimator='max')

gs.set_titles('Train iters = {col_name}')
gs.set_xlabels('Distance')
gs.set_ylabels('Accuracy')

for g in gs.axes.ravel():
    g.axvline(x=5, color='gray', linestyle='dashed')

plt.savefig('fig/rl_star_wide_comp_trunc.png', bbox_inches='tight')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['trace_to_start'] == True]
mdf['test_len'] = mdf['test_len'].astype(float)

gs = sns.relplot(mdf, kind='line', x='test_len', y='gen_acc', hue='mode', col='n_hidden', row='train_iters', marker='o', height=2, aspect=2, alpha=0.7, estimator='max')

gs.set_titles('width = {col_name} | train = {row_name}')
gs.set_xlabels('Distance')
gs.set_ylabels('Accuracy')

for g in gs.axes.ravel():
    g.axvline(x=5, color='gray', linestyle='dashed')

plt.savefig('fig/rl_star_size_sweep_lin.png', bbox_inches='tight')