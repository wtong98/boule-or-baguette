"""Exploring generalization on large graphs"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import *
from task.graph import *

depth = 10
n_hidden = 512
batch_size = 128

cot = False
ttr = False
nouveau = False
n_arms = 100
n_hop = 5
test_n_hop = 7

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(n_hop + 1, test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

# xs, ys = next(train_task)
# print(xs[:3])
# print(ys[:3])

# <codecell>
# config = TrConfig(n_vocab=n_vocab, 
#                   pos_emb=not cot,
#                   rand_pos_emb=True,
#                   big_pe=False,
#                   n_out=n_vocab if cot else 1,
#                   n_hidden=n_hidden, 
#                   return_format=None if cot else True)

config = TransformerConfig(n_layers=1,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                        #    pos_emb=not cot,
                           pos_emb=False,
                        #    max_len=100,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=False,
                           freeze_emb=True,
                           use_bias=False,
                        #    return_format=None if cot else True,
                           return_format=None if cot else 'final_logit',
                           mup_scale=True,
                           linear_att=False
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    train_iters=50_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    print_fn=print_gen if cot else None,
                    lr=1e-2,
                    # lr=3e-5,
                    # lr=1,
                    # optim=optax.sgd,
                    # lr=optax.schedules.warmup_cosine_decay_schedule(
                    #     init_value=1e-4,
                    #     peak_value=1e-2,
                    #     warmup_steps=2000,
                    #     decay_steps=50_000,
                    #     end_value=1e-4
                    # )
                    )

# <codecell>
jax.tree.map(np.shape, state.params)

A = state.params['Dense_0']['kernel']
w = state.params['Dense_1']['kernel']

emb = state.params['Embed_freeze']['embedding']
pos = state.params['PE_freeze']['embedding']

# xs, ys = next(test_task)
xs = jnp.array([[10, 19]])
xs_orig = np.copy(xs)
out = state.apply_fn({'params': state.params}, xs)

xs_s = emb[xs]
# ps = pos * np.sqrt(n_hidden)
ps = pos
xs = xs_s + ps

a_ss = xs_s @ t(xs_s @ A)
a_sp = xs_s @ t(ps @ A)
a_ps = ps @ t(xs_s @ A)
a_pp = ps @ t(ps @ A)

att = a_ss + a_sp + a_ps + a_pp
print('a_ss', a_ss[0,-1])
print('a_sp', a_sp[0,-1])
print('a_ps', a_ps[0,-1])
print('a_pp', a_pp[-1])
# att_t = jnp.tril((xs_s + ps) @ t((xs_s + ps) @ A))
# print(att[:3])
# print(att_t[:3])

w1 = (xs[:,0] @ w).flatten()
w2 = (xs[:,1] @ w).flatten()
# ps = ps[None]
# w1 = (ps[:,0] @ w).flatten()
# w2 = (ps[:,1] @ w).flatten()

a1 = att[:,1,0]
a2 = att[:,1,1]

pred = a1 * w1 + a2 * w2

print('a1', a1)
print('w1', w1)
print('a2', a2)
print('w2', w2)


print(pred)
print(out)

# plt.hist(-out[out < 0], bins=10)


# <codecell>
df = collate_dfs('remote/10_big_graph/length', show_progress=True)
df

# %%
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].samp_dist[1],
        row['info'],
    ], index=['name', 'n_arms', 'n_hop', 'info'])

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
mdf = plot_df.copy()
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', row='n_arms', col='n_hop', kind='line', estimator='mean', marker='o', height=2, aspect=1.2, hue_order=['Zero', 'AR', 'AR full'])

plt.savefig(f'fig/tr_star_many_arms.png')


# <codecell>
df = collate_dfs('remote/10_big_graph/length_sweep', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    prop = row['info']['n_hop_prop']
    del row['info']['n_hop_prop']

    return pd.Series([
        row['name'],
        row['train_task'].samp_dist[1],
        row['train_task'].n_arms,
        row['config']['linear_att'],
        row['config']['layer_norm'],
        row['config']['residual_connections'],
        row['config']['n_mlp_layers'],
        row['train_task'].trace_to_start,
        row['train_task'].depth,
        row['train_task'].cot,
        prop,
        row['info'],
    ], index=['name', 'n_hop', 'n_arms', 'linear_att', 'layer_norm', 'resid', 'n_mlp_layers', 'trace_to_start', 'depth', 'cot', 'n_hop_prop', 'info'])

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

# %%
for linear_att, cot in itertools.product([True, False], [True, False]):
    mdf = plot_df.copy()
    mdf = mdf[
        (mdf['n_arms'] == 10)
        & (mdf['linear_att'] == linear_att)
        & (mdf['cot'] == cot)
        & (mdf['trace_to_start'] == False)
        ]

    sns.relplot(mdf, x='test_n_hop', y='acc', col='n_hop', row='depth', hue='name')
    plt.savefig(f'fig/tr_star_big_sweep_lin_att_{linear_att}_cot_{cot}.png')
    plt.show()


# <codecell>
df = collate_dfs('remote/10_big_graph/mlp_size', show_progress=True)
df

# %%
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['info'],
    ], index=['name', 'n_arms', 'n_hop', 'n_hidden', 'info'])

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

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()

name = 'Zero (LN+resid)'
# name = 'Zero (base)'
# name = 'AR full'

mdf = mdf[
    (mdf['test_n_hop'] == 7)
    & (mdf['name'] == name)
    ]

mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).max()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 34 - 1 * xs, color='black', linestyle='dashed')

# xs = 2**np.linspace(-5, 8)
# g.plot(xs, 1 - 2 * xs + 13, color='black', linestyle='dashed')

g.set_ylabel('n_arms')
g.set_xlabel('n_hidden')

g.set_title(name)
plt.savefig(f'fig/{name}_mlp_arms_v_size_debug.png', bbox_inches='tight')


# <codecell>
df = collate_dfs('remote/10_big_graph/tf_size', show_progress=True)
df

# %%
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['info'],
    ], index=['name', 'n_arms', 'n_hop', 'n_hidden', 'info'])

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

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()

# typ = 'simp lin att, sin PE'
# name = 'Zero (big PE)'

mdf = mdf[
    # (mdf['test_n_hop'] == 6)
    # & (mdf['name'] == name)
    (mdf['n_hidden'] == 512)
    ]

gs = sns.relplot(mdf, x='n_arms', y='acc', hue='name', marker='o', legend='full', col='test_n_hop', height=2)
gs.set(xscale='log')

plt.savefig(f'fig/lin_att_many_arms.png')