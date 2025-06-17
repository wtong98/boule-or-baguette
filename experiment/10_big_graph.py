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
n_arms = 150
n_hop = 5
test_n_hop = 8

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(n_hop + 1, test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

# xs, ys = next(train_task)
# print(xs[:3])
# print(ys[:3])
# <codecell>
next(train_task)

# <codecell>
# config = TrConfig(n_vocab=n_vocab, 
#                   pos_emb=not cot,
#                   rand_pos_emb=True,
#                   n_out=n_vocab if cot else 1,
#                   n_hidden=n_hidden, 
#                   return_final_logits_only=False if cot else True)

config = TransformerConfig(n_layers=2,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                        #    pos_emb=not cot,
                           pos_emb=True,
                           max_len=100,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False if cot else True,
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
                    train_iters=100_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    print_fn=print_gen if cot else None,
                    # lr=3e-5,
                    # lr=1e-3
                    # lr=1,
                    # optim=optax.sgd,
                    )

# <codecell>
jax.tree.map(np.shape, state.params)

# <codecell>

xs, ys = next(test_task)

out, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

atts = intm['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['attention_weights'][0].squeeze()

print(np.mean((out > 0).astype(int) == ys))

atts[-4]


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
n_arms = 2
mdf = plot_df.copy()
mdf = mdf[mdf['n_arms'] == n_arms]
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', estimator='mean', marker='o', height=2, aspect=1.2, hue_order=['Zero', 'AR', 'AR full'])

plt.savefig(f'fig/tr_star_arms_{n_arms}.png')
plt.suptitle(f'n_arms = {n_arms}')


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