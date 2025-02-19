"""Can our models learn the graph task?"""


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

# <codecell>
df = collate_dfs('remote/1_graph/zero_shot', show_progress=True)
df

# <codecell>
idx = ['name', 'use_bias', 'freeze_emb', 'd_on', 'd_off', 'acc']

def extract_plot_vals(row):
    t1, _, t2 = row['train_task'].tasks
    d1 = t1.samp_dist[1] if isinstance(t1.samp_dist, Iterable) else t1.samp_dist
    d2 = t2.samp_dist[1] if isinstance(t2.samp_dist, Iterable) else t2.samp_dist

    return pd.Series([
        row['name'],
        row['config']['use_bias'],
        row['config']['freeze_emb'],
        d1,
        d2,
        row['info']
    ], index=idx)

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['acc'].tolist()) \
        .stack() \
        .reset_index(level=1, name='acc')

plot_df = plot_df.drop('acc', axis='columns').join(adf)

ldf = plot_df['level_1'].str.split('_', expand=True) \
                        .drop(0, axis='columns') \
                        .rename(columns={
                            1: 'test_len',
                            2: 'order',
                            3: 'branch'
                        })

plot_df = pd.concat((plot_df, ldf), axis='columns').drop('level_1', axis='columns')
plot_df['acc'] = plot_df['acc'].astype('float')

plot_df
# <codecell>
orders = ['fwd', 'rev']
branches = ['on', 'off']

for order, branch in itertools.product(orders, branches):
    mdf = plot_df.copy()
    mdf = mdf[(mdf['use_bias'] == False) 
            & (mdf['freeze_emb'] == True)
            & (mdf['order'] == order)
            & (mdf['branch'] == branch)]

    gs = sns.relplot(mdf, x='test_len', y='acc', hue='name', col='d_on', row='d_off', kind='line', marker='o', height=1.5, aspect=1.2)
    fig = gs.figure
    fig.suptitle(f'order={order}, branch={branch}', size=14)
    fig.subplots_adjust(top=0.9)

    plt.savefig(f'fig/acc_reduce_bal_{order}_{branch}.png')
    plt.show()

# <codecell>
df.iloc[165]['info']

# <codecell>
depth = 10
n_vocab = 2**depth
n_hidden = 256

gamma0 = 1
gamma = gamma0 * np.sqrt(n_hidden)
base_lr = 10
lr = gamma0**2 * base_lr

seed = new_seed()

train_task = Chain(
    BinaryTreeTiTask(order='fwd', depth=depth, samp_dist=1, on_branch=True),
    BinaryTreeTiTask(order='rev', depth=depth, samp_dist=1, on_branch=True),
    # BinaryTreeTiTask(order='split', depth=depth, samp_dist=1, on_branch=True),
    BinaryTreeTiTask(order='split', depth=depth, samp_dist=1, on_branch=False, fill_gaps=False), weights=[3, 1, 2])

xs, ys = next(train_task)
np.mean(ys)
# <codecell>

test_task = Chain(
    BinaryTreeTiTask(order='split', depth=depth, samp_dist=8, on_branch=True),
)


config = MlpConfig(mup_scale=False,
                   n_out=1, 
                   n_vocab=n_vocab, 
                   n_layers=1, 
                   n_hidden=n_hidden, 
                   use_bias=False,
                   freeze_emb=True,
                   act_fn='relu')


# config = MixerConfig(n_layers=2, n_vocab=n_vocab, layer_norm=True)

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
                    loss='bce',
                    test_every=1000,
                    train_iters=20_000, 
                    # gamma=gamma,
                    # optim=optax.sgd,
                    # lr=lr,
                    )

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
