"""Can our models learn the graph task with CoT?"""


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
df = collate_dfs('remote/2_cot/generalize', show_progress=True)
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

    # plt.savefig(f'fig/acc_reduce_bal_{order}_{branch}.png')
    plt.show()

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

# xs, ys = next(test_task)

# config = MlpConfig(mup_scale=False,
#                    n_out=n_vocab, 
#                    n_vocab=n_vocab, 
#                    n_layers=1, 
#                    n_hidden=n_hidden, 
#                    use_bias=False,
#                    freeze_emb=True,
#                    act_fn='relu')


# config = MixerConfig(n_layers=2, 
#                      n_vocab=n_vocab, 
#                      n_out=n_vocab,
#                      layer_norm=False,
#                      n_hidden=n_hidden, 
#                      n_channels=32)

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

state, hist = train(config,
                    data_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=25_000,
                    lr=1e-3,
                    # gamma=gamma,
                    # optim=optax.sgd,
                    # lr=lr,
                    use_tqdm=True
                    )


# <codecell>
### RL experimentation

# TODO: define mechanism to sample iteratively on Transformer



# <codecell>
xs = np.array([[10, 126, 0, 126, 63, 31, 15, 7, -3, -3, -3, -3, -3, -3]]) + 3

logits = state.apply_fn({'params': state.params}, xs)
np.argmax(logits, axis=-1) - 3

# <codecell>
xs, ys = next(train_task)
logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(-1)
preds

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
