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

n_layers = 2

cot = True
ttr = True
nouveau = False
n_arms = 2
n_hop = 3
test_n_hop = 5

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(n_hop + 1, test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

xs, ys = next(train_task)
print(xs[:3])
print(ys[:3])

# <codecell>
# config = TrConfig(n_vocab=n_vocab, 
#                   pos_emb=not cot,
#                   n_out=n_vocab if cot else 1,
#                   n_hidden=n_hidden, 
#                   return_final_logits_only=False if cot else True)

config = TransformerConfig(n_layers=n_layers,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                           pos_emb=not cot,
                           max_len=100,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=True,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False if cot else True,
                           mup_scale=True,
                           linear_att=True
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    print_fn=print_gen if cot else None,
                    # lr=3e-5,
                    # lr=1e-3
                    # lr=1,
                    # optim=optax.sgd,
                    )

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
mdf = mdf[mdf['n_arms'] == 13]
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', estimator='mean', marker='o', height=2, aspect=1.2)
