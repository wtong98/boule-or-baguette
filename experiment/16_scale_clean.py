"""Plotting script for the scale clean experiments."""

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
from model.mlp import MlpConfig
from model.transformer_std import *
from task.graph import *


# <codecell>
df = collate_dfs('remote/16_scale_clean/dp_breadth', show_progress=True)
df

# %%
# TODO: clean up and plot rest <-- STOPPED HERE
def extract_plot_vals(row):
    info = row['info'].copy()
    n_hop_prop = info.pop('n_hop_prop', None)

    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].samp_dist[1],
        row['train_task'].depth,
        row['config']['n_hidden'],
        row['config']['residual_connections'],
        row['config']['n_layers'],
        row['train_args']['lr'],
        row['hist']['test'][-1]['acc'],
        row['train_args']['gamma'] if 'gamma' in row['train_args'] else -1,
        n_hop_prop,
        info
    ], index=['name', 'n_arms', 'n_hop', 'n_depth', 'n_hidden', 'resid', 'n_layers',  'lr', 'acc_hist', 'gamma', 'n_hop_prop', 'info'])

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
# bdf.loc[~pd.isna(bdf['gen_acc']),'acc'] = bdf[~pd.isna(bdf['gen_acc'])]['gen_acc']
# bdf = bdf.drop('gen_acc', axis=1)

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['test_n_hop'] == 0.5)
          & (mdf['n_depth'] == 10)
        #   & (mdf['gamma'] == 1)
        #   & (mdf['resid'] == False)
          ]


# TODO: track generative historical accuracy, too
mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)
# g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)
# g = sns.heatmap(mdf, square=False)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 30 - 1 * xs, color='cyan', linestyle='dashed')
g.plot(xs, 30 - 2 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 50 - 2 * xs, color='cyan', linestyle='dashed')

# g.plot(xs, 39 - xs, color='gray', linestyle='dashed')

# xs = 2**np.linspace(-5, 8)
# g.plot(xs, 1 - 2 * xs + 13, color='black', linestyle='dashed')

g.set_ylabel('Breadth (B)')
g.set_xlabel('Hidden (H)')

plt.title('AR early')
plt.text(9, 5, r'$\propto H^2$', color='cyan')
plt.text(22, 5, r'$\propto H$', color='cyan')
# plt.savefig(f'fig/ar_mlp_BH_early.png', bbox_inches='tight')
# plt.savefig(f'fig/zero_mlp_arms_v_size_short.png', bbox_inches='tight')
# plt.savefig(f'fig/ar_mlp_arms_v_size_debug.png', bbox_inches='tight')
# plt.savefig('fig/debug.png', bbox_inches='tight')
plt.show()
