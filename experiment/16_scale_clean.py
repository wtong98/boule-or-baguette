"""Plotting script for the scale clean experiments."""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import collate_dfs

# NOTE: would be nice to have larger n_arms in sweep
# NOTE: may be able to get away with fewer training iterations?

# <codecell>
df = collate_dfs('remote/16_scale_clean/dp_breadth', show_progress=True)
df

# %%
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
          & (mdf['n_depth'] == 20)
          ]


mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 28 - 2 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Breadth (B)')
g.set_xlabel('Hidden (H)')

plt.title('DP breadth')
plt.show()


# <codecell>
df = collate_dfs('remote/16_scale_clean/dp_depth', show_progress=True)
df

# %%
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
          & (mdf['n_arms'] == 20)
          ]


mdf = mdf[['n_depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 20 - 1 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('DP depth')
plt.show()



# <codecell>
df = collate_dfs('remote/16_scale_clean/ar_breadth/set1_bare', show_progress=True)
df

# %%
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
bdf.loc[~pd.isna(bdf['gen_acc']),'acc'] = bdf[~pd.isna(bdf['gen_acc'])]['gen_acc']
bdf = bdf.drop('gen_acc', axis=1)

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['test_n_hop'] == 0.5)
          & (mdf['n_depth'] == 10)
          ]


mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)
# g = sns.heatmap(mdf, square=False)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 23 - 2 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Breadth (B)')
g.set_xlabel('Hidden (H)')

plt.title('AR breadth')
plt.show()


# <codecell>
df = collate_dfs('remote/16_scale_clean/ar_depth', show_progress=True)
df

# %%
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
bdf.loc[~pd.isna(bdf['gen_acc']),'acc'] = bdf[~pd.isna(bdf['gen_acc'])]['gen_acc']
bdf = bdf.drop('gen_acc', axis=1)

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['test_n_hop'] == 0.5)
          & (mdf['n_arms'] == 10)
          ]


mdf = mdf[['n_depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 11 - 0.5 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('AR depth')
plt.show()