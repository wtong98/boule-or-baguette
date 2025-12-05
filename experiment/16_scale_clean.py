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
          & (mdf['n_depth'] == 25)
          ]


mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)
# g = sns.heatmap(mdf, square=False)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 45 - 2 * xs, color='cyan', linestyle='dashed')

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
          & (mdf['n_arms'] == 10)
          ]


mdf = mdf[['n_depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.6, vmax=0.9)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 30 - 1 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('DP depth')
plt.show()



# <codecell>
df = collate_dfs('remote/16_scale_clean/ar_breadth', show_progress=True)
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
          & (mdf['n_depth'] == 8)
          ]


mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=0.95)
# g = sns.heatmap(mdf, square=False)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 40 - 2 * xs, color='cyan', linestyle='dashed')

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
          & (mdf['n_arms'] == 8)
          ]


mdf = mdf[['n_depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=0.9)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 33 - 0.5 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('AR depth')
plt.show()


# <codecell>
df = collate_dfs('remote/16_scale_clean/gen/set', show_progress=True)
df

# <codecell>
rand_idxs = np.random.choice(len(df), size=100, replace=False)
for ex in df['hist'].iloc[rand_idxs]:
    vals = [p['loss'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# df['hist'].iloc[0]['train']
    

# %%
def extract_plot_vals(row):
    n_hop_prop = row['info']['n_hop_prop']
    del row['info']['n_hop_prop']
    del row['info']['n_hop']

    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].depth,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['config']['n_layers'],
        row['hist']['test'][5]['acc'],
        n_hop_prop,
        row['info'],
    ], index=['name', 'n_arms', 'depth', 'n_hop', 'n_hidden', 'n_layers', 'acc_hist', 'n_hop_prop', 'info'])

plot_df = df.copy().apply(extract_plot_vals, axis=1) \
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

depth = 30
n_hop_prop = 0.25

train_split = np.round(depth * n_hop_prop)

mdf = mdf[
    (mdf['n_hop_prop'] == n_hop_prop)
    & (mdf['depth'] == depth)
    # & (mdf['name'] == 'AR')
]

g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='n_arms', style='name', legend='full')
g.axvline(x=train_split, color='red', linestyle='dashed', alpha=0.7)
g.set_title('DP generalization')

# plt.savefig(f'fig/dp_gen_depth_{depth}_hop_{n_hop_prop}.png', bbox_inches='tight')

# <codecell>
# adf = mdf[mdf['test_n_hop'] < 15]
# adf = mdf.copy()

g = sns.lineplot(adf, x='n_arms', y='acc', hue='test_n_hop', alpha=0.5, marker='o')
g.set_xscale('log')
g.set_yscale('log')

xs = np.linspace(np.log(8), np.log(27))
g.plot(np.exp(xs), np.exp(-0.5 * xs + 1), color='red', linestyle='dashed')
g.text(12, 0.8, r'$\propto B^{-1/2}$', color='red')
# g.plot(np.exp(xs), np.exp(-1 * xs + 2), color='blue', linestyle='dashed')

# plt.savefig(f'fig/dp_gen_arms_pred_depth_{depth}_hop_{n_hop_prop}.png', bbox_inches='tight')