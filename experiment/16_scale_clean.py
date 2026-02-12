"""Plotting script for the scale clean experiments."""

# <codecell>
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import collate_dfs, set_theme

set_theme()

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

g = sns.heatmap(mdf, square=True, vmin=0.6, vmax=0.9)
g.tick_params(axis='both', labelsize=10)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 40 - 2 * xs, color='cyan', linestyle='dashed')
g.text(6, 10, r'$\propto H^2$', color='cyan', fontsize=12)

g.set_ylabel('Breadth (B)')
g.set_xlabel('Hidden (H)')

plt.title('DP breadth')
plt.tight_layout()
plt.savefig('fig/final/fig_ti/dp_breadth.svg')
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

g = sns.heatmap(mdf, square=True, vmin=0.6, vmax=0.9)
g.tick_params(axis='both', labelsize=10)


xs = 2**np.linspace(-5, 8)
g.plot(xs, 27.5 - 1 * xs, color='cyan', linestyle='dashed')
g.text(3, 16, r'$\propto H$', color='cyan', fontsize=12)

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('DP depth')
plt.tight_layout()
plt.savefig('fig/final/fig_ti/dp_depth.svg')
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

g = sns.heatmap(mdf, square=True, vmin=0.6, vmax=0.9)
g.tick_params(axis='both', labelsize=10)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 40 - 2 * xs, color='cyan', linestyle='dashed')
g.text(8, 10, r'$\propto H^2$', color='cyan', fontsize=12)

g.set_ylabel('Breadth (B)')
g.set_xlabel('Hidden (H)')

plt.title('AR breadth')
plt.tight_layout()
plt.savefig('fig/final/fig_ti/ar_breadth.svg')
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

g = sns.heatmap(mdf, square=True, vmin=0.6, vmax=0.9)
g.tick_params(axis='both', labelsize=10)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 33 - 0.5 * xs, color='cyan', linestyle='dashed')
g.text(10, 20, r'$\propto \sqrt{H}$', color='cyan', fontsize=12)

g.set_ylabel('Depth (D)')
g.set_xlabel('Hidden (H)')

plt.title('AR depth')
plt.tight_layout()
plt.savefig('fig/final/fig_ti/ar_depth.svg')
plt.show()


# <codecell>
df = collate_dfs('remote/16_scale_clean/gen', show_progress=True)
df


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
    & (mdf['name'] == 'DP')
    & (mdf['n_arms'] % 3 == 0)
]

plt.gcf().set_size_inches((3.5, 2.5))
g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='n_arms', legend='auto')
g.axvline(x=train_split, color='red', linestyle='dashed', alpha=0.7)
g.set_xlabel('Distance ($k$)')
g.set_ylabel('Accuracy')
g.set_title('DP')
g.legend(title='Breadth ($B$)', loc='upper right')
g.set_ylim((0.5, 1))

plt.tight_layout()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_hop_prop'] == n_hop_prop)
    & (mdf['depth'] == depth)
    & (mdf['name'] == 'AR')
]

plt.gcf().set_size_inches((3.5, 2.5))
g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='n_arms', legend='auto')
g.axvline(x=train_split, color='red', linestyle='dashed', alpha=0.7)
g.set_xlabel('Distance ($k$)')
g.set_ylabel('Accuracy')
g.set_title('CoT')
g.legend(title='Breadth ($B$)', loc='upper right')
g.set_ylim((0.5, 1))

plt.tight_layout()
plt.savefig('fig/final/fig_ti/ar_gen.svg')


# <codecell>
adf = mdf[mdf['test_n_hop'] > 15]
adf = adf[adf['test_n_hop'] % 2 == 0]

g = sns.lineplot(adf, x='n_arms', y='acc', hue='test_n_hop', alpha=0.5, legend='brief')
g.set_xscale('log')
g.set_yscale('log')

g.set_xlabel('Breadth ($B$)')
g.set_ylabel('Test accuracy')

xs = np.linspace(np.log(9.5), np.log(13))
g.plot(np.exp(xs), np.exp(-2 * xs + 4.45), color='red', linestyle='dashed')
g.text(4, 0.8, r'$\propto B^{-2}$', color='red')

g.legend(title='Distance ($k$)', loc='upper left')

plt.tight_layout()
plt.savefig('fig/final/app_fig_dp_gen_acc/dp_gen_acc.svg')

# <codecell>
df = collate_dfs('remote/16_scale_clean/gen_extend/set', show_progress=True)
df


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

plot_df = pd.concat((plot_df.drop('info', axis=1), bdf), axis=1)
plot_df

# <codecell>
names = plot_df['name'].unique()
depths = plot_df['depth'].unique()

for depth, name in itertools.product(depths, names):
    mdf = plot_df.copy()
    train_split = np.round(depth * 0.3)

    mdf = mdf[
        (mdf['depth'] == depth)
        & (mdf['name'] == name)
    ]

    plt.gcf().set_size_inches((2.7, 2))
    g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='n_arms', legend='full')
    g.axvline(x=train_split, color='red', linestyle='dashed', alpha=0.7)
    g.set_xlabel('Distance ($k$)')
    g.set_ylabel('Accuracy')
    g.set_title(name)
    g.legend(title='Breadth ($B$)', loc='upper right')
    g.set_ylim((0.5, 1))

    plt.tight_layout()
    plt.savefig(f'fig/final/app_fig_variation/{name}_{depth}.svg')
    plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['n_hop_prop'] == n_hop_prop)
    & (mdf['depth'] == depth)
    & (mdf['name'] == 'AR')
]

plt.gcf().set_size_inches((3.5, 2.5))
g = sns.lineplot(mdf, x='test_n_hop', y='acc', hue='n_arms', legend='auto')
g.axvline(x=train_split, color='red', linestyle='dashed', alpha=0.7)
g.set_xlabel('Distance ($k$)')
g.set_ylabel('Accuracy')
g.set_title('CoT')
g.legend(title='Breadth ($B$)', loc='upper right')
g.set_ylim((0.5, 1))

plt.tight_layout()
plt.savefig('fig/final/fig_ti/ar_gen.svg')


# <codecell>
adf = mdf[mdf['test_n_hop'] > 15]
adf = adf[adf['test_n_hop'] % 2 == 0]

g = sns.lineplot(adf, x='n_arms', y='acc', hue='test_n_hop', alpha=0.5, legend='brief')
g.set_xscale('log')
g.set_yscale('log')

g.set_xlabel('Breadth ($B$)')
g.set_ylabel('Test accuracy')

xs = np.linspace(np.log(9.5), np.log(13))
g.plot(np.exp(xs), np.exp(-2 * xs + 4.45), color='red', linestyle='dashed')
g.text(4, 0.8, r'$\propto B^{-2}$', color='red')

g.legend(title='Distance ($k$)', loc='upper left')

plt.tight_layout()