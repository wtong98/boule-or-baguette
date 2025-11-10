"""Qwen sweep plotting"""

# <codecell>
import pickle

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
from task.prop import *


# <codecell>
df1 = collate_dfs('remote/14_qwen_sweep/qwen', show_progress=True)
df2 = collate_dfs('remote/14_qwen_sweep/qwen_php', show_progress=True)

df = pd.concat([df1, df2], ignore_index=True)
df
# <codecell>
def extract_plot_vals(row):
    run_name = row['model_name'].rsplit('/', 1)[-1].split('-')[-1]
    prompt_type = row['run_name'].split(' ')[0].split('-')[-1]

    return pd.DataFrame({
        'name': [run_name] * len(row['res']),
        'prompt_type': prompt_type,
        'ckpt': row['ckpt_num'],
        'task': row['project_name'],
        'split': row['train_split'],
        'range': list(row['res']),
        'acc': [item['gen_acc'] for item in row['res'].values()],
    })

plot_df = pd.concat(
    [extract_plot_vals(row) for _, row in df.iterrows()],
    ignore_index=True,
)

plot_df

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_full')
    & (mdf['split'] == 4)
    & (mdf['range'] == 'range_(5, 9)')
    # & (mdf['range'] == 'range_(9, inf)')
]

g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
g.set_ylim((0.6, 0.97))
plt.title('Full dataset')
plt.savefig('fig/qwen_full_time.png')

# <codecell>
# last_ckpt = (
#     mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(1)
# )

# sns.barplot(data=last_ckpt, x='name', y='acc', hue='prompt_type')

first_ckpt = mdf[mdf['ckpt'] == 250]
sns.barplot(data=first_ckpt, x='name', y='acc', hue='prompt_type')

plt.ylim((0.6, 0.97))
plt.title('Full dataset (early)')
plt.savefig('fig/qwen_full_first.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_imply')
    & (mdf['split'] == 4)
    & (mdf['range'] == 'range_(5, 11)')
    # & (mdf['range'] == 'range_(11, inf)')
]

g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
g.set_ylim((0.6, 0.97))

plt.title('Imply dataset')
plt.savefig('fig/qwen_imply_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(1)
)

sns.barplot(data=last_ckpt, x='name', y='acc', hue='prompt_type')

# first_ckpt = mdf[mdf['ckpt'] == 250]

sns.barplot(data=last_ckpt, x='name', y='acc', hue='prompt_type')
# plt.ylim((0.6, 0.97))
plt.title('Imply dataset')
plt.savefig('fig/qwen_impy_last.png')


# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_or')
    & (mdf['split'] == 6)
    & (mdf['range'] == 'range_(7, 13)')
    # & (mdf['range'] == 'range_(13, inf)')
]

sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)

plt.title('Or dataset')
plt.savefig('fig/qwen_or_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(1)
)

first_ckpt = mdf[mdf['ckpt'] == 250]

sns.barplot(data=first_ckpt, x='name', y='acc', hue='prompt_type')
# plt.ylim((0.6, 1.1))
plt.title('Or dataset (early)')
plt.savefig('fig/qwen_or_early.png')

# %%
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_php')
    & (mdf['split'] == 60)
    & (mdf['range'] == 'range_(61, 121)')
    # & (mdf['range'] == 'range_(121, inf)')
]

sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
plt.title('PHP dataset')
plt.savefig('fig/qwen_php_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(1)
)

first_ckpt = mdf[mdf['ckpt'] == 250]

sns.barplot(data=first_ckpt, x='name', y='acc', hue='prompt_type')
# plt.ylim((0.6, 1.1))
plt.title('PHP dataset (early)')
plt.savefig('fig/qwen_php_early.png')