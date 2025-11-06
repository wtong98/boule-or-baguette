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
df = collate_dfs('remote/14_qwen_sweep/qwen', show_progress=True)
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
]

g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
g.set_ylim((0.6, 0.97))


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

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_or')
    & (mdf['split'] == 6)
    & (mdf['range'] == 'range_(7, 13)')
    # & (mdf['range'] == 'range_(13, inf)')
]

sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
