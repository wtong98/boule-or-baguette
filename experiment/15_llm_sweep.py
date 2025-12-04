"""LLM sweep plotting"""

# <codecell>
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import collate_dfs, set_theme

# set_theme()

# <codecell>
# df1 = collate_dfs('remote/15_llm_sweep/gemma', show_progress=True)
df2 = collate_dfs('remote/15_llm_sweep/qwen_php', show_progress=True)
# df3 = collate_dfs('remote/15_llm_sweep/qwen_or_exp', show_progress=True)

df = df2
# df = pd.concat([df1, df2, df3], ignore_index=True)

# <codecell>
# NOTE: why are there NaN entries?
len_before = len(df)
df = df[pd.notna(df['model_name'])]
len_after = len(df)

print(f'Removed {len_before - len_after} NaN entries')
# <codecell>
def extract_plot_vals(row):
    run_name = row['run_name'].split('-')[2]
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
    (mdf['task'] == 'gemma_prop_full')
    & (mdf['split'] == 4)
    & (mdf['range'] == 'range_(5, inf)')
    # & (mdf['range'] == 'range_(10, inf)')
]


g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
# g.set_ylim((0.6, 0.97))
plt.title('Full dataset')
# plt.savefig('fig/qwen_full_time.png')


# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
    # mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).head(3)
)

last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'])
g.legend(title=None)
sns.move_legend(g, "lower right")

# g.set_ylim((0.6, 0.97))
g.set_title('Full (broad, shallow)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

# plt.savefig('fig/qwen_full.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'gemma_prop_imply')
    & (mdf['split'] == 6)
    & (mdf['range'] == 'range_(7, inf)')
]


g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
# g.set_ylim((0.6, 0.97))

plt.title('Imply dataset')
# plt.savefig('fig/qwen_imply_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
)

# last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'])
g.legend(title=None)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

# g.set_ylim((0.6, 0.97))
g.set_title('Imply (medium breadth, depth)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

# plt.savefig('fig/qwen_impy.png')


# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'exp_prop_or')
    & (mdf['split'] == 12)
    & (mdf['range'] == 'range_(13, inf)')
]

sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)

plt.title('Or dataset')
# plt.savefig('fig/qwen_or_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
    # mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).head(6)
)

# last_ckpt = mdf[mdf['ckpt'] == 750]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', order=['0.5B', '1.5B', '7B', '32B'], hue_order=['DP', 'CoT'])
# g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', order=['0.5B', '7B', '32B'], hue_order=['DP', 'CoT'])
g.legend(title=None)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

# g.set_ylim((0.5, 1.1))
g.set_title('Or (very narrow, deep)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

# plt.savefig('fig/qwen_or.png')

# %%
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'prop_php')
    & (mdf['split'] == 80)
    # & (mdf['range'] == 'range_(1, 81)')
    # & (mdf['range'] == 'range_(81, 161)')
    # & (mdf['range'] == 'range_(161, 221)')
    & (mdf['range'] == 'range_(221, inf)')
    # & (mdf['range'] == 'range_(1, 61)')
]

g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.title('PHP dataset')
# plt.savefig('fig/qwen_php_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
)


last_ckpt = mdf[mdf['ckpt'] == 1000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', order=['0.5B', '1.5B', '7B'], hue_order=['DP', 'CoT'])
g.legend(title=None)
# g.legend_.set_visible(False)


# g.set_ylim((0.6, 0.97))
g.set_title('PHP (narrow, very deep)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()
# plt.savefig('fig/qwen_php.png')