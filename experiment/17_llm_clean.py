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
dirs = [
    # 'remote/15_llm_sweep/gemma',
    # 'remote/15_llm_sweep/gemma_or',
    # 'remote/15_llm_sweep/gemma_php_par',
    # 'remote/15_llm_sweep/gemma_php_enum/set6_tmp',
    'remote/17_llm_clean/qwen/set',
    # 'remote/15_llm_sweep/qwen_or',
    # 'remote/15_llm_sweep/qwen_php_par',
    # 'remote/15_llm_sweep/qwen_php_enum',
]

dfs = [collate_dfs(d, show_progress=True) for d in dirs]
df = pd.concat(dfs, ignore_index=True)
df

# NOTE: why are there NaN entries?
len_before = len(df)
df = df[pd.notna(df['model_name'])]
len_after = len(df)

print(f'Removed {len_before - len_after} NaN entries')
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
plot_df['task'].unique()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    # (mdf['task'] == 'gemma_prop_full')
    (mdf['task'] == 'qwen_full')
    & (mdf['range'] == 'range_(5, inf)')
    # & (mdf['range'] == 'range_(5, 8)')
    # & (mdf['range'] == 'range_(8, 12)')
    # & (mdf['range'] == 'range_(12, 15)')
    # & (mdf['range'] == 'range_(15, inf)')
    
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

# last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'])
# g = sns.barplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'], errorbar=None)
# g = sns.stripplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'])

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
    # (mdf['task'] == 'gemma_prop_imply')
    (mdf['task'] == 'qwen_imply')
    & (mdf['range'] == 'range_(7, inf)')
    # & (mdf['range'] == 'range_(7, 21)')
    # & (mdf['range'] == 'range_(21, 31)')
    # & (mdf['range'] == 'range_(31, 46)')
    # & (mdf['range'] == 'range_(46, inf)')
]


g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
# g.set_ylim((0.6, 0.97))

plt.title('Imply dataset')
# plt.savefig('fig/qwen_imply_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(2)
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
    (mdf['task'] == 'gemma_prop_or')
    # (mdf['task'] == 'prop_or')
    # & (mdf['range'] == 'range_(1, 19)')
    # & (mdf['range'] == 'range_(19, 31)')
    # & (mdf['range'] == 'range_(31, 46)')
    # & (mdf['range'] == 'range_(46, inf)')
    & (mdf['range'] == 'range_(19, inf)')
    
]

sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)

plt.title('Or dataset')
# plt.savefig('fig/qwen_or_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
    # mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).head(6)
)

last_ckpt = mdf[mdf['ckpt'] == 2000]
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
    (mdf['task'] == 'prop_php_par')
    # (mdf['task'] == 'gemma_prop_php_par')
    # & (mdf['range'] == 'range_(1, 81)')
    # & (mdf['range'] == 'range_(81, 171)')
    # & (mdf['range'] == 'range_(171, 281)')
    # & (mdf['range'] == 'range_(281, inf)')
    & (mdf['range'] == 'range_(331, inf)')
    # & (mdf['range'] == 'range_(81, inf)')
]


g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.title('PHP dataset')
# plt.savefig('fig/qwen_php_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
)


last_ckpt = mdf[mdf['ckpt'] == 500]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

# g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', order=['0.5B', '1.5B', '7B'], hue_order=['DP', 'CoT'])
g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', hue_order=['DP', 'CoT'])
g.legend(title=None)
# g.legend_.set_visible(False)


g.set_ylim((0.3, 1))
g.set_title('PHP (narrow, very deep)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()
# plt.savefig('fig/qwen_php.png')


# %%
mdf = plot_df.copy()
mdf = mdf[
    # (mdf['task'] == 'gemma_prop_php_enum')
    (mdf['task'] == 'gemma_prop_php_par')
    & (mdf['range'] == 'range_(61, inf)')
    # & (mdf['range'] == 'range_(91, 121)')
    # & (mdf['range'] == 'range_(121, 171)')
    # & (mdf['range'] == 'range_(171, inf)')
    # & (mdf['range'] == 'range_(91, inf)')
]

# mdf['range'].unique()

# # <codecell>


g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True, estimator='mean')
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.title('PHP dataset')
# plt.savefig('fig/qwen_php_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
)


# last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

g = sns.boxplot(data=last_ckpt, x='name', y='acc', hue='prompt_type', order=['0.5B', '1.5B', '7B'], hue_order=['DP', 'CoT'])
g.legend(title=None)
# g.legend_.set_visible(False)


# g.set_ylim((0.3, 1))
g.set_title('PHP (narrow, very deep)')
g.set_xlabel('Qwen2.5 Coder Size')
g.set_ylabel('Accuracy')
g.figure.tight_layout()
# plt.savefig('fig/qwen_php.png')