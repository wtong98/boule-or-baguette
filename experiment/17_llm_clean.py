"""LLM sweep plotting"""

# <codecell>
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import collate_dfs, set_theme
import numpy as np

set_theme()

# <codecell>
dirs = [
    # 'remote/17_llm_clean/gemma/set',
    # 'remote/17_llm_clean/gemma_or/set',
    # 'remote/17_llm_clean/gemma_php_enum/set',
    # 'remote/17_llm_clean/qwen/set',
    # 'remote/17_llm_clean/qwen_or/set',
    # 'remote/17_llm_clean/qwen_php_enum/set',
    'remote/17_llm_clean/llama_ege',
    'remote/17_llm_clean/llama_ege_or',
    'remote/17_llm_clean/llama_php_enum/set',
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
    # (mdf['task'] == 'qwen_full')
    (mdf['task'] == 'llama3_prop_full')
    & (mdf['range'] == 'range_(5, inf)')
    # & (mdf['range'] == 'range_(5, 8)')
    # & (mdf['range'] == 'range_(8, 12)')
    # & (mdf['range'] == 'range_(12, 15)')
    # & (mdf['range'] == 'range_(15, inf)')
    
]

# g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
g = sns.lineplot(mdf, x='ckpt', y='acc', hue='prompt_type', style='name', markers=True)
# g.set_ylim((0.6, 0.97))
plt.title('Full dataset')
# plt.savefig('fig/qwen_full_time.png')


# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
    # mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).head(3)
)

last_ckpt = mdf[mdf['ckpt'] == 2000]
# last_ckpt = mdf[mdf['ckpt'] == 2500]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'RT'}})

# plt.gcf().set_size_inches(5, 2.5)
# plt.gcf().set_size_inches(3.8, 2.5)
plt.gcf().set_size_inches(3.6, 2.5)

# order = ['0.5B', '1.5B', '3B', '7B', '14B', '32B']
# order = ['1b', '4b', '12b', '27b']
order = ['1B', '3B', '8B']
g = sns.boxplot(
    data=last_ckpt, 
    x='name', y='acc', hue='prompt_type', 
    hue_order=['DP', 'RT'], order=order, 
    fliersize=2, fill=False, gap=0.1)

g.axhline(0.55, ls='--', color='gray')
g.legend(title=None)
sns.move_legend(g, "lower right")

g.set_ylim((0, 1))
g.set_title('Full')
# g.set_xlabel('Model (Qwen2.5-Coder)')
g.set_xlabel('Model (Llama 3.*)')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

plt.savefig('fig/final/fig_pita_llama/llama_full.svg')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    # (mdf['task'] == 'gemma_prop_imply')
    # (mdf['task'] == 'qwen_imply')
    (mdf['task'] == 'llama3_prop_imply')
    & (mdf['range'] == 'range_(7, inf)')
    # & (mdf['range'] == 'range_(7, 21)')
    # & (mdf['range'] == 'range_(21, 31)')
    # & (mdf['range'] == 'range_(31, 46)')
    # & (mdf['range'] == 'range_(46, inf)')
]


# g = sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
g = sns.lineplot(mdf, x='ckpt', y='acc', hue='prompt_type', style='name', markers=True)
# g.set_ylim((0.6, 0.97))

plt.title('Imply dataset')
# plt.savefig('fig/qwen_imply_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(2)
)

last_ckpt = mdf[mdf['ckpt'] == 2000]
# last_ckpt = mdf[mdf['ckpt'] == 2500]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

plt.gcf().set_size_inches(3.6, 2.5)
# order = ['0.5B', '1.5B', '3B', '7B', '14B', '32B']
order = ['1B', '3B', '8B']

g = sns.boxplot(
    data=last_ckpt, 
    x='name', y='acc', hue='prompt_type', 
    hue_order=['DP', 'CoT'], order=order, 
    fliersize=2, fill=False, gap=0.1)
g.legend(title=None)
g.set_ylim((0, 1))
g.axhline(0.15, ls='--', color='gray')
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

g.set_title('Imply')
g.set_xlabel('Model (Llama 3.*)')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

# plt.savefig('fig/final/fig_pita_res/qwen25_imply.svg')
# plt.savefig('fig/final/fig_pita_gemma/gemma_imply.svg')
plt.savefig('fig/final/fig_pita_llama/llama_imply.svg')


# <codecell>
mdf = plot_df.copy()
mdf = mdf[
    (mdf['task'] == 'llama3_or')
    # (mdf['task'] == 'gemma_or')
    # (mdf['task'] == 'qwen_or')
    # & (mdf['range'] == 'range_(1, 19)')
    # & (mdf['range'] == 'range_(19, 31)')
    # & (mdf['range'] == 'range_(31, 46)')
    # & (mdf['range'] == 'range_(46, inf)')
    & (mdf['range'] == 'range_(19, inf)')
    
]

# sns.lineplot(mdf, x='ckpt', y='acc', hue='name', style='prompt_type', markers=True)
sns.lineplot(mdf, x='ckpt', y='acc', hue='prompt_type', style='name', markers=True)

plt.title('Or dataset')
# plt.savefig('fig/qwen_or_time.png')

# <codecell>
last_ckpt = (
    mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).tail(3)
    # mdf.sort_values('ckpt').groupby(['name', 'prompt_type'], as_index=False).head(6)
)

last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})

plt.gcf().set_size_inches(3.6, 2.5)
# order = ['0.5B', '1.5B', '3B', '7B', '14B', '32B']
order=['1B', '3B', '8B']

g = sns.boxplot(
    data=last_ckpt, 
    x='name', y='acc', hue='prompt_type', 
    hue_order=['DP', 'CoT'], order=order, 
    fliersize=2, fill=False, gap=0.1)
g.legend(title=None)

g.axhline(0.47, ls='--', color='gray')

g.set_ylim((0, 1))
g.set_title('Or')
g.set_xlabel('Model (Llama 3.*)')
g.set_ylabel('Accuracy')
g.figure.tight_layout()

# plt.savefig('fig/final/fig_pita_res/qwen25_or.svg')
# plt.savefig('fig/final/fig_pita_gemma/gemma_or.svg')
plt.savefig('fig/final/fig_pita_llama/llama_or.svg')


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
    (mdf['task'] == 'llama_php_enum')
    # (mdf['task'] == 'gemma_php_enum')
    # (mdf['task'] == 'qwen_php_enum')
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


last_ckpt = mdf[mdf['ckpt'] == 2000]
last_ckpt = last_ckpt.replace({'prompt_type': {'dp': 'DP', 'cot': 'Chain-of-Thought', 'ar_cot': 'CoT'}})


plt.gcf().set_size_inches(3.6, 2.5)
# order = ['0.5B', '1.5B', '3B', '7B',]
# order = ['1b', '4b']
order = ['1B', '3B', '8B']

g = sns.boxplot(
    data=last_ckpt, 
    x='name', y='acc', hue='prompt_type', 
    hue_order=['DP', 'CoT'], order=order, 
    fliersize=2, fill=False, gap=0.1)
g.legend(title=None)
g.axhline(0.58, ls='--', color='gray')

g.set_ylim((0, 1))


g.set_title('PHP')
g.set_xlabel('Model (Llama 3.*)')
g.set_ylabel('Accuracy')
g.figure.tight_layout()
# plt.savefig('fig/final/fig_pita_res/qwen25_php.svg')
# plt.savefig('fig/final/fig_pita_gemma/gemma_php_enum.svg')
plt.savefig('fig/final/fig_pita_llama/llama_php_enum.svg')



# <codecell>
### Granular error plotting
dirs = [
    'remote/17_llm_clean/qwen/set_granular',
    # 'remote/17_llm_clean/qwen_or/set_granular',
    # 'remote/17_llm_clean/qwen_php_enum/set_granular',
]

dfs = [collate_dfs(d, show_progress=True) for d in dirs]
df = pd.concat(dfs, ignore_index=True)


# <codecell>
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
        'true_pos': [item['true_pos'] for item in row['res'].values()],
        'true_neg': [item['true_neg'] for item in row['res'].values()],
        'false_pos': [item['false_pos'] for item in row['res'].values()],
        'false_neg': [item['false_neg'] for item in row['res'].values()],
        'prop_none': [item['prop_none'] for item in row['res'].values()],
    })

plot_df = pd.concat(
    [extract_plot_vals(row) for _, row in df.iterrows()],
    ignore_index=True,
)

plot_df
plot_df['task'].unique()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[(mdf['ckpt'] == 2000)
    & (mdf['range'] == 'range_(5, inf)')
    & (mdf['task'] == 'qwen_full')]

mdf = mdf.replace({'prompt_type': {'dp': 'DP', 'ar_cot': 'RT'}})

# Group by name and prompt_type, then average the metrics
grouped = mdf.groupby(['name', 'prompt_type'])[['true_pos', 'true_neg', 'false_pos', 'false_neg', 'prop_none']].mean()

# Define colors - blue for true_pos/false_neg, red for true_neg/false_pos, gray for prop_none
colors = {
    'true_pos': '#89CFF0',    # pastel blue
    'false_neg': '#A7C7E7',   # lighter pastel blue
    'true_neg': '#FFB3BA',    # pastel red
    'false_pos': '#FFCCCB',   # lighter pastel red
    'prop_none': '#D3D3D3',   # light gray
}

# Order: true_pos, false_neg, prop_none, true_neg, false_pos (to group blues and reds together)
order = ['true_pos', 'false_neg', 'prop_none', 'true_neg', 'false_pos']

order_names = {
    'true_pos': 'TP',
    'false_neg': 'FN',
    'true_neg': 'TN',
    'false_pos': 'FP',
    'prop_none': 'None',
}

# Get unique model sizes and prompt types
model_sizes = sorted(grouped.index.get_level_values('name').unique(), 
                     key=lambda x: float(x.replace('B', '')))
prompt_types = ['DP', 'RT']

n_cols = len(model_sizes)
n_rows = len(prompt_types)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
if n_cols == 1:
    axes = axes.reshape(-1, 1)

for row_idx, prompt_type in enumerate(prompt_types):
    for col_idx, name in enumerate(model_sizes):
        ax = axes[row_idx, col_idx]
        
        if (name, prompt_type) in grouped.index:
            row = grouped.loc[(name, prompt_type)]
            values = np.array([row[col] for col in order])
            pie_colors = np.array([colors[col] for col in order])
            
            keep_idx = values > 0
            values = values[keep_idx]
            pie_colors = pie_colors[keep_idx]
            curr_order = np.array(order)[keep_idx]
            curr_order = [order_names[o] for o in curr_order]
            
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=curr_order, 
                colors=pie_colors, 
                autopct='%1.1f%%',
                startangle=90,
            )
            for text in texts:
                text.set_fontsize(14)
            for autotext in autotexts:
                autotext.set_fontsize(12)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        
        if row_idx == 0:
            ax.set_title(f'{name}', fontsize=20)
        if col_idx == 0:
            ax.set_ylabel(prompt_type, fontsize=20)

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.savefig('fig/final/app_fig_breakdown/full.svg', bbox_inches='tight')
plt.show()
