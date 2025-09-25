"""Exploring generalization on prop task"""

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
n_hidden = 512
batch_size = 128

cot = False
depth = 8

train_task = PropTask(depth, split='train', cot=cot, batch_size=batch_size, ds_path=or_ds_path)
test_task = PropTask(depth, split='test', cot=cot, batch_size=batch_size, ds_path=or_ds_path)

n_vocab = len(train_task.tokenizer)

xs, ys = next(train_task)
print(xs[:3])
print(ys[:3])

# <codecell>
train_task.tokenizer.decode(xs[-19])

# <codecell>
# config = TrConfig(n_vocab=n_vocab, 
#                   pos_emb=not cot,
#                   rand_pos_emb=True,
#                   big_pe=False,
#                   n_out=n_vocab if cot else 1,
#                   n_hidden=n_hidden, 
#                   return_format=None if cot else True)

config = TransformerConfig(n_layers=1,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                           pos_emb=True,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           freeze_emb=False,
                           use_bias=True,
                           return_format=None if cot else 'final_logit_up_to_pad',
                           mup_scale=True,
                           linear_att=False
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask' if cot else 'bce',
                    test_every=100,
                    test_iters=1,
                    train_iters=100_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot_prop] if cot else None,
                    print_fn=print_gen if cot else None,
                    lr=3e-5
                    )


# <codecell>
# df = collate_dfs('remote/11_prop/length', show_progress=True)
# df = collate_dfs('remote/11_prop/width', show_progress=True)
# df = collate_dfs('remote/11_prop/width/set2_imply', show_progress=True)
# df1 = collate_dfs('remote/11_prop/ar/set1_good', show_progress=True)
# df2 = collate_dfs('remote/11_prop/direct/set1_good', show_progress=True)
df1 = collate_dfs('remote/11_prop/ar', show_progress=True)
df2 = collate_dfs('remote/11_prop/direct', show_progress=True)
df = pd.concat([df1, df2], ignore_index=True)
df

# <codecell>
# rand_idxs = np.random.choice(len(df), size=100, replace=False)
for ex in df1['hist']:
    vals = [p['loss'] for p in ex['train']]
    plt.plot(vals, color='C0', alpha=0.1)

# %%
def extract_plot_vals(row):
    if 'n_hop' in row['info']:
        train_hop = row['info']['n_hop']
        new_info = row['info'].copy()
        new_info.pop('n_hop')
    else:
        train_hop = 0
        new_info = row['info']
    
    return pd.Series([
        row['name'],
        train_hop,
        new_info
    ], index=['name', 'train_hop', 'info'])

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

n_hop_val = [v[0] for v in plot_df['test_n_hop']]
plot_df['test_n_hop'] = n_hop_val
plot_df

# <codecell>
# for n_hop in [2, 4, 6, 10]:
for n_hop in [6]:
    mdf = plot_df.copy()
    mdf = mdf[mdf['train_hop'] == n_hop]

    g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', hue_order=['Direct_full', 'Direct_imply', 'AR_full', 'AR_trunc', 'AR_imply'], estimator='max')
    # g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', hue_order=['Direct_or', 'AR_or'], estimator='mean')
    g.set_ylim(0.4, 1)
    g.axhline(y=0.5, color='brown', linestyle='dashed', alpha=0.5)
    g.set_title('Train hop: ' + str(n_hop))

    # plt.savefig(f'fig/prop_task_n_hop_{n_hop}.png')
    plt.show()


# <codecell>
df1 = collate_dfs('remote/11_prop/ar_med_sweep', show_progress=True)
df2 = collate_dfs('remote/11_prop/ar_lg_sweep', show_progress=True)
df = pd.concat([df1, df2], ignore_index=True)
df

# <codecell>
for ex in df2['hist']:
    vals = [p['loss'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# %%
def extract_plot_vals(row):
    if 'n_hop' in row['info']:
        train_hop = row['info']['n_hop']
        new_info = row['info'].copy()
        new_info.pop('n_hop')
    else:
        train_hop = 0
        new_info = row['info']
    
    return pd.Series([
        row['name'],
        train_hop,
        new_info
    ], index=['name', 'train_hop', 'info'])

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

n_hop_val = [v[0] for v in plot_df['test_n_hop']]
plot_df['test_n_hop'] = n_hop_val
plot_df

# <codecell>
# for n_hop in [2, 4, 6, 10]:
for n_hop in [6]:
    mdf = plot_df.copy()
    mdf = mdf[mdf['train_hop'] == n_hop]

    g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', estimator='max')
    # g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', hue_order=['Direct_or', 'AR_or'], estimator='mean')
    # g.set_ylim(0.4, 1)
    g.axhline(y=0.5, color='brown', linestyle='dashed', alpha=0.5)
    g.set_title('Train hop: ' + str(n_hop))

    # plt.savefig(f'fig/prop_task_n_hop_{n_hop}.png')
    plt.show()


# <codecell>
### OR experiment
df = collate_dfs('remote/11_prop/direct_or', show_progress=True)
df

# <codecell>
for ex in df['hist']:
    vals = [p['acc'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# %%
def extract_plot_vals(row):
    if 'n_hop' in row['info']:
        train_hop = row['info']['n_hop']
        new_info = row['info'].copy()
        new_info.pop('n_hop')
    else:
        train_hop = 0
        new_info = row['info']
    
    return pd.Series([
        row['name'],
        train_hop,
        new_info
    ], index=['name', 'train_hop', 'info'])

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

n_hop_val = [v[0] for v in plot_df['test_n_hop']]
plot_df['test_n_hop'] = n_hop_val
plot_df

# <codecell>
# for n_hop in [2, 4, 6, 10]:
for n_hop in [6]:
    mdf = plot_df.copy()
    mdf = mdf[mdf['train_hop'] == n_hop]

    g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', estimator='max')
    # g = sns.barplot(mdf, x='test_n_hop', y='acc', hue='name', hue_order=['Direct_or', 'AR_or'], estimator='mean')
    # g.set_ylim(0.4, 1)
    g.axhline(y=0.5, color='brown', linestyle='dashed', alpha=0.5)
    g.set_title('Train hop: ' + str(n_hop))

    # plt.savefig(f'fig/prop_task_n_hop_{n_hop}.png')
    plt.show()


# <codecell>
### MODEL INSPECTION
with open('remote/11_prop/weights/AR_full_4_weights.pkl', 'rb') as fp:
    params = pickle.load(fp)

# <codecell>
task = PropTask(5, split='train', cot=True, batch_size=8, max_len=1024, padding='max_length', ds_path=full_ds_path)
xs, ys = next(task)
xs.shape

# <codecell>
print(task.tokenizer.decode(xs[0]))

# <codecell>
cfg = TransformerConfig(n_heads=12, 
                    n_out=len(task.tokenizer),
                    n_vocab=len(task.tokenizer),
                    n_layers=12,
                    n_hidden=768,
                    pos_emb=False, 
                    return_format=None,
                    n_mlp_layers=2,
                    layer_norm=True,
                    residual_connections=True,
                    mup_scale=True,
                    linear_att=False)

state = create_train_state(model=cfg.to_model(), params=params)
state.apply_fn({'params': state.params}, xs)

# %%
out = fast_gen_acc_cot_prop(state, (xs, ys), seed=new_seed(), return_preds=True)
out

# <codecell>
def dec(ids):
    return print(task.tokenizer.decode(ids))

dec(out['preds'][-1])
dec(xs[-1])

# <codecell>
yes_id = 13138
no_id = 32165

np.argmax(out['preds'][1] == no_id)

# <codecell>
def score(xs, preds):
    is_true = jnp.argmax(xs == yes_id, axis=-1) > 0
    t = jnp.argmax(preds == yes_id, axis=-1)
    f = jnp.argmax(preds == no_id, axis=-1)

    pred_is_true = (t != 0) * ((f == 0) + (t < f))
    pred_is_false = (f != 0) * ((t == 0) + (f < t))

    true_pos = is_true * pred_is_true
    true_neg = (1 - is_true) * pred_is_false
    false_pos = (1 - is_true) * pred_is_true
    false_neg = is_true * pred_is_false

    return true_pos, true_neg, false_pos, false_neg

