"""Exploring generalization on prop task"""

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
from model.transformer import *
from task.prop import *

# <codecell>
n_hidden = 512
batch_size = 128

cot = False
depth = 3

train_task = PropTask(depth, split='train', cot=cot, batch_size=batch_size)
test_task = PropTask(depth, split='test', cot=cot, batch_size=batch_size)

n_vocab = len(train_task.tokenizer)

xs, ys = next(train_task)
print(xs[:3])
print(ys[:3])

# <codecell>
# config = TrConfig(n_vocab=n_vocab, 
#                   pos_emb=not cot,
#                   rand_pos_emb=True,
#                   big_pe=False,
#                   n_out=n_vocab if cot else 1,
#                   n_hidden=n_hidden, 
#                   return_format=None if cot else True)

config = TransformerConfig(n_layers=2,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           freeze_emb=True,
                           use_bias=False,
                           return_format=None if cot else 'final_logit_up_to_pad',
                           mup_scale=True,
                           linear_att=False
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    test_iters=1,
                    train_iters=100_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot_prop] if cot else None,
                    print_fn=print_gen if cot else None,
                    )


# <codecell>
# df = collate_dfs('remote/11_prop/length', show_progress=True)
df = collate_dfs('remote/11_prop/width', show_progress=True)
df

# %%
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info'],
    ], index=['name', 'info'])

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
g = sns.barplot(plot_df, x='test_n_hop', y='acc', hue='name', hue_order=['Zero', 'Zero (small)', 'AR full'])
g.set_ylim(0.4, 1)
g.axhline(y=0.5, color='brown', linestyle='dashed', alpha=0.5)
g.set_title('Full dataset')

# plt.savefig('fig/prop_task_full.png')
