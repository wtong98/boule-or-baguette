"""Experimenting with RL finetuning"""


# <codecell>
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import TransformerConfig 
from task.graph import *


# <codecell>
df = collate_dfs('remote/3_rl/generalize', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    t1, _ = row['train_task'].tasks
    d1 = t1.samp_dist[1] if isinstance(t1.samp_dist, Iterable) else t1.samp_dist

    return pd.Series([
        row['name'],
        row['config']['n_layers'],
        row['config']['n_hidden'],
        d1,
        row['info']['etc']['train_len_rl'],
        row['info']
    ], index=['name', 'n_layer', 'n_hidden', 'dist_pr', 'dist_rl', 'info'])

plot_df = df.apply(extract_plot_vals, axis=1) \
            .reset_index(drop=True) \

adf = pd.DataFrame(plot_df['info'].tolist()) \
        .stack() \
        .reset_index(level=1, name='info')

adf = adf[adf['level_1'] != 'etc']

plot_df = plot_df.drop('info', axis='columns').join(adf)

plot_df

ldf = plot_df['level_1'].str.split('_', expand=True) \
                        .rename(columns={
                            0: 'branch',
                            1: 'mode',
                            2: 'test_len',
                        })

plot_df = pd.concat((plot_df, ldf), axis='columns').drop('level_1', axis='columns')
adf = pd.DataFrame(plot_df['info'].to_list())
plot_df = pd.concat((plot_df.drop('info', axis=1).reset_index(), adf), axis=1)
plot_df

# <codecell>
branches = ['on', 'off']

for branch in branches:
    mdf = plot_df.copy()
    mdf = mdf[(mdf['branch'] == branch)
              & (mdf['n_hidden'] == 1024)
              & (mdf['n_layer'] == 2)]

    gs = sns.relplot(mdf, x='test_len', y='gen_acc', hue='mode', col='dist_pr', row='dist_rl', marker='o', height=1.5, aspect=1.2, alpha=0.7)
    fig = gs.figure
    fig.suptitle(f'branch={branch}')
    fig.subplots_adjust(top=0.88)

    # plt.savefig(f'fig/acc_rl_stable_{branch}.png')
    plt.show()



# <codecell>
mdf = plot_df.copy()
mdf[(mdf['dist_pr'] == 1) 
    & (mdf['dist_rl'] == 3)
    & (mdf['n_hidden'] == 128)
    & (mdf['n_layer'] == 2)
    & (mdf['branch'] == 'on')
    ]

# <codecell>
plt.plot(df.iloc[4]['info']['etc']['rl_hist']['rew'], '--o')


# <codecell>
depth = 10
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hidden = 128
batch_size = 32
unwrap = False

seed = new_seed()


train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size),
    BinaryTreeTiTask(depth=depth, samp_dist=(1), on_branch=False, fill_gaps=False, cot=True, unwrap=unwrap, batch_size=batch_size))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=8, on_branch=True, cot=True, unwrap=unwrap, batch_size=batch_size)

config = TransformerConfig(n_layers=2,
                           n_vocab=n_vocab,
                           n_out=n_vocab,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=2,
                           layer_norm=True,
                           as_rf_model=False,
                           residual_connections=True,
                           use_simple_att=False,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False,
                           )


# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=50_000,
                    # lr=1e-3,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen
                    )


# <codecell>
with open('state.pkl', 'wb') as fp:
    pickle.dump(state.params, fp)

# <codecell>
with open('state.pkl', 'rb') as fp:
    params = pickle.load(fp)

state = create_train_state(
    model=config.to_model(), 
    params=params,
    optim=optax.sgd,
    lr=5e-4
    )

# <codecell>
batch_size = 1024

train_task = Chain(
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None),
    BinaryTreeTiTask(depth=depth, samp_dist=(1, 3), on_branch=False, fill_gaps=False, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None))

test_task = BinaryTreeTiTask(depth=depth, samp_dist=3, on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None)

### RL finetune
state, hist = reinforce(state, train_task, 
                        test_iter=test_task,
                        action_fn=generate, 
                        reward_fn=bt_rew_fn, 
                        rl_loss=bt_rl_loss,
                        train_iters=100_000,
                        test_every=1000,
                        test_iters=10,
                        use_tqdm=True,
                        eval_fns=[gen_acc_rl]
                        )



# <codecell>
# task = Chain(
#     BinaryTreeTiTask(depth=depth, samp_dist=(8), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None),
#     BinaryTreeTiTask(depth=depth, samp_dist=(8), on_branch=False, fill_gaps=False, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None))

task = BinaryTreeTiTask(depth=depth, samp_dist=(8), on_branch=True, rl_prompt=True, unwrap=unwrap, batch_size=batch_size, n_thought=None)

batch = next(task)

gen_acc_rl(state, batch)

# <codecell>
task.cot = True
xs, _ = next(task)
# logits = state.apply_fn({'params': state.params}, xs)
# preds = logits.argmax(-1)
preds = generate(state, xs)

print(preds[:3])
print(xs[:3])

# <codecell>
state.params

# <codecell>
with open('state.pkl', 'rb') as fp:
    params = pickle.load(fp)

jax.tree.map(lambda x,y: np.mean((x - y))**2, state.params, params)

# <codecell>
for t in task.tasks:
    t.cot = True

xs, ys = next(task)
traj = generate(state, xs)

print(traj[:3])
print(ys[:3])
