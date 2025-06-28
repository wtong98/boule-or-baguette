"""Transformer operation as logistic regression"""


# <codecell>
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from common import *
from train import *
from model.transformer import *
from task.graph import *

# <codecell>
depth = 10
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 512
batch_size = 128

n_layers = 1

cot = True
ttr = False
nouveau = True

train_task = StarfishTask(depth=depth, samp_dist=(1,3), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(depth=depth, samp_dist=5, batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

# config = TransformerConfig(n_layers=n_layers,
#                            n_vocab=n_vocab,
#                            n_out=n_vocab if cot else 1,
#                            n_hidden=n_hidden,
#                            pos_emb=False,
#                            max_len=100,
#                            n_mlp_layers=0,
#                            n_heads=1,
#                            layer_norm=False,
#                            as_rf_model=False,
#                            residual_connections=False,
#                            freeze_emb=True,
#                            use_bias=False,
#                            return_format=None if cot else True,
#                            mup_scale=True,
#                            linear_att=True
#                            )

# config = TrLogRegConfig(n_vocab=n_vocab, 
#                         pos_emb=True,
#                         n_out=n_vocab if cot else 1,
#                         n_hidden=n_hidden, 
#                         flatten=False,
#                         return_format=None if cot else True)

config = TrConfig(n_vocab=n_vocab, 
                  pos_emb=False,
                  n_out=n_vocab if cot else 1,
                  n_hidden=n_hidden, 
                  return_format=None if cot else True)

# xs, ys = next(train_task)

# print(xs[:3])
# print(ys[:3])

# fix_emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)

# <codecell>
# state = create_train_state(jax.random.key(new_seed()),
#                            config.to_model(),
#                            next(train_task)[0],
#                            lr=1e-3)

# state.params['Embed_freeze']['embedding'] = fix_emb

state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    print_fn=print_gen if cot else None,
                    lr=1e-3
                    )


# <codecell>
emb = np.array(state.params['Embed_freeze']['embedding'])
A = np.array(state.params['Dense_0']['kernel'])
W = np.array(state.params['Dense_1']['kernel'])

# <codecell>
coeff = np.linalg.pinv(emb @ emb.T) @ emb @ W
W_est = emb.T @ coeff

out = np.einsum('ai,bj->abij', emb, emb)
out = out.reshape(out.shape[0]**2, -1)

a_coeff = np.linalg.pinv(out @ out.T) @ out @ A.reshape(-1, 1)
A_est = out.T @ a_coeff
A_est = A_est.reshape(A.shape)

xs, ys = next(train_task)
X = emb[xs]

logits = np.tril(X @ t(A_est) @ t(X)) @ X @ W_est
logits = logits

true_logits = state.apply_fn({'params': state.params}, xs)
np.mean(np.abs(logits - true_logits))

# <codecell>
# EXPLICATION OF FAILURE MODE
xs = jnp.array([[3, 6, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
gen2(state, xs, beta=1)

# <codecell>
xs = jnp.array([[3, 33, 50]])
X = emb[xs]

att = X @ t(A) @ t(X)
logits = np.tril(att) @ X @ W
plt.plot(logits[0, -1])

np.round(np.tril(att), decimals=2)

# <codecell>
out = X @ W
plt.imshow(out.squeeze(), cmap='BrBG', vmin=-5, vmax=5)
# plt.plot(out.squeeze().T, '--o', alpha=0.8)
# plt.plot(logits[0, -1] / 4, '--o', color='gray', alpha=0.5)
plt.axvline(48, color='magenta', alpha=0.3)
# plt.colorbar()

# <codecell>
# tok = emb[3][None]
# plt.plot((tok @ W_est).flatten())
thr = max(np.max(emb @ W), -np.min(emb @ W))

plt.imshow(emb @ W, cmap='BrBG', vmin=-thr, vmax=thr)
# plt.imshow(emb @ W, cmap='BrBG', vmin=-4, vmax=4)
# plt.imshow(emb @ W, cmap='BrBG', vmin=-10, vmax=10)
plt.colorbar()
plt.title('readout')

xs = np.linspace(0, 21)
plt.plot(xs, xs + 2, color='red', alpha=0.3)

plt.savefig('fig/lin_readout.png')

# <codecell>
att = emb @ t(A) @ t(emb)
thr = max(np.max(att), -np.min(att))

plt.imshow(att, cmap='BrBG', vmin=-thr, vmax=thr)
# plt.imshow(att, cmap='BrBG', vmin=-20, vmax=20)
# plt.imshow(att, cmap='BrBG', vmin=-30, vmax=30)
plt.colorbar()

xs = np.linspace(0, 23)
plt.plot(xs, xs, color='red', alpha=0.3)

plt.title('att')
plt.savefig('fig/lin_att.png')


# <codecell>
Wr = emb @ W

Wr[:,13]

res_next = -0.1 * np.sum(np.abs(Wr), axis=1) + 2 * Wr[:,13]

res_1 = -0.2 * np.sum(np.abs(Wr), axis=1) + 2 * Wr[:,1]
res_2 = -0.2 * np.sum(np.abs(Wr), axis=1) + 2 * Wr[:,2]

plt.plot(res_next)
plt.plot(res_1)
plt.plot(res_2)

res_total = 0.2 * (res_1 + res_2) + res_next
plt.plot(res_total)


# <codecell>

readout = emb @ W

plt.imshow(att.T / np.max(att), cmap='BrBG')
# plt.imshow(readout / np.max(readout), cmap='BrBG')
plt.colorbar()


# a = att.T.flatten()
# r = readout.flatten()
# plt.scatter(a, r, alpha=0.3)

# <codecell>
### CUSTOM GENERALIZING SOLUTION
n_vocab = emb.shape[0]
readout = np.zeros((n_vocab, n_vocab))

readout[3, 4:] = 2

readout[4:,1] = -4
readout[4:,2] = 2

readout[5, 4] = 1
readout[np.arange(6, n_vocab),np.arange(4, n_vocab - 2)] = 1
readout = emb.T @ readout
plt.imshow(emb @ readout, cmap='BrBG', vmin=-4, vmax=4)
plt.colorbar()

# <codecell>
att = np.zeros((n_vocab, n_vocab))
att[np.arange(4, n_vocab), np.arange(4, n_vocab)] = 1
att[np.arange(4, n_vocab - 1), np.arange(5, n_vocab)] = -2
att[4:,3] = 1

att = emb.T @ att.T @ emb

plt.imshow(emb @ t(att) @ emb.T, cmap='BrBG', vmin=-2, vmax=2)
plt.colorbar()

# <codecell>
params = {
    'Dense_0': {'kernel': att},
    'Dense_1': {'kernel': readout},
    'Embed_freeze': {'embedding': emb}
}

xs, ys = next(test_task)
logits = state.apply_fn({'params': params}, xs)
preds = logits.argmax(-1)

print('ys', ys[:3])
print('pr', preds[:3])

print('ys', ys[-3:])
print('pr', preds[-3:])

# <codecell>
state = state.replace(params=params)
preds = gen2(state, xs, beta=1)[:3]

print('ys', ys[:3])
print('pr', preds[:3])

# <codecell>
att = params['Dense_0']['kernel']
W = params['Dense_1']['kernel']

xs = jnp.array([[3, 7, 20, 18, 16, 14, 12, 10]])
X = emb[xs]

att = X @ t(att) @ t(X)
logits = np.tril(att) @ X @ W
plt.plot(logits[0, -1])
print(np.argmax(logits[0, -1]))

np.round(np.tril(att), decimals=2)

# <codecell>
plt.imshow(emb @ W, cmap='BrBG', vmin=-2, vmax=2)
out = emb @ W
out[3, 8]

# <codecell>
xs = jnp.array([[3, 7, 20, 0, 0, 0]])
gen2(state, xs, beta=100)


# <codecell>
emb = state.params['Embed_freeze']['embedding']
W = state.params['Dense_0']['kernel']
# K = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['key']['kernel'].squeeze()
Q = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['query']['kernel'].squeeze()
# V = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze()
# O = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze()

# W = V @ O @ W
# A = Q @ K.T
A = Q

xs, ys = next(train_task)
params = {
    'Embed_freeze': {'embedding': emb},
    'Dense_0': {'kernel': A.squeeze().T},
    'Dense_1': {'kernel': W}
}

config = TrConfig(n_vocab=n_vocab, n_hidden=n_hidden, return_format='final_logit')
m = config.to_model()
m.apply({'params': params}, xs).squeeze()


# X = emb[xs]
# X = transformer_phi(X, flatten=False)

# logits = ((X @ A.reshape(-1, 1)).squeeze() @ W).squeeze()
# logits[:,-1]

# <codecell>
state.apply_fn({'params': state.params}, xs)

# <codecell>
xs, ys = next(test_task)

# TODO: investigate with zero temperature
preds = gen2(state, xs)

# print('INPT', xs[:3])
# print('PRED', preds[:3])
# print('LABL', ys[:3])

print('INPT', xs[-3:])
print('PRED', preds[-3:])
print('LABL', ys[-3:])

# <codecell>
logits = state.apply_fn({'params': state.params}, xs)
logits[3].argmax(-1)

# <codecell>
vals = logits[3][18]
p = np.exp(vals) / np.sum(np.exp(vals))
plt.plot(p)

# <codecell>
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/7_logreg/length', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['pos_emb'],
        row['train_task'].samp_dist[1],
        row['train_task'].cot,
        row['info'],
    ], index=['name', 'pos_emb', 'n_hop', 'cot', 'info'])

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
mdf = mdf[
    (mdf['cot'] == False)
    & (mdf['pos_emb'] == True)
    ]
sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', estimator='max', marker='o', height=2, aspect=1.2, hue_order=['Full', 'Mix (dot)', 'Mix (phi)', 'Flat'])
plt.savefig('fig/tr_logreg_compare_no_cot_pe.png')

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_hop'] == 6]

sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='pos_emb', row='cot', kind='line', estimator='max', marker='o', height=2, aspect=1.2, hue_order=['Full', 'Mix (dot)', 'Mix (phi)', 'Flat'])
plt.savefig('fig/n_hop_6_sweep.png')


# <codecell>
### LENGTHWISE GENERALIZATION
df = collate_dfs('remote/7_logreg/complex', show_progress=True)
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['config']['pos_emb'],
        row['train_task'].samp_dist[1],
        row['train_task'].cot,
        row['train_args']['lr'],
        row['info'],
    ], index=['name', 'pos_emb', 'n_hop', 'cot', 'lr', 'info'])

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
adf = plot_df.drop(['orig_index', 'loss'], axis=1)
adf = adf.groupby(['pos_emb', 'cot', 'n_hop', 'test_n_hop', 'name'], as_index=False).max()

for pe, cot in itertools.product([True, False], [True, False]):
    mdf = adf.copy()
    mdf = mdf[
        (mdf['cot'] == cot)
        & (mdf['pos_emb'] == pe)
        ]
    sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='n_hop', col_wrap=4, kind='line', estimator='mean', marker='o', height=2, aspect=1.2)
    plt.savefig(f'fig/tr_compare_lin_pe_{pe}_cot_{cot}.png')
    plt.show()

# <codecell>
mdf = plot_df.copy()
mdf = mdf[mdf['n_hop'] == 6]

sns.relplot(mdf, x='test_n_hop', y='acc', hue='name', col='pos_emb', row='cot', kind='line', estimator='max', marker='o', height=2, aspect=1.2, hue_order=['Full', 'Mix (dot)', 'Mix (phi)', 'Flat'])
plt.savefig('fig/n_hop_6_sweep.png')

