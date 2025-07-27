"""Exploring generalization on large graphs"""

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
from task.graph import *

depth = 100
n_hidden = 128
batch_size = 128

cot = True
ttr = True
nouveau = True
force_bin_label = True
n_arms = 10
n_hop = 5
test_n_hop = 7

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau, force_bin_label=force_bin_label)

xs, ys = next(train_task)
# print(xs[:3])
# print(ys[:3])

# # <codecell>
# train_task.batch_size = 4096
# xs, ys = next(train_task)
# emb = np.random.randn(n_vocab, n_hidden) / np.sqrt(n_hidden)

# xs_emb = emb[xs]
# ys = 2 * ys - 1

# xs_tot = ys[:,None] * (xs_emb[:,0] + xs_emb[:,1])

# xs_mat = xs_tot.T @ xs_tot / train_task.batch_size

# # <codecell>
# evals, evecs = np.linalg.eig(xs_mat)

# plt.plot(emb @ evecs[:,[1]])
# # plt.plot(evals)



# <codecell>
config = TransformerConfig(n_layers=1,
                           n_vocab=n_vocab,
                        #    n_out=n_vocab if cot else 1,
                           n_out=1,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                        #    residual_connections=True if cot else False,
                           residual_connections=False,
                           freeze_emb=True,
                           use_bias=False,
                           return_format='final_logit_up_to_pad' if cot else 'final_logit',
                           mup_scale=True,
                           unif_att=True
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    # loss='ce_mask' if cot else 'bce',
                    loss='bce',
                    test_every=1000,
                    train_iters=25_000,
                    # eval_fns=[loss_and_acc, gen_acc_cot1] if cot else None,
                    eval_fns=None,
                    # print_fn=print_gen if cot else None,
                    print_fn=None,
                    lr=1e-3,
                    )

# <codecell>
xs, ys = next(train_task)

preds = state.apply_fn({'params': state.params}, xs)
np.mean((preds > 0) == ys)

# <codecell>
### ANALYSIS OF COT SIMP MODEL
jax.tree.map(np.shape, state.params)

emb = state.params['Embed_freeze']['embedding']
readout = state.params['Dense_0']['kernel'] / n_hidden
V = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze() / np.sqrt(n_hidden)
O = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze() / np.sqrt(n_hidden)
W1 = state.params['TransformerBlock_0']['Dense_0']['kernel'] / np.sqrt(n_hidden)
W2 = state.params['TransformerBlock_0']['Dense_1']['kernel'] / np.sqrt(n_hidden)

xs = next(train_task)[0]

# <codecell>
emb = emb.at[0].set(0)
xs_emb = emb[xs]
W = V @ O @ W1
a = W2 @ readout

xs_lens = (xs != 0).sum(axis=-1)
xs_att = xs_emb.sum(axis=1) / xs_lens[:,None]

pred = jax.nn.relu(xs_att @ W) @ a
logits, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

pred = jax.nn.relu(xs_att @ W) @ a
pred = pred.flatten()

np.mean((pred - logits)**2) / np.mean(logits**2)


# <codecell>
sort_idxs = np.argsort(a.flatten())
proj = emb @ W
proj = proj[:,sort_idxs]

plt.gcf().set_size_inches(20, 12)
bound = max(np.max(proj), np.min(proj)) * 1
im = plt.imshow(proj, cmap='BrBG', vmin=-bound, vmax=bound)
plt.colorbar(im, shrink=0.2)
plt.tight_layout()

# plt.savefig('fig/zero_mlp_coeffs.svg')

# <codecell>
plt.plot(a.flatten()[sort_idxs])

# <codecell>
sort_idxs = np.argsort(a.flatten())
idx = sort_idxs[4]
plt.plot(proj[:,10])
plt.plot(proj[:,15])

plt.axhline(y=0, color='gray', linestyle='dashed', alpha=0.7)

# plt.xlim((200, 300))

# <codecell>
plt.gcf().set_size_inches(9, 3)
plt.plot(proj[33])
plt.plot(proj[131])
plt.plot(a[sort_idxs])


# <codecell>
plt.plot(proj[31,100:])

# <codecell>
np.mean(proj[:,-5:] > 0)

# <codecell>
plt.hist(proj[:,0], bins=25)


# <codecell>
### ANALYSIS OF ZERO MODEL
# xs = jnp.array([[305, 600, 500, 400, 300, 200, 100, 4]])
xs = jnp.array([[65, 75]])

logits = state.apply_fn({'params': state.params}, xs)
logits

# <codecell>
jax.tree.map(np.shape, state.params)

emb = state.params['Embed_freeze']['embedding']
a = state.params['Dense_0']['kernel'] / n_hidden
V = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze() / np.sqrt(n_hidden)
O = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze() / np.sqrt(n_hidden)
W1 = state.params['TransformerBlock_0']['Dense_0']['kernel'] / np.sqrt(n_hidden)
W2 = state.params['TransformerBlock_0']['Dense_1']['kernel'] / np.sqrt(n_hidden)

xs = next(test_task)[0]
xs_emb = emb[xs]
W = V @ O @ W1
a = W2 @ a

xs_emb = 0.5 * (xs_emb[:,0] + xs_emb[:,1])

pred = jax.nn.relu(xs_emb @ W) @ a
logits, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')

np.mean((pred.flatten() - logits)**2) / np.mean(logits**2)

# <codecell>
sort_idxs = np.argsort(a.flatten())
proj = emb @ W
proj = proj[:,sort_idxs]

plt.gcf().set_size_inches(20, 12)
bound = max(np.max(proj), np.min(proj)) * 0.5
im = plt.imshow(proj, cmap='BrBG', vmin=-bound, vmax=bound)
plt.colorbar(im, shrink=0.2)
plt.tight_layout()

# plt.savefig('fig/zero_mlp_coeffs.svg')

# <codecell>
plt.plot(a.flatten()[sort_idxs])

# <codecell>
sort_idxs = np.argsort(a.flatten())
idx = sort_idxs[4]
plt.plot(proj[:,0])
plt.plot(proj[:,-1])
plt.axhline(y=0, color='gray', linestyle='dashed', alpha=0.7)
a[sort_idxs[0]]

# plt.xlim((200, 300))

# <codecell>
sort_idxs = np.argsort(a.flatten())
proj = emb @ W
proj = proj[:,sort_idxs]

plt.gcf().set_size_inches(9, 3)
plt.plot(proj[29])
plt.plot(proj[39])
plt.plot(a[sort_idxs])

plt.axhline(y=0, color='gray', linestyle='dashed', alpha=0.7)

# <codecell>
log = []
for idx in range(proj.shape[1]):
    samp = proj[:,idx]
    curr = []
    for i in range(n_arms):
        count = np.mean(samp[i::n_arms] > 0)
        curr.append(count)
    log.append(curr)

log = np.array(log)

# <codecell>
plt.imshow(log, cmap='BrBG', vmin=0.2, vmax=0.8)
plt.colorbar()

# <codecell>
plt.plot(log[0])
plt.plot(log[-1])
plt.plot(log[-2])


# <codecell>
xs = np.array([[27, 137]])
state.apply_fn({'params': state.params}, xs)

# <codecell>
np.mean(proj[:,-5:] > 0)

# <codecell>
plt.hist(proj[:,-4], bins=50)


# <codecell>
### ANALYSIS OF COT MODEL
# xs = jnp.array([[305, 600, 500, 400, 300, 200, 100, 4]])
xs = jnp.array([[25, 75]])

logits = state.apply_fn({'params': state.params}, xs)
preds = logits.argmax(-1)
print(preds)

plt.plot(logits[0,-1])


# <codecell>
batch = next(train_task)
xs, _ = batch
traj = gen1(state, xs)
preds = extract_pred(traj)
print(traj[-10:])
print(preds)


# <codecell>
jax.tree.map(np.shape, state.params)

# <codecell>
W = state.params['Dense_0']['kernel'] / n_hidden
emb = state.params['Embed_freeze']['embedding']

M1 = state.params['TransformerBlock_0']['Dense_0']['kernel'] / np.sqrt(n_hidden)
M2 = state.params['TransformerBlock_0']['Dense_1']['kernel'] / np.sqrt(n_hidden)

# K = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['key']['kernel'].squeeze() / np.sqrt(n_hidden)
# Q = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['query']['kernel'].squeeze() / np.sqrt(n_hidden)
V = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['value']['kernel'].squeeze() / np.sqrt(n_hidden)
O = state.params['TransformerBlock_0']['SimpleSelfAttention_0']['out']['kernel'].squeeze() / np.sqrt(n_hidden)

# <codecell>
xs, _ = next(test_task)
# print(xs[:3])

xs = jnp.array([[20, 80, 70, 60, 50, 40, 30, 20, 10, 4, 2]])
xs = jnp.array([xs[0,:-np.sum(xs[0] == 0) - 1]])
print('XS', xs)
# m = config.replace(remove_att=True).to_model()
m = config.to_model()

# logits_orig, intm = state.apply_fn({'params': state.params}, xs, mutable='intermediates')
logits_orig, intm = m.apply({'params': state.params}, xs, mutable='intermediates')
print('PS', logits_orig.argmax(-1))
# intm['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['attention_weights']
# intm['intermediates']


# <codecell>
xs_emb = emb[xs]

att = jnp.ones((xs_emb.shape[1], xs_emb.shape[1]))
att = jnp.tril(att, k=0)
att = att.at[att == 0].set(-jnp.inf)

att = jax.nn.softmax(att, axis=-1)

xs_att = att @ xs_emb @ V @ O
xs_mlp_comb = jax.nn.relu((xs_att + xs_emb) @ M1) @ M2

xs_out = (xs_mlp_comb + xs_att + xs_emb) @ W
xs_red = (xs_mlp_comb) @ W

W_eff = V @ O @ M1
a_eff = M2 @ W

xs_eff = jax.nn.relu(att @ xs_emb @ W_eff + xs_emb @ M1) @ a_eff
np.mean((xs_eff - logits_orig)**2) / np.mean(logits_orig**2)

# <codecell>
sort_idxs = np.argsort(a_eff[:,2])
plt.plot(a_eff[sort_idxs,2])

# <codecell>
proj = emb @ W_eff
proj = proj[:,sort_idxs]
bound = max(np.max(proj), np.min(proj))
plt.imshow(proj, vmin=-bound, vmax=bound, cmap='BrBG')
plt.colorbar(shrink=0.2)

# <codecell>
plt.plot(proj[:,-1])
plt.plot(proj[:,0])

# <codecell>
plt.hist(proj[4,:], bins=50)

# <codecell>




# <codecell>

# # <codecell>
# xs_mlp_comb = jax.nn.gelu((xs_att) @ M1) @ M2 @ W
# print(logits_orig[0][-1][2])
# print(xs_mlp_comb[0][-1][2])
# plt.plot(xs_mlp_comb[0][-1])
# plt.plot(logits_orig[0][-1])

# # <codecell>
# z = emb @ M1

# # plt.imshow(z @ z.T, cmap='bwr', vmin=-50, vmax=50)
# plt.imshow(z, cmap='bwr')
# plt.colorbar()

# # vs = np.linspace(0, 100)
# # plt.plot(vs, vs)

# # <codecell>
# z = emb @ V @ O @ M1

# # plt.imshow(z @ z.T, cmap='bwr', vmin=-40_000, vmax=40_000)
# plt.imshow(z, cmap='BrBG', vmin=-100, vmax=100)
# plt.colorbar()

# # <codecell>
# a = M2 @ W
# # plt.imshow(a, cmap='bwr', vmin=-1, vmax=1)
# # plt.colorbar()

# plt.plot(a[:,1], alpha=0.7)


# # <codecell>

# preds_mlp = jax.nn.relu(emb @ V @ O @ M1) @ M2 @ W
# plt.imshow(preds_mlp)
# plt.colorbar()

# preds_mlp.argmax(-1)

# preds_mlp[4]


# # <codecell>
# preds_w = emb @ W
# plt.imshow(preds_w)
# plt.colorbar()
# preds_w.argmax(-1)




# <codecell>
df = collate_dfs('remote/12_scale/zero_arm', show_progress=True)
# df = collate_dfs('remote/12_scale/ar_arm', show_progress=True)
# df = collate_dfs('remote/12_scale/arm', show_progress=True)
df

# %%
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['info'],
    ], index=['name', 'n_arms', 'n_hop', 'n_hidden', 'info'])

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
plot_df

# <codecell>
# for name in np.unique(plot_df['name']):
mdf = plot_df.copy()
mdf = mdf[(mdf['test_n_hop'] == 7)]

mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).max()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 51 - 0.5 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 40 - 1.5 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 35 - 1 * xs, color='cyan', linestyle='dashed')
g.plot(xs, 45 - 2 * xs, color='cyan', linestyle='dashed')

# g.plot(xs, 39 - xs, color='gray', linestyle='dashed')

# xs = 2**np.linspace(-5, 8)
# g.plot(xs, 1 - 2 * xs + 13, color='black', linestyle='dashed')

g.set_ylabel('n_arms')
g.set_xlabel('n_hidden')

plt.title('Zero')
# plt.savefig(f'fig/zero_mlp_arms_v_size.png', bbox_inches='tight')
plt.show()


# <codecell>
df = collate_dfs('remote/12_scale/zero_length', show_progress=True)
df

# <codecell>
rand_idxs = np.random.choice(len(df), size=100, replace=False)
for ex in df['hist'].iloc[rand_idxs]:
    vals = [p['loss'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# df['hist'].iloc[0]['train']
    

# %%
def extract_plot_vals(row):
    n_hop_prop = row['info']['n_hop_prop']
    del row['info']['n_hop_prop']
    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].depth,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['config']['residual_connections'],
        n_hop_prop,
        row['info'],
    ], index=['name', 'n_arms', 'depth', 'n_hop', 'n_hidden', 'residual_connections', 'n_hop_prop', 'info'])

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
plot_df

# <codecell>
# for name in np.unique(plot_df['name']):
mdf = plot_df.copy()
mdf = mdf[
    (mdf['test_n_hop'] == 0.5)
    & (mdf['n_hop_prop'] == 0.5)
    & (mdf['n_arms'] == 10)
    & (mdf['residual_connections'] == False)
    ]

mdf = mdf[['depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['depth', 'n_hidden'], as_index=False).max()
mdf = mdf.pivot(index='depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
# g.plot(xs, 35 - 0.7 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 50 - 1.5 * xs, color='cyan', linestyle='dashed')
# g.plot(xs + np.log(xs), 40 - 1 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 20 - 0.67 * xs, color='cyan', linestyle='dashed')
g.plot(xs, 20 - 1 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 30 - 0.5 * xs, color='cyan', linestyle='dashed')

# g.plot(xs, 50 - 1.5 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('depth')
g.set_xlabel('n_hidden')

plt.title('Zero')
# plt.savefig(f'fig/zero_mlp_depth_v_size.png', bbox_inches='tight')
plt.show()

# <codecell>
df = collate_dfs('remote/12_scale/ar_length', show_progress=True)
df

# <codecell>
rand_idxs = np.random.choice(len(df), size=100, replace=False)
for ex in df['hist'].iloc[rand_idxs]:
    vals = [p['loss'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# df['hist'].iloc[0]['train']
    

# %%
def extract_plot_vals(row):
    if 'n_hop_prop' in row['info']:
        n_hop_prop = row['info']['n_hop_prop']
        del row['info']['n_hop_prop']
    else:
        n_hop_prop = row['train_task'].samp_dist[1] / row['train_task'].depth
        props = np.array([0.25, 0.5, 0.7])
        dists = np.abs(n_hop_prop - props)
        n_hop_prop = props[np.argmin(dists)]

    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].depth,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        n_hop_prop,
        row['info'],
    ], index=['name', 'n_arms', 'depth', 'n_hop', 'n_hidden', 'n_hop_prop', 'info'])

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
plot_df

# <codecell>
# for name in np.unique(plot_df['name']):
mdf = plot_df.copy()
mdf = mdf[
    (mdf['test_n_hop'] == 0.5)
    & (mdf['n_hop_prop'] == 0.5)
    & (mdf['n_arms'] == 5)
    ]

mdf = mdf[['depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
# g.plot(xs, 35 - 0.7 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 50 - 1.5 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 20 - 1 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 45 - 0.67 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 45 - 0.5 * xs, color='cyan', linestyle='dashed')

g.plot(xs, 20 - 0.5 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('depth')
g.set_xlabel('n_hidden')

plt.title('AR full')
# plt.savefig(f'fig/zero_mlp_depth_v_size_debug.png', bbox_inches='tight')
plt.show()

# <codecell>
df = collate_dfs('remote/12_scale/ar_bd_sweep', show_progress=True)
df

# <codecell>
rand_idxs = np.random.choice(len(df), size=100, replace=False)
for ex in df['hist'].iloc[rand_idxs]:
    vals = [p['loss'] for p in ex['test']]
    plt.plot(vals, color='C0', alpha=0.1)

# df['hist'].iloc[0]['train']
    

# %%
def extract_plot_vals(row):
    if 'n_hop_prop' in row['info']:
        n_hop_prop = row['info']['n_hop_prop']
        del row['info']['n_hop_prop']
    else:
        n_hop_prop = row['train_task'].samp_dist[1] / row['train_task'].depth
        props = np.array([0.25, 0.5, 0.7])
        dists = np.abs(n_hop_prop - props)
        n_hop_prop = props[np.argmin(dists)]

    return pd.Series([
        row['name'],
        row['train_task'].n_arms,
        row['train_task'].depth,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        n_hop_prop,
        row['info'],
    ], index=['name', 'n_arms', 'depth', 'n_hop', 'n_hidden', 'n_hop_prop', 'info'])

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
plot_df

# <codecell>
# for name in np.unique(plot_df['name']):
mdf = plot_df.copy()
mdf = mdf[
    (mdf['test_n_hop'] == 0.5)
    & (mdf['n_hop_prop'] == 0.5)
    ]

mdf = mdf[['depth', 'n_arms', 'acc']]
mdf = mdf.groupby(['depth', 'n_arms'], as_index=False).mean()
mdf = mdf.pivot(index='depth', columns='n_arms', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
# g.plot(xs, 35 - 0.7 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 50 - 1.5 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 20 - 1 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 45 - 0.67 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 45 - 0.5 * xs, color='cyan', linestyle='dashed')

g.plot(xs, 25 + 0.5 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('depth')
g.set_xlabel('n_arms')

plt.title('AR full')
# plt.savefig(f'fig/zero_mlp_depth_v_size_debug.png', bbox_inches='tight')
plt.show()