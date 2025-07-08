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

depth = 10
n_hidden = 16
batch_size = 128

cot = True
ttr = True
nouveau = True
n_arms = 2
n_hop = 5
test_n_hop = 7

n_vocab = n_arms * depth + 1 + StarfishTask.offset

train_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(1,n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)
test_task = StarfishTask(n_arms=n_arms, depth=depth, samp_dist=(n_hop + 1, test_n_hop), batch_size=batch_size, cot=cot, trace_to_start=ttr, nouveau=nouveau)

# <codecell>
config = TransformerConfig(n_layers=1,
                           n_vocab=n_vocab,
                           n_out=n_vocab if cot else 1,
                           n_hidden=n_hidden,
                        #    pos_emb=not cot,
                           pos_emb=False,
                        #    max_len=100,
                           n_mlp_layers=2,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=True,
                           freeze_emb=True,
                           use_bias=False,
                        #    return_format=None if cot else True,
                           return_format=None if cot else 'final_logit',
                           mup_scale=True,
                           unif_att=True
                           )

# <codecell>
state, hist = train(config,
                    train_iter=iter(train_task), 
                    test_iter=iter(test_task), 
                    test_iters=1,
                    loss='ce_mask' if cot else 'bce',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot1] if cot else None,
                    print_fn=print_gen if cot else None,
                    lr=1e-2,
                    )

# <codecell>
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
att0 = intm['intermediates']['TransformerBlock_0']['SimpleSelfAttention_0']['attention_weights'][0].squeeze()
# att1 = intm['intermediates']['TransformerBlock_1']['SimpleSelfAttention_0']['attention_weights'][0].squeeze()
plt.imshow(att0)
# plt.imshow(att1)
print(att0)


# <codecell>
xs_emb = emb[xs]
# k = xs_emb @ K
# q = xs_emb @ Q

# att = q @ t(k) / n_hidden
att = jnp.ones((xs_emb.shape[1], xs_emb.shape[1]))
att = jnp.tril(att, k=0)
att = att.at[att == 0].set(-jnp.inf)

att = jax.nn.softmax(att, axis=-1)
# plt.imshow(att.squeeze())

xs_att = att @ xs_emb @ V @ O
xs_mlp_comb = jax.nn.gelu((xs_att + xs_emb) @ M1) @ M2

# xs_att_mlp = jax.nn.gelu(xs_att @ M1) @ M2
# xs_mlp = jax.nn.gelu(xs_emb @ M1) @ M2

xs_out = (xs_mlp_comb + xs_att + xs_emb) @ W
xs_red = (xs_mlp_comb) @ W


np.mean((xs_red - logits_orig)**2) / np.mean(logits_orig**2)
# xs_out / logits_orig

# <codecell>
xs_mlp_comb = jax.nn.gelu((xs_att) @ M1) @ M2 @ W
print(logits_orig[0][-1][2])
print(xs_mlp_comb[0][-1][2])
plt.plot(xs_mlp_comb[0][-1])
plt.plot(logits_orig[0][-1])

# <codecell>
z = emb @ M1

# plt.imshow(z @ z.T, cmap='bwr', vmin=-50, vmax=50)
plt.imshow(z, cmap='bwr')
plt.colorbar()

# vs = np.linspace(0, 100)
# plt.plot(vs, vs)

# <codecell>
z = emb @ V @ O @ M1

# plt.imshow(z @ z.T, cmap='bwr', vmin=-40_000, vmax=40_000)
plt.imshow(z, cmap='bwr')
plt.colorbar()

# <codecell>
a = M2 @ W
# plt.imshow(a, cmap='bwr', vmin=-1, vmax=1)
# plt.colorbar()

plt.plot(a[:,1], alpha=0.7)




# <codecell>

preds_mlp = jax.nn.relu(emb @ V @ O @ M1) @ M2 @ W
plt.imshow(preds_mlp)
plt.colorbar()

preds_mlp.argmax(-1)

preds_mlp[4]


# <codecell>
preds_w = emb @ W
plt.imshow(preds_w)
plt.colorbar()
preds_w.argmax(-1)



# <codecell>
A = state.params['Dense_0']['kernel']
w = state.params['Dense_1']['kernel']

emb = state.params['Embed_freeze']['embedding']
pos = state.params['PE_freeze']['embedding']

# xs, ys = next(test_task)
xs = jnp.array([[10, 19]])
xs_orig = np.copy(xs)
out = state.apply_fn({'params': state.params}, xs)

xs_s = emb[xs]
# ps = pos * np.sqrt(n_hidden)
ps = pos
xs = xs_s + ps

a_ss = xs_s @ t(xs_s @ A)
a_sp = xs_s @ t(ps @ A)
a_ps = ps @ t(xs_s @ A)
a_pp = ps @ t(ps @ A)

att = a_ss + a_sp + a_ps + a_pp
print('a_ss', a_ss[0,-1])
print('a_sp', a_sp[0,-1])
print('a_ps', a_ps[0,-1])
print('a_pp', a_pp[-1])
# att_t = jnp.tril((xs_s + ps) @ t((xs_s + ps) @ A))
# print(att[:3])
# print(att_t[:3])

w1 = (xs[:,0] @ w).flatten()
w2 = (xs[:,1] @ w).flatten()
# ps = ps[None]
# w1 = (ps[:,0] @ w).flatten()
# w2 = (ps[:,1] @ w).flatten()

a1 = att[:,1,0]
a2 = att[:,1,1]

pred = a1 * w1 + a2 * w2

print('a1', a1)
print('w1', w1)
print('a2', a2)
print('w2', w2)

print(pred)
print(out)

# plt.hist(-out[out < 0], bins=10)


# <codecell>
df = collate_dfs('remote/12_scale/arm', show_progress=True)
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
mdf = mdf[(mdf['test_n_hop'] == 5)]

mdf = mdf[['n_arms', 'n_hidden', 'acc']]
mdf = mdf.groupby(['n_arms', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='n_arms', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
# g.plot(xs, 14 - 0.5 * xs, color='black', linestyle='dashed')
# g.plot(xs, 40 - 1.5 * xs, color='cyan', linestyle='dashed')
g.plot(xs, 35 - 1 * xs, color='cyan', linestyle='dashed')

# g.plot(xs, 39 - xs, color='gray', linestyle='dashed')

# xs = 2**np.linspace(-5, 8)
# g.plot(xs, 1 - 2 * xs + 13, color='black', linestyle='dashed')

g.set_ylabel('n_arms')
g.set_xlabel('n_hidden')

plt.title('Zero')
plt.savefig(f'fig/zero_mlp_arms_v_size.png', bbox_inches='tight')
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
    return pd.Series([
        row['name'],
        row['train_task'].depth,
        row['train_task'].samp_dist[1],
        row['config']['n_hidden'],
        row['info'],
    ], index=['name', 'depth', 'n_hop', 'n_hidden', 'info'])

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
np.unique(np.round(plot_df['n_hop'] / plot_df['depth'], decimals=1))

# <codecell>
plot_df['n_hop_prop'] = np.round(plot_df['n_hop'] / plot_df['depth'], decimals=1)
plot_df['n_hop_prop']
# <codecell>
# for name in np.unique(plot_df['name']):
mdf = plot_df.copy()
mdf = mdf[
    (mdf['test_n_hop'] == 0.75)
    & ((mdf['n_hop_prop'] == 0.7) | (mdf['n_hop_prop'] == 0.7))
    ]

mdf = mdf[['depth', 'n_hidden', 'acc']]
mdf = mdf.groupby(['depth', 'n_hidden'], as_index=False).mean()
mdf = mdf.pivot(index='depth', columns='n_hidden', values='acc')

mdf = mdf.iloc[::-1]

g = sns.heatmap(mdf, square=False, vmin=0.5, vmax=1)

xs = 2**np.linspace(-5, 8)
g.plot(xs, 35 - 0.7 * xs, color='cyan', linestyle='dashed')
# g.plot(xs, 40 - 1.5 * xs, color='cyan', linestyle='dashed')
g.plot(xs, 35 - 1 * xs, color='cyan', linestyle='dashed')

g.plot(xs, 70 - 2 * xs, color='cyan', linestyle='dashed')

g.set_ylabel('depth')
g.set_xlabel('n_hidden')

plt.title('Zero')
# plt.savefig(f'fig/zero_mlp_arms_v_size.png', bbox_inches='tight')
plt.show()