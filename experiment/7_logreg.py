"""Transformer operation as logistic regression"""


# <codecell>
from pathlib import Path
import pickle

from flax import linen as nn
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
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from task.graph import *

def sinusoidal_init(max_len=2048,
                    squeeze=False):

    def init(key, shape, dtype=np.float32):
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        pe[:, :d_feature // 2] = np.sin(position * np.pi / 2)
        pe[:, d_feature // 2:] = np.cos(position * np.pi / 2)

        if not squeeze:
            pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]

        return jnp.array(pe)

    return init


def transformer_phi(X, flatten=True):
    X_curr = X.reshape(*X.shape, 1, 1)

    X = jnp.repeat(jnp.expand_dims(X, axis=1), X.shape[1], axis=1)
    X = jnp.permute_dims(X, (0, 3, 1, 2))  # B x H x L x L
    X = jnp.tril(X)
    X = jnp.permute_dims(X, (0, 2, 3, 1))  # B x L x L x H
    X = t(X) @ X

    X = jnp.expand_dims(X, axis=2)
    X = X_curr * X                            # B x L x j x k x m

    X = jnp.permute_dims(X, (0, 1, 4, 2, 3))  # B x L x m x j x k

    if flatten:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    return X


@struct.dataclass
class TrLogRegConfig:
    n_vocab: float
    n_out: float = 1
    n_hidden: float = 32
    flatten: bool = False
    return_final_logits_only: bool = False
    pos_emb: bool = False
    max_len: int = 2

    def to_model(self):
        return TrLogReg(self)


class TrLogReg(nn.Module):
    config: TrLogRegConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs)

        if self.config.pos_emb:
            pos_emb_shape = (1, config.max_len, x.shape[-1])
            pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                                    pos_emb_shape,
                                                                    None)
        
            pe = pos_embedding[:, :x.shape[1], :]

            # ps = jnp.arange(x.shape[1])[None]
            # ps = nn.Embed(self.config.max_length, features=self.config.n_hidden, name='PE_freeze')(ps)

            x = x + pe

        x = transformer_phi(x, flatten=self.config.flatten)
        
        if not self.config.flatten:
            x = nn.Dense(1, use_bias=False)(x).squeeze()

        x = nn.Dense(self.config.n_out, use_bias=False)(x)

        if self.config.return_final_logits_only:
            x = x[:,-1]

            if config.n_out == 1:
                x = x.flatten()

        return x


@struct.dataclass
class TrConfig:
    n_vocab: int
    n_out: int = 1
    n_hidden: int = 32
    return_final_logits_only: bool = False
    pos_emb: bool = False
    max_len: int = 2

    def to_model(self):
        return Tr(self)


class Tr(nn.Module):
    config: TrConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs) # B x L x H

        if self.config.pos_emb:
            pos_emb_shape = (1, config.max_len, x.shape[-1])
            pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
                                                                    pos_emb_shape,
                                                                    None)
        
            pe = pos_embedding[:, :x.shape[1], :]

            # ps = jnp.arange(x.shape[1])[None]
            # ps = nn.Embed(self.config.max_length, features=self.config.n_hidden, name='PE_freeze')(ps)

            x = x + pe

        Ax = nn.Dense(self.config.n_hidden, use_bias=False)(x)
        att = jnp.tril(x @ t(Ax))  # B x L x L
        x = att @ x

        x = nn.Dense(self.config.n_out, use_bias=False)(x)
        if self.config.return_final_logits_only:
            x = x[:,-1]

            if config.n_out == 1:
                x = x.flatten()

        return x


# config = TrConfig(10, max_len=2)
# x = np.ones((10, 2, 64))

# pos_emb_shape = (1, config.max_len, x.shape[-1])
# pos_embedding = sinusoidal_init(max_len=config.max_len)(None,
#                                                         pos_emb_shape,
#                                                         None)

# plt.plot(pos_embedding[0].T, '--o')

# <codecell>
depth = 10
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 512
batch_size = 64

n_layers = 1

cot = True
ttr = False

train_task = StarfishTask(depth=depth, samp_dist=(1,5), batch_size=batch_size, cot=cot, trace_to_start=ttr)
test_task = StarfishTask(depth=depth, samp_dist=6, batch_size=batch_size, cot=cot, trace_to_start=ttr)


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
#                            return_final_logits_only=False if cot else True,
#                            mup_scale=True,
#                            linear_att=True
#                            )

# config = TrLogRegConfig(n_vocab=n_vocab, 
#                         pos_emb=True,
#                         n_out=n_vocab if cot else 1,
#                         n_hidden=n_hidden, 
#                         flatten=False,
#                         return_final_logits_only=False if cot else True)

config = TrConfig(n_vocab=n_vocab, 
                  pos_emb=False,
                  n_out=n_vocab if cot else 1,
                  n_hidden=n_hidden, 
                  return_final_logits_only=False if cot else True)

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
jax.tree.map(np.shape, state.params)

# <codecell>
emb = state.params['Embed_freeze']['embedding']
A = state.params['Dense_0']['kernel']
W = state.params['Dense_1']['kernel']

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
logits

# <codecell>
true_logits = state.apply_fn({'params': state.params}, xs)
true_logits

# <codecell>
np.mean(np.abs(logits - true_logits))

# <codecell>
# EXPLICATION OF FAILURE MODE
xs = jnp.array([[6, 20, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
gen2(state, xs)

# <codecell>
xs = jnp.array([[6, 20, 3, 20, 18]])
X = emb[xs]

att = X @ t(A_est) @ t(X)
logits = np.tril(att) @ X @ W_est
plt.plot(logits[0, -1])

np.round(np.tril(att), decimals=2)

# <codecell>
tok = emb[14][None]
plt.plot((tok @ W_est).flatten())

# <codecell>
A_orig = hist['params'][0]['Dense_0']['kernel']

toks = emb[np.array([6, 24])]
toks @ t(A) @ t(toks)

# <codecell>
state.apply_fn({'params': state.params}, xs).squeeze()

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

config = TrConfig(n_vocab=n_vocab, n_hidden=n_hidden, return_final_logits_only=True)
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

