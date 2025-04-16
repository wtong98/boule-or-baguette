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
    n_hidden: float = 32
    flatten: bool = False

    def to_model(self):
        return TrLogReg(self)


class TrLogReg(nn.Module):
    config: TrLogRegConfig

    @nn.compact
    def __call__(self, inputs):
        x = nn.Embed(self.config.n_vocab, features=self.config.n_hidden, name='Embed_freeze')(inputs)
        x = transformer_phi(x, flatten=self.config.flatten)
        
        if not self.config.flatten:
            x = nn.Dense(1, use_bias=False)(x).squeeze()

        x = nn.Dense(self.config.n_vocab, use_bias=False)(x)
        return x


depth = 15
n_vocab = 2 * depth + 1 + StarfishTask.offset
n_hidden = 64
batch_size = 64

n_layers = 1

cot = True
ttr = False

train_task = StarfishTask(depth=depth, samp_dist=(1,5), batch_size=batch_size, cot=cot, trace_to_start=ttr)
test_task = StarfishTask(depth=depth, samp_dist=6, batch_size=batch_size, cot=cot, trace_to_start=ttr)


config = TransformerConfig(n_layers=n_layers,
                           n_vocab=n_vocab,
                           n_out=n_vocab,
                           n_hidden=n_hidden,
                           pos_emb=False,
                           n_mlp_layers=0,
                           n_heads=1,
                           layer_norm=False,
                           as_rf_model=False,
                           residual_connections=False,
                           freeze_emb=True,
                           use_bias=False,
                           return_final_logits_only=False,
                           mup_scale=True,
                           linear_att=True
                           )

# config = TrLogRegConfig(n_vocab=n_vocab, n_hidden=n_hidden, flatten=False)

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
                    # loss='bce',
                    loss='ce_mask',
                    test_every=1000,
                    train_iters=10_000,
                    use_tqdm=False,
                    eval_fns=[loss_and_acc, gen_acc_cot],
                    print_fn=print_gen,
                    lr=1e-3,
                    # optim=optax.sgd
                    )


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

