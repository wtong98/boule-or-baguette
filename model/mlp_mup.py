"""
Simple MLP model
"""

# <codecell>

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

def parse_act_fn(fn: str):
    if fn == 'relu':
        return jax.nn.relu
    elif fn == 'linear':
        return lambda x: x
    elif fn == 'gelu':
        return jax.nn.gelu
    elif fn =='quadratic':
        return lambda x: x**2
    else:
        raise ValueError(f'function not recognized: {fn}')


@struct.dataclass
class MlpConfig:
    """Global hyperparamters"""
    n_vocab: int | None = None
    n_layers: int = 2
    n_hidden: int = 128
    n_out: int = 1
    act_fn: str = 'relu'
    layer_norm: bool = False
    mup_scale: bool = False
    as_rf_model: bool = False
    use_bias: bool = True
    freeze_emb: bool = False

    def to_model(self):
        return MLP(self)


class MLP(nn.Module):

    config: MlpConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)

        if self.config.n_vocab is not None:
            name = 'Embed_freeze' if (self.config.freeze_emb or self.config.as_rf_model) else None
            
            x = nn.Embed(
                num_embeddings=self.config.n_vocab,
                features=self.config.n_hidden,
                name=name)(x)
        
        x = x.reshape(x.shape[0], -1)

        for i in range(self.config.n_layers):
            name = None
            if self.config.as_rf_model:
                name = f'Dense_{i}_freeze'
            
            # if i == 0:
            # mup_init = jax.nn.initializers.normal(1 / np.sqrt(self.config.n_hidden))
            mup_init = jax.nn.initializers.truncated_normal(1)
            x = nn.Dense(self.config.n_hidden,
                            use_bias=self.config.use_bias,
                            kernel_init=mup_init,
                            name=name)(x)
            x = x / np.sqrt(self.config.n_hidden)

            # else:
            #     mup_init = jax.nn.initializers.normal(1 / np.sqrt(self.config.n_hidden))
            #     x = nn.Dense(self.config.n_hidden, 
            #                 use_bias=self.config.use_bias,
            #                 name=name,
            #                 kernel_init=mup_init)(x)
                # x = x / self.config.n_hidden

            self.sow('intermediates', f'layer_{i}', x)
            # x = act_fn(x)

        # mup_init = jax.nn.initializers.normal(1)
        # mup_init = jax.nn.initializers.normal(1 / np.sqrt(self.config.n_hidden))
        mup_init = jax.nn.initializers.truncated_normal(1)
        out = nn.Dense(self.config.n_out, use_bias=self.config.use_bias, kernel_init=mup_init)(x)
        # out = out / np.sqrt(self.config.n_hidden)
        out = out / self.config.n_hidden

        if self.config.n_out == 1:
            out = out.flatten()

        return out


## COORDINATE CHECKING
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from task.graph import *
from common import *
from train import *


base_lr = 1e-3
depth = 10
n_vocab = 2 * depth + 4

train_task = StarfishTask(depth=depth, n_arms=2)

xs_init, _ = next(train_task)

state = None

all_norms = []
for n_steps in tqdm([1]):
    curr_norms = []
    curr_norms_att = []

    for n_hidden in [64, 128, 256, 512, 1024]:
        # lr = base_lr / np.sqrt(n_hidden)
        lr = base_lr


        config = MlpConfig(n_vocab=n_vocab,
                           n_layers=3,
                           n_hidden=n_hidden,
                           use_bias=False,
                           freeze_emb=True,
                           mup_scale=True)


        state = create_train_state(jax.random.key(new_seed()),
                                model=config.to_model(),
                                dummy_input=xs_init,
                                optim=optax.adamw,
                                lr=lr)

        logits_init, intm = config.to_model().apply({'params': state.params}, xs_init, mutable='intermediates')
        w_init = intm['intermediates']['layer_0'][0]
        w_init2 = intm['intermediates']['layer_1'][0]
        

        state, hist = train(state,
                            train_iter=iter(train_task), 
                            loss='bce',
                            test_every=1000,
                            train_iters=n_steps, 
                            test_iters=1,
                            lr=lr,
                            seed=None)

        logits, intm = config.to_model().apply({'params': state.params}, xs_init, mutable='intermediates')
        w = intm['intermediates']['layer_0'][0]
        w2 = intm['intermediates']['layer_1'][0]

        norm = np.std(logits - logits_init).item()
        norm_w = np.std(w_init - w).item()
        norm_w2 = np.std(w_init2 - w2).item()
        curr_norms.append((norm, norm_w, norm_w2))
    
    all_norms.append(curr_norms)

# <codecell>
for norms in all_norms:
    norms = np.array(norms)
    plt.plot(norms[:,0], 'o--')
    plt.plot(norms[:,1], 'o--')
    plt.plot(norms[:,2], 'o--')
    # plt.yscale('log')

# %%
_, intm = state.apply_fn({'params': state.params}, xs_init, mutable='intermediates')
intm['intermediates']['layer_0'][0]
