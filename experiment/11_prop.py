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
batch_size = 4

cot = True
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
#                   return_final_logits_only=False if cot else True)

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
                    test_every=10,
                    test_iters=0,
                    train_iters=0,
                    use_tqdm=True,
                    eval_fns=[loss_and_acc, gen_acc_cot] if cot else None,
                    print_fn=print_gen if cot else None,
                    )

# <codecell>
yes_id = 13138
no_id = 32165
state_id = 5219

def score(state, xs):
    start_idx = jnp.argmax((xs[2:] == state_id)) + 4
    is_true = jnp.argmax(xs == yes_id) > 0

    preds = generate(state, xs, idx=start_idx)
    pred_is_true = jnp.argmax(preds == yes_id) > 0
    pred_is_false = jnp.argmax(preds == no_id) > 0

    if pred_is_true == pred_is_false == False:
        return False

    return is_true == pred_is_true

def generate(state, xs, idx, beta=1, seed=None):
    if seed is None:
        seed = new_seed()

    xs = xs[None]
    source = jax.random.key(seed)
    while idx < xs.shape[1] - 1:
        key, source = jax.random.split(source)
        xs = _gen_pass(key, state, xs, idx, beta)
        idx += 1
    
    return xs.squeeze()

@jax.jit
def _gen_pass(key, state, xs, idx, beta):
    logits = state.apply_fn({'params': state.params}, xs)
    preds = jax.random.categorical(key, beta * logits)
    # xs = xs.at[:,idx+1].set(preds[:,idx])
    xs = xs.at[:,idx+1].set(preds[idx])
    return xs


xs = jnp.array(xs)
score(state, xs[2])

# TODO: instrument correctly and submit <-- STOPPED HERE
