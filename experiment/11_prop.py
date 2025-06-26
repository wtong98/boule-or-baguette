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

