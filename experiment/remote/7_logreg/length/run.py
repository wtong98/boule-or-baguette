"""
Relationship between size and accuracy on trees of various size
"""

# <codecell>
import flax.linen as nn
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
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

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 50_000
depth = 15

n_hops = np.arange(1, depth - 1)

n_hidden = 64


### START TEST CONFIGS
# run_split = 1
# train_iters = 10

# depth = 5
# n_hops = [1]

# n_hidden = 16
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for n_hop in n_hops:
    n_vocab = 2 * depth + 1 + StarfishTask.offset

    model_args = {
        'n_vocab': n_vocab,
        'n_hidden': n_hidden,
        'use_bias': False,
        'freeze_emb': True,
    }

    def make_train_args(loss='ce_mask'):
        args = {
            'loss': loss,
            'test_every': 1000,
            'train_iters': train_iters,
            'lr': 1e-3
        }

        args['eval_fns'] = [loss_and_acc]

        if loss == 'ce_mask':
            args['eval_fns'] = eval_fns
            args['print_fn'] = print_gen
        
        return args


    def make_chain(cot=True):
        task_args = {
            'depth': depth,
            'samp_dist': (1, n_hop),
        }

        return StarfishTask(cot=cot, **task_args)
    

    all_cases.extend([
        Case('Full',
                TransformerConfig(n_heads=1,
                                  n_out=n_vocab,
                                  n_layers=1,
                                  pos_emb=False, 
                                  return_final_logits_only=False,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=True,
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(cot=True)
        ), 

        Case('Mix',
                TrLogRegConfig(n_vocab=n_vocab, n_hidden=n_hidden, flatten=False),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(cot=True)
        ), 

        Case('Flat',
                TrLogRegConfig(n_vocab=n_vocab, n_hidden=n_hidden, flatten=True),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(cot=True)
        ), 
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for n_hop in n_hops:
        tt = case.train_task
        test_task = StarfishTask(depth=tt.depth, samp_dist=n_hop, cot=tt.cot)

        case.eval(
            test_task,
            eval_fns=case.train_args['eval_fns'],
            prefix=n_hop
        )

    case.state = None
    case.hist = None
    case.train_args['eval_fns'] = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
