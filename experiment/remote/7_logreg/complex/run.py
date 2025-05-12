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
from model.transformer import TransformerConfig, sinusoidal_init
from task.graph import *


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
            pos_emb_shape = (1, self.config.max_len, x.shape[-1])
            pos_embedding = sinusoidal_init(max_len=self.config.max_len)(None,
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

            if self.config.n_out == 1:
                x = x.flatten()

        return x

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000
cots = [False, True]
pos_emb = [False, True]
depth = 20
lrs = [1e-3, 1e-4, 1e-5]

n_hops = np.arange(1, depth - 1)

n_hidden = 512


### START TEST CONFIGS
run_split = 1
train_iters = 100

depth = 5
n_hops = [1]

n_hidden = 16
cots = [False]
pos_emb = [True]
lrs = [1e-3]
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for lr, pos_emb, cot, n_hop in itertools.product(lrs, pos_emb, cots, n_hops):
    n_vocab = 2 * depth + 1 + StarfishTask.offset

    model_args = {
        'n_vocab': n_vocab,
        'n_hidden': n_hidden,
        'n_out': n_vocab if cot else 1,
        'return_final_logits_only': False if cot else True,
        'pos_emb': pos_emb,
        'max_len': 128
    }

    def make_train_args():
        loss = 'ce_mask' if cot else 'bce'

        args = {
            'loss': loss,
            'test_every': 1000,
            'train_iters': train_iters,
            'lr': lr
        }

        args['eval_fns'] = [loss_and_acc]

        if loss == 'ce_mask':
            args['eval_fns'] = eval_fns
            args['print_fn'] = print_gen
        
        return args


    def make_chain():
        task_args = {
            'depth': depth,
            'samp_dist': (1, n_hop),
        }

        return StarfishTask(cot=cot, **task_args)
    

    all_cases.extend([
        Case('Linear 1',
                TransformerConfig(n_heads=1,
                                  n_layers=1,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=True,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('Linear 2',
                TransformerConfig(n_heads=1,
                                  n_layers=2,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=True,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('Linear 2 + resid',
                TransformerConfig(n_heads=1,
                                  n_layers=2,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=True,
                                  mup_scale=True,
                                  linear_att=True,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('Linear 4',
                TransformerConfig(n_heads=1,
                                  n_layers=4,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=True,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('Linear 4 + resid',
                TransformerConfig(n_heads=1,
                                  n_layers=4,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=True,
                                  mup_scale=True,
                                  linear_att=True,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('SM 1',
                TransformerConfig(n_heads=1,
                                  n_layers=1,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=False,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        # Case('SM 2',
        #         TransformerConfig(n_heads=1,
        #                           n_layers=1,
        #                           n_mlp_layers=0,
        #                           layer_norm=False,
        #                           residual_connections=False,
        #                           mup_scale=True,
        #                           linear_att=False,
        #                           use_bias=False,
        #                           freeze_emb=False,
        #                           **model_args),
        #         train_args=make_train_args(),
        #         train_task=make_chain()
        # ), 

        # Case('SM 4',
        #         TransformerConfig(n_heads=1,
        #                           n_layers=4,
        #                           n_mlp_layers=0,
        #                           layer_norm=False,
        #                           residual_connections=False,
        #                           mup_scale=True,
        #                           linear_att=False,
        #                           use_bias=False,
        #                           freeze_emb=False,
        #                           **model_args),
        #         train_args=make_train_args(),
        #         train_task=make_chain()
        # ), 

        # Case('SM 2 + MLP',
        #         TransformerConfig(n_heads=1,
        #                           n_layers=2,
        #                           n_mlp_layers=2,
        #                           layer_norm=False,
        #                           residual_connections=False,
        #                           mup_scale=True,
        #                           linear_att=False,
        #                           use_bias=False,
        #                           freeze_emb=False,
        #                           **model_args),
        #         train_args=make_train_args(),
        #         train_task=make_chain()
        # ), 

        Case('SM 2 + MLP + resid',
                TransformerConfig(n_heads=1,
                                  n_layers=2,
                                  n_mlp_layers=2,
                                  layer_norm=False,
                                  residual_connections=True,
                                  mup_scale=True,
                                  linear_att=False,
                                  use_bias=False,
                                  freeze_emb=False,
                                  **model_args),
                train_args=make_train_args(),
                train_task=make_chain()
        ), 

        Case('Mix (dot)',
                TrConfig(**model_args),
                train_args=make_train_args(),
                train_task=make_chain()
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
