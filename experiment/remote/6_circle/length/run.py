"""
Relationship between size and accuracy on trees of various size
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.mlp import MlpConfig
from model.transformer import TransformerConfig
from task.graph import *

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000

depth = 26
n_hops = np.arange(1, depth - 1)

n_hidden = 512


### START TEST CONFIGS
# run_split = 1
# train_iters = 10

# depth = 5
# n_hops = [1]

# n_hidden = 64
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for n_hop in n_hops:
    n_vocab = 2 * depth + 1 + CircleTask.offset

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

        return CircleTask(cot=cot, **task_args)
    

    all_cases.extend([
        Case('MLP',
                MlpConfig(n_out=1,
                          n_layers=1,
                          **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ), 

        Case('Att',
                TransformerConfig(n_heads=1,
                                  n_out=1,
                                  n_layers=2,
                                  pos_emb=True, 
                                  return_format='final_logit',
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ), 

        Case('Att + MLP',
                TransformerConfig(n_heads=1,
                                  n_out=1,
                                  n_layers=2,
                                  pos_emb=True, 
                                  return_format='final_logit',
                                  n_mlp_layers=2,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ), 

        Case('Att + CoT',
                TransformerConfig(n_heads=1, 
                                  n_out=n_vocab,
                                  n_layers=2,
                                  pos_emb=False, 
                                  return_format=None,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(cot=True)
        ), 

        Case('Att + CoT + MLP',
                TransformerConfig(n_heads=1, 
                                  n_out=n_vocab,
                                  n_layers=2,
                                  pos_emb=False, 
                                  return_format=None,
                                  n_mlp_layers=2,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  **model_args),
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
        test_task = CircleTask(depth=depth, samp_dist=n_hop, cot=tt.cot)

        case.eval(
            test_task,
            eval_fns=case.train_args['eval_fns'],
            prefix=n_hop
        )

    case.state = None
    case.train_args['eval_fns'] = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
