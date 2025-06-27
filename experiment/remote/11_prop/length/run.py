"""
Relationship between size and accuracy on trees of various size
"""

# <codecell>
from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.transformer import TransformerConfig
from task.prop import *

run_id = new_seed()
print('RUN ID', run_id)

run_split = 8

train_iters = 100_000
batch_size = 128  # TODO: need gradient accumulation to handle large batch sizes
eval_batch_size = 50

n_hops = [2, 3, 4, 5]

n_hidden = 768
n_layer = 12
n_head = 12

save_dir = Path('~/scratch/prop_weights')


### START TEST CONFIGS
run_split = 1

train_iters = 3
batch_size = 4
eval_batch_size = 2

n_hops = [2]

n_hidden = 100
n_layer = 1
n_head = 1

save_dir = Path('.').parent
### END TEST CONFIGS

all_cases = []

for n_hop in n_hops:
    n_vocab = PropTask.n_vocab

    model_args = {
        'n_vocab': n_vocab,
        'n_hidden': n_hidden,
        # 'use_bias': False,
        # 'freeze_emb': True,
    }

    def make_train_args(loss='ce_mask'):
        args = {
            'loss': loss,
            'test_every': 1_000,
            'test_iters': 1,
            'train_iters': train_iters,
        }

        args['eval_fns'] = [loss_and_acc]
        return args


    def make_chain(cot=True, split='train', batch_size=128):
        task_args = {
            'depth': n_hop,
            'split': split,
            'batch_size': batch_size
        }

        return PropTask(cot=cot, **task_args)
    

    all_cases.extend([
        # Case('Zero',
        #         TransformerConfig(n_heads=n_head,
        #                           n_out=1,
        #                           n_layers=n_layer,
        #                           pos_emb=True, 
        #                           return_format='final_logit_up_to_pad',
        #                           n_mlp_layers=2,
        #                           layer_norm=True,
        #                           residual_connections=True,
        #                           mup_scale=True,
        #                           linear_att=False,
        #                           **model_args),
        #         train_args=make_train_args('bce'),
        #         train_task=make_chain(cot=False, split='train', batch_size=batch_size),
        #         test_task=make_chain(cot=False, split='test', batch_size=batch_size)
        # ), 

        Case('AR full',
                TransformerConfig(n_heads=n_head, 
                                  n_out=n_vocab,
                                  n_layers=n_layer,
                                  pos_emb=False, 
                                  return_format=None,
                                  n_mlp_layers=2,
                                  layer_norm=True,
                                  residual_connections=True,
                                  mup_scale=True,
                                  linear_att=False,
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(cot=True, split='train', batch_size=batch_size),
                test_task=make_chain(cot=True, split='test', batch_size=batch_size)
        ), 

    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    case.run()
    case.state.params['Dense_0']['kernel'].block_until_ready()

    if case.train_task.cot == True:
        case.train_args['eval_fns'].append(gen_acc_cot_prop)

    case.test_task.batch_size = eval_batch_size
    case.eval(case.test_task, eval_fns=case.train_args['eval_fns'], prefix=n_hop, n_iters=1)

    save_name = f'{case.name}_{n_hop}_weights.pkl'

    with (save_dir / save_name).open('wb') as fp:
        pickle.dump(case.state.params, fp)

    case.state = None
    case.train_args['eval_fns'] = None
    case.train_task = None
    case.test_task = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
