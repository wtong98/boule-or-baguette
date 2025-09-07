"""
Relationship between size and accuracy on trees of various size
"""

# <codecell>
from pathlib import Path
import pickle

import pandas as pd
import optax
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.transformer import TransformerConfig
from task.prop import *

run_id = new_seed()
print('RUN ID', run_id)

run_split = 18

batch_size = 64
multistep_k = 1
train_iters = 100_000
warmup_iters = 2000
eval_k = 12
max_len = 1024

n_hops = [2, 3, 4, 5, 6, 10]

n_hidden = 768
n_layer = 12
n_head = 12

save_dir = Path('/n/netscratch/pehlevan_lab/Lab/wlt/prop_weights') 


### START TEST CONFIGS
# run_split = 1

# train_iters = 2
# warmup_iters = 1
# batch_size = 4
# eval_k = 1
# multistep_k = 2

# n_hops = [2]

# n_hidden = 64
# n_layer = 1
# n_head = 1

# max_len = 200

# save_dir = Path('.').parent
### END TEST CONFIGS

range_hops = [1] + [(h + 1) for h in n_hops] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))

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
            'k': multistep_k,
            'lr': optax.schedules.warmup_cosine_decay_schedule(
                init_value=1e-4,
                peak_value=3e-4,
                warmup_steps=warmup_iters,
                decay_steps=train_iters,
                end_value=5e-5
            )
        }

        args['eval_fns'] = [loss_and_acc]
        return args


    def make_chain(cot=True, split='train', batch_size=128, **kwargs):
        task_args = {
            'depth': n_hop,
            'split': split,
            'batch_size': batch_size,
            'max_len': max_len,
        }

        return PropTask(cot=cot, **task_args, **kwargs)
    

    all_cases.extend([
        Case('AR_full',
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
                train_task=make_chain(cot=True, split='train', batch_size=batch_size, ds_path=full_ds_path),
                test_task=make_chain(cot=True, split='test', batch_size=batch_size, ds_path=full_ds_path),
                info={'n_hop': n_hop}
        ), 

        Case('AR_imply',
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
                train_task=make_chain(cot=True, split='train', batch_size=batch_size, ds_path=full_ds_path, filter_ops=PropTask.imply_ops),
                test_task=make_chain(cot=True, split='test', batch_size=batch_size, ds_path=full_ds_path, filter_ops=PropTask.imply_ops),
                info={'n_hop': n_hop}
        ), 

        Case('AR_trunc',
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
                train_task=make_chain(cot=True, split='train', batch_size=batch_size, ds_path=trunc_ds_path),
                test_task=make_chain(cot=True, split='test', batch_size=batch_size, ds_path=trunc_ds_path),
                info={'n_hop': n_hop}
        ), 
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    case.run()

    if case.train_task.cot == True:
        case.train_args['eval_fns'].append(gen_acc_cot_prop)

    case.eval(case.test_task, eval_fns=case.train_args['eval_fns'], prefix=(-1, -1), n_iters=eval_k)

    for r in ranges:
        t = case.test_task
        test_task = PropTask(depth=r, 
                             split='range', 
                             cot=t.cot, 
                             batch_size=t.batch_size, 
                             ds_path=t.ds_path, 
                             max_len=max_len,
                             filter_ops=t.filter_ops,
                             padding='max_length')

        case.eval(test_task, eval_fns=case.train_args['eval_fns'], prefix=r, n_iters=eval_k)

    n_hop = case.info['n_hop']
    save_name = f'{case.name}_{n_hop}_weights.pkl'

    with (save_dir / save_name).open('wb') as fp:
        pickle.dump(case.state.params, fp)

    case.state = None
    case.train_args['eval_fns'] = None
    case.train_args['lr'] = None
    case.train_task = None
    case.test_task = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
