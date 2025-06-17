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
from model.transformer import TransformerConfig, TrConfig
from task.graph import *

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000

depth = 10
n_hop = 5
n_hops_test = [6, 7, 8, 9]

# n_arms = (2**np.linspace(6, 10, num=20)).astype(int)
# n_widths = (2**np.linspace(6, 10, num=20)).astype(int)
n_arms = [2, 3, 5, 8, 12, 17]
n_widths = [64, 128, 256, 512, 1024, 2048, 4096]

### START TEST CONFIGS
# run_split = 1
# train_iters = 1000

# depth = 10
# n_hop = 5
# n_hops_test = [6]

# n_arms = [3]
# n_widths = [128]
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for n_arm, n_hidden in itertools.product(n_arms, n_widths):
    n_vocab = n_arm * depth + 1 + StarfishTask.offset

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


    def make_chain(cot=True, ttr=False):
        task_args = {
            'depth': depth,
            'samp_dist': (1, n_hop),
            'n_arms': n_arm
        }

        return StarfishTask(cot=cot, trace_to_start=ttr, **task_args)
    

    all_cases.extend([
        Case('Zero (simp lin att, sin PE)',
                TrConfig(n_vocab=n_vocab, 
                         pos_emb=True,
                         rand_pos_emb=False,
                         n_out=1,
                         n_hidden=n_hidden, 
                         return_final_logits_only=True),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ),

        Case('Zero (simp lin att, rand PE)',
                TrConfig(n_vocab=n_vocab, 
                         pos_emb=True,
                         rand_pos_emb=True,
                         n_out=1,
                         n_hidden=n_hidden, 
                         return_final_logits_only=True),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ),

        Case('Zero (lin att)',
                TransformerConfig(n_heads=1,
                                  n_out=1,
                                  n_layers=1,
                                  pos_emb=True, 
                                  return_final_logits_only=True,
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  mup_scale=True,
                                  linear_att=True,
                                  **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False)
        ), 
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for n_hop in n_hops_test:
        tt = case.train_task
        test_task = StarfishTask(n_arms=tt.n_arms, depth=tt.depth, samp_dist=n_hop, cot=tt.cot, trace_to_start=tt.trace_to_start)

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
