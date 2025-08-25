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
from model.transformer import TransformerConfig
from task.graph import *

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000

depth = 10
n_hop = 5
n_hops_test = [5, 6, 7, 8, 9]

switch = [False]  # resid
all_lrs = [1e-2]

n_arms = (2**np.linspace(3, 9, num=30)).astype(int) * 2
n_widths = (2**np.linspace(3, 9, num=30)).astype(int) * 2
max_batch_size = 1024

all_n_layer = [1, 4]

### START TEST CONFIGS
run_split = 1
train_iters = 500

depth = 10
n_hop = 5
n_hops_test = [7]

all_n_layer = [2]
switch = [False]
n_arms = [10]
n_widths = [256]

max_batch_size = 27
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot1]

for lr, resid, n_arm, n_hidden, n_layer in itertools.product(all_lrs, switch, n_arms, n_widths, all_n_layer):
    n_vocab = n_arm * depth + 1 + StarfishTask.offset
    batch_size = n_arm * depth
    batch_size, k = split_batch(batch_size, max_batch_size)

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
            'lr': lr / n_layer,
            'k': k
        }

        args['eval_fns'] = [loss_and_acc]

        if loss == 'ce_mask':
            args['eval_fns'] = eval_fns
            args['print_fn'] = print_gen
        
        return args


    def make_chain(cot=True, ttr=False, **kwargs):
        task_args = {
            'depth': depth,
            'samp_dist': (1, n_hop),
            'n_arms': n_arm,
            'nouveau': True,
            'batch_size': batch_size
        }

        return StarfishTask(cot=cot, trace_to_start=ttr, **task_args, **kwargs)
    

    all_cases.extend([
        Case(f'Zero',
                TransformerConfig(n_heads=1,
                                n_out=1,
                                n_layers=n_layer,
                                pos_emb=False, 
                                return_format='final_logit',
                                n_mlp_layers=2,
                                layer_norm=False,
                                residual_connections=resid,
                                mup_scale=True,
                                linear_att=False,
                                unif_att=True,
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
        test_task = StarfishTask(n_arms=tt.n_arms, 
                                 depth=tt.depth, 
                                 samp_dist=n_hop, 
                                 cot=tt.cot, 
                                 trace_to_start=tt.trace_to_start, 
                                 nouveau=tt.nouveau, 
                                 force_bin_label=tt.force_bin_label,
                                 batch_size=1024)

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
