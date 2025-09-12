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

run_split = 400

train_iters = 50_000

n_arms = [10]
n_depths = (2**np.linspace(3, 8.5, num=20)).astype(int) * 2
n_widths = (2**np.linspace(3, 8.5, num=20)).astype(int) * 2

all_n_layer = [1]

test_n_hop_props = [0.25, 0.5, 0.7, 0.95]
n_hop_props = [0.5]

max_batch_size = 256

shuffle_seed = 100

### START TEST CONFIGS
# run_split = 2
# train_iters = 100
# batch_size = 10

# n_arms = [10]
# n_depths = [10]
# n_widths = [32]

# all_n_layer = [1]
# n_hop_props = [0.25, 0.5, 0.75, 0.95]

# max_batch_size = 27
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot1]

for n_hop_prop, n_arm, n_hidden, depth, n_layer in itertools.product(n_hop_props, n_arms, n_widths, n_depths, all_n_layer):
    n_hop = np.round(n_hop_prop * depth).astype(int)
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
            'lr': 1e-2 / n_layer,
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
        # Case(f'AR full',
        #         TransformerConfig(n_heads=1,
        #                         n_out=n_vocab,
        #                         n_layers=1,
        #                         pos_emb=False, 
        #                         return_format=None,
        #                         n_mlp_layers=2,
        #                         layer_norm=False,
        #                         residual_connections=True,
        #                         mup_scale=True,
        #                         linear_att=False,
        #                         unif_att=True,
        #                         **model_args),
        #         train_args=make_train_args('ce_mask'),
        #         train_task=make_chain(cot=True, ttr=True),
        #         info={'n_hop_prop': n_hop_prop}
        # ), 

        Case(f'AR direct',
                TransformerConfig(n_heads=1,
                                n_out=1,
                                n_layers=n_layer,
                                pos_emb=False, 
                                return_format='final_logit_up_to_pad',
                                n_mlp_layers=2,
                                layer_norm=False,
                                residual_connections=False,
                                mup_scale=True,
                                linear_att=False,
                                unif_att=True,
                                **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=True, ttr=True, force_bin_label=True),
                info={'n_hop_prop': n_hop_prop}
        ), 
    ])
    
    
    
all_cases = split_cases(all_cases, run_split, shuffle_seed=shuffle_seed)
print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for n_hop_prop in test_n_hop_props:
        tt = case.train_task
        n_hop = np.round(tt.depth * n_hop_prop).astype(int)
        test_task = StarfishTask(n_arms=tt.n_arms, 
                                 depth=tt.depth, 
                                 samp_dist=n_hop, 
                                 cot=tt.cot, 
                                 trace_to_start=tt.trace_to_start, 
                                 nouveau=tt.nouveau, 
                                 batch_size=1024, 
                                 force_bin_label=tt.force_bin_label)

        case.eval(
            test_task,
            eval_fns=case.train_args['eval_fns'],
            prefix=n_hop_prop,
            n_iters=1
        )

    case.state = None
    case.train_args['eval_fns'] = None
    case.train_args['lr'] = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')
