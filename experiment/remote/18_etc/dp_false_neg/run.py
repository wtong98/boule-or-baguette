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

run_split = 1

train_iters = 50_000

n_arms = [50]
n_depths = [10]
n_widths = [512]

n_hop_props = [0.5]
lrs = [1e-2]

all_n_layer = [1]
max_batch_size = 128

seed = None


### START TEST CONFIGS
# run_split = 1
# train_iters = 50

# n_arms = [2]
# n_depths = [10]
# n_widths = [32]

# all_n_layer = [1]
# n_hop_props = [0.25]
# lrs = [1e-2]

# max_batch_size = 128
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot1]

for lr, n_hop_prop, n_arm, depth, n_hidden, n_layer in itertools.product(lrs, n_hop_props, n_arms, n_depths, n_widths, all_n_layer):
    n_hop = np.round(n_hop_prop * depth).astype(int)
    n_vocab = n_arm * depth + 1 + StarfishTask.offset

    batch_size = n_arm * depth
    batch_size, k = split_batch(batch_size, max_batch_size)

    model_args = {
        'n_vocab': n_vocab,
        'n_hidden': n_hidden,
        'use_bias': False,
        'freeze_emb': True,
        'dtype': jnp.bfloat16
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


    def make_chain(cot=True, ttr=False):
        task_args = {
            'depth': depth,
            'samp_dist': (1, n_hop),
            'n_arms': n_arm,
            'nouveau': True,
            'batch_size': batch_size,
            'force_bin_label': True
        }

        return StarfishTask(cot=cot, trace_to_start=ttr, **task_args)
    

    all_cases.extend([
        Case(f'DP',
                TransformerConfig(n_heads=1,
                                n_out=1,
                                n_layers=n_layer,
                                pos_emb=False, 
                                return_format='final_logit',
                                n_mlp_layers=2,
                                layer_norm=False,
                                residual_connections=False,
                                mup_scale=True,
                                linear_att=False,
                                unif_att=True,
                                **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=False),
                info={'n_hop_prop': n_hop_prop, 'n_hop': n_hop}
        ), 
    ])

    
all_cases = split_cases(all_cases, run_split, shuffle_seed=seed)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    tt = case.train_task
    test_task = StarfishTask(
        depth=tt.depth,
        n_arms=tt.n_arms,
        samp_dist=(tt.samp_dist[-1].item() + 1, tt.depth - 1),
        batch_size=512)

    xs, ys = next(test_task)
    logits = case.state.apply_fn({'params': case.state.params}, xs)

    fnr = np.mean(logits[ys == 1] < 0)
    case.info['fnr'] = fnr

    case.train_args['eval_fns'] = None
    case.state = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')