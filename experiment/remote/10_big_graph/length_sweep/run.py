"""
Length-wise generalization on Starfish, sweeping across more hyperparameters
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

depths = [10, 30]
n_hop_props = [0, 0.25, 0.5, 0.75]
n_arms = [2, 10, 30]

n_hidden = 512

n_layers = [2]
use_layer_norm = [False, True]
use_resid = [False, True]
mup_scale = [True]
use_trace_to_start = [False, True]
n_mlp_layers = [0, 2]
use_nonlinears = [False, True]
use_cots = [False, True]

### START TEST CONFIGS
# run_split = 1
# train_iters = 10
# n_hidden = 64

# depths = [5]
# n_hop_props = [0]
# n_arms = [3]

# n_layers = [2]
# use_layer_norm = [False]
# use_resid = [False]
# mup_scale = [True]

# use_trace_to_start = [False]
# n_mlp_layers = [0]

# use_nonlinears = [False, True]
# use_cots = [False, True]
### END TEST CONFIGS

all_cases = []

for n_arm, n_hop_prop, depth in itertools.product(n_arms, n_hop_props, depths):
    n_vocab = n_arm * depth + 1 + StarfishTask.offset
    n_hop = int(np.round(n_hop_prop * depth)) if n_hop_prop > 0 else 1

    model_args = {
        'n_vocab': n_vocab,
        'n_hidden': n_hidden,
        'use_bias': False,
        'freeze_emb': True,
    }

    def make_train_args(loss='ce_mask'):
        args = {
            'loss': loss,
            'test_every': 10_000,
            'train_iters': train_iters,
        }

        args['eval_fns'] = [loss_and_acc]

        if loss == 'ce_mask':
            args['eval_fns'].append(gen_acc_cot)
            args['print_fn'] = print_gen
        
        return args


    def make_chain(**kwargs):
        task_args = {
            'n_arms': n_arm,
            'depth': depth,
            'samp_dist': (1, n_hop)
        }

        return StarfishTask(**task_args, **kwargs)
    

    # all_cases.extend([
    #     Case('MLP',
    #             MlpConfig(n_out=1,
    #                       n_layers=1,
    #                       **model_args),
    #             train_args=make_train_args('bce'),
    #             train_task=make_chain(cot=False)
    #     ), 
    # ])

    for use_nonlinear, n_layer, layer_norm, resid, mup, trace_to_start, mlp_layers, cot \
        in itertools.product(use_nonlinears, n_layers, use_layer_norm, use_resid, mup_scale, use_trace_to_start, n_mlp_layers, use_cots):
        if cot is False and trace_to_start is True:
            continue

        all_cases.append(
            Case(f'(lnorm={layer_norm},resid={resid},full_tr={trace_to_start},n_mlp={mlp_layers})',
                    TransformerConfig(n_heads=1, 
                                    n_out=n_vocab if cot else 1,
                                    n_layers=n_layer,
                                    pos_emb=not cot, 
                                    return_final_logits_only=not cot,
                                    n_mlp_layers=mlp_layers,
                                    layer_norm=layer_norm,
                                    residual_connections=resid,
                                    mup_scale=mup,
                                    linear_att=not use_nonlinear,
                                    **model_args),
                    train_args=make_train_args('ce_mask' if cot else 'bce'),
                    train_task=make_chain(cot=cot, trace_to_start=trace_to_start),
                    info={'n_hop_prop': n_hop_prop}
            )
        )

    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    tt = case.train_task
    for n_hop in range(tt.depth):
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
