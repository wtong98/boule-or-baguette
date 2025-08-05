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

run_split = 36

train_iters = 200_000
# warmup_iters = 2000

depth = 10
n_hop = 5
n_hops_test = [5, 6, 7, 8, 9]

all_n_layer = [1]

n_arms = (2**np.linspace(3, 9, num=40)).astype(int) * 2
n_widths = (2**np.linspace(3, 9, num=40)).astype(int) * 2

lrs = [5e-3]
gammas = [1]

### START TEST CONFIGS
# run_split = 1
# train_iters = 10_000

# depth = 10
# n_hop = 5
# n_hops_test = [7]

# all_n_layer = [1]
# n_arms = [100]
# n_widths = [256]

# lrs = [5e-3]
# gammas = [1]
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot1]

for lr, gamma, n_arm, n_hidden in itertools.product(lrs, gammas, n_arms, n_widths):
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
            'lr': lr * gamma,
            'gamma': gamma,
            # 'lr': optax.schedules.warmup_cosine_decay_schedule(
            #     init_value=1e-3,
            #     peak_value=1e-2,
            #     warmup_steps=warmup_iters,
            #     decay_steps=train_iters,
            #     end_value=1e-4
            # )
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
            'nouveau': True
        }

        return StarfishTask(cot=cot, trace_to_start=ttr, **task_args, **kwargs)
    

    all_cases.extend([
        Case(f'AR (direct)',
                TransformerConfig(n_heads=1,
                                n_out=1,
                                n_layers=1,
                                pos_emb=False, 
                                return_format='final_logit_up_to_pad',
                                n_mlp_layers=2,
                                layer_norm=False,
                                residual_connections=False,
                                mup_scale=True,
                                unif_att=True,
                                linear_att=False,
                                **model_args),
                train_args=make_train_args('bce'),
                train_task=make_chain(cot=True, ttr=True, force_bin_label=True)
        ), 
    ])

        # if n_layers == 1 and resid == True and cot == True:
        #     all_cases.extend([
        #         Case(f'(cot={cot},n_layers={n_layers},resid={resid},LN={layer_norm},unif_att)',
        #                 TransformerConfig(n_heads=1,
        #                                 n_out=n_vocab if cot else 1,
        #                                 n_layers=n_layers,
        #                                 pos_emb=False, 
        #                                 return_format=None if cot else 'final_logit',
        #                                 n_mlp_layers=2,
        #                                 layer_norm=layer_norm,
        #                                 residual_connections=resid,
        #                                 mup_scale=True,
        #                                 linear_att=False,
        #                                 unif_att=True,
        #                                 **model_args),
        #                 train_args=make_train_args('ce_mask' if cot else 'bce'),
        #                 train_task=make_chain(cot=cot, ttr=True)
        #         ), 
        #     ])
    
    
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for n_hop in n_hops_test:
        tt = case.train_task
        test_task = StarfishTask(n_arms=tt.n_arms, depth=tt.depth, samp_dist=n_hop, cot=tt.cot, trace_to_start=tt.trace_to_start, nouveau=tt.nouveau, force_bin_label=tt.force_bin_label)

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
