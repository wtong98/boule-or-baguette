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
from task.graph import Chain, BinaryTreeTiTask, bt_rew_fn, bt_rl_loss

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 100_000

depths = [5, 6, 7, 8, 9, 10]
n_hiddens = [128, 256, 512, 1024, 2048, 4096]
n_mlp_layers = [0, 2]
use_layer_norm = [False, True]
use_mup_scale = [False, True]
n_layers = 2


### START TEST CONFIGS
# run_split = 1
# train_iters = 10

# depths = [5]
# n_hiddens = [128]
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for mup_scale, layer_norm, depth, n_hidden, n_mlp_layer \
    in itertools.product(use_mup_scale, use_layer_norm, depths, n_hiddens, n_mlp_layers):

    n_vocab = 2**depth + BinaryTreeTiTask.offset
    train_len_pr = depth - 2
    test_len_pr = depth - 2

    model_args = {
        'n_vocab': n_vocab,
        'n_out': n_vocab,
        'n_hidden': n_hidden,
        'n_layers': n_layers,
        'use_bias': False,
        'freeze_emb': True,
        'mup_scale': False
    }

    def make_train_args(loss='ce_mask'):
        return {
            'loss': loss,
            'test_every': 1000,
            'train_iters': train_iters,
            'eval_fns': eval_fns,
            'print_fn': print_gen
        }


    def make_chain(unwrap=False):
        task_args = {
            'depth': depth,
            'samp_dist': (1, train_len_pr),
            'cot': True
        }

        return Chain(
            BinaryTreeTiTask(on_branch=True, unwrap=unwrap, **task_args),
            BinaryTreeTiTask(on_branch=False, unwrap=unwrap, **task_args))
    

    all_cases.extend([
        Case('Transformer',
                TransformerConfig(n_heads=1, 
                                  pos_emb=False, 
                                  return_final_logits_only=False,
                                  n_mlp_layers=n_mlp_layer,
                                  layer_norm=layer_norm,
                                  residual_connections=False,
                                  mup_scale=mup_scale
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(unwrap=False),
                test_task=make_chain(unwrap=False)
        )
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    case.eval(
        case.train_task,
        eval_fns=eval_fns
    )

    case.state = None
    case.train_args['eval_fns'] = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
