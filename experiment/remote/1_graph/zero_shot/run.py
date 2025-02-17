"""
Match task accuracies
"""

# <codecell>
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.mlp import MlpConfig , MixerConfig
from model.transformer import TransformerConfig
from task.graph import Chain, BinaryTreeTiTask

run_id = new_seed()
print('RUN ID', run_id)

run_split = 8

train_iters = 100_000

depth = 10
n_vocab = 2**depth
n_hidden = 256

use_biases = [False, True]
freeze_embs = [False, True]
train_lens = [1, 2, 3, 4]
test_lens = [1, 2, 3, 4, 5, 6, 7, 8]


### START TEST CONFIGS
run_split = 1

train_iters = 100

depth = 5
n_vocab = 2**depth
n_hidden = 256

use_biases = [False, True]
freeze_embs = [False, True]
train_lens = [1]
test_lens = [1]
### END TEST CONFIGS

all_cases = []
test_tasks = []

for use_bias, freeze_emb, train_len in itertools.product(use_biases, freeze_embs, train_lens):
    inc = (1, train_len)
    if train_len == 1:
        samp_dist_sets = [(1, 1)]
    else:
        samp_dist_sets = [(1, inc), (inc, 1), (inc, inc)]

    for train_set in samp_dist_sets:
        model_args = {
            'n_vocab': n_vocab,
            'n_hidden': n_hidden,
            'use_bias': use_bias,
            'freeze_emb': freeze_emb,
        }

        train_args = {
            'loss': 'bce',
            'test_every': 1000,
            'train_iters': train_iters
        }

        def make_chain():
            # return Chain(
            #     BinaryTreeTiTask(depth=depth, samp_dist=train_set[0], on_branch=True),
            #     BinaryTreeTiTask(depth=depth, samp_dist=train_set[1], on_branch=False, fill_gaps=False)
            # )

            return Chain(
                BinaryTreeTiTask(order='fwd', depth=depth, samp_dist=train_set[0], on_branch=True),
                BinaryTreeTiTask(order='rev', depth=depth, samp_dist=train_set[0], on_branch=True),
                BinaryTreeTiTask(order='split', depth=depth, samp_dist=train_set[1], on_branch=False, fill_gaps=False), weights=[3, 1, 2])
        
        def make_test():
            return BinaryTreeTiTask(depth=depth, samp_dist=test_lens[-1], on_branch=False)


        common_train_args = {'order': 'split', 'depth': depth}

        all_cases.extend([
            Case('MLP',
                 MlpConfig(n_layers=1, **model_args),
                 train_args=train_args,
                 train_task=make_chain(),
                 test_task=make_test()
            ),
            
            Case('Mixer',
                 MixerConfig(n_layers=2, n_channels=32, **model_args),
                 train_args=train_args,
                 train_task=make_chain(),
                 test_task=make_test()
            ),
            
            Case('Transformer',
                 TransformerConfig(n_layers=4, n_heads=2, pos_emb=False, **model_args),
                 train_args=train_args,
                 train_task=make_chain(),
                 test_task=make_test()
            )
        ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

for test_len in test_lens:
    task = BinaryTreeTiTask(depth=depth, order=None, samp_dist=test_len, on_branch=True)
    eval_cases(all_cases, eval_task=task, key_name=f'acc_{test_len}_fwd_on')

    task = BinaryTreeTiTask(depth=depth, order='rev', samp_dist=test_len, on_branch=True)
    eval_cases(all_cases, eval_task=task, key_name=f'acc_{test_len}_rev_on')

    task = BinaryTreeTiTask(depth=depth, order=None, samp_dist=test_len, on_branch=False)
    eval_cases(all_cases, eval_task=task, key_name=f'acc_{test_len}_fwd_off')

    task = BinaryTreeTiTask(depth=depth, order='rev', samp_dist=test_len, on_branch=False)
    eval_cases(all_cases, eval_task=task, key_name=f'acc_{test_len}_rev_off')


for case in all_cases:
    case.state = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
