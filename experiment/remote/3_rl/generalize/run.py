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
from model.transformer import TransformerConfig
from task.graph import Chain, BinaryTreeTiTask, bt_rew_fn, bt_rl_loss

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iters = 50_000

depth = 6
n_vocab = 2**depth + BinaryTreeTiTask.offset
n_hiddens = [128, 512]
n_layers = [2]

train_lens_pr = [1, 2, 3]
train_lens_rl = [1, 2, 3]
test_lens = [1, 2, 3, 4]


### START TEST CONFIGS
# run_split = 1

# train_iters = 10

# depth = 5
# n_vocab = 2**depth + BinaryTreeTiTask.offset
# n_hiddens = [128]
# n_layers = [1]

# use_biases = [False]
# freeze_embs = [True]
# train_lens_pr = [1]
# train_lens_rl = [1]
# test_lens = [1]
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for train_len_pr, train_len_rl, n_hidden, n_layer \
    in itertools.product(train_lens_pr, train_lens_rl, n_hiddens, n_layers):

    model_args = {
        'n_vocab': n_vocab,
        'n_out': n_vocab,
        'n_hidden': n_hidden,
        'n_layers': n_layer,
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
    

    def make_test(unwrap=False):
        return BinaryTreeTiTask(depth=depth, samp_dist=test_lens[-1], on_branch=False, cot=True, unwrap=unwrap)


    all_cases.extend([
        Case('Transformer',
                TransformerConfig(n_heads=1, 
                                  pos_emb=False, 
                                  return_final_logits_only=False, 
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(unwrap=False),
                test_task=make_test(unwrap=False),
                info={'etc': {'train_len_rl': train_len_rl}}
        )
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for test_len in test_lens:
        case.eval(
            BinaryTreeTiTask(depth=depth, samp_dist=test_len, on_branch=True, cot=True),
            eval_fns=eval_fns,
            prefix=f'on_pr_{test_len}'
        )

        case.eval(
            BinaryTreeTiTask(depth=depth, samp_dist=test_len, on_branch=False, cot=True),
            eval_fns=eval_fns,
            prefix=f'off_pr_{test_len}'
        )
    
    # RL training
    task_args = {
        'depth': depth,
        'samp_dist': (1, case.info['etc']['train_len_rl']),
        'rl_prompt': True,
        'batch_size': 128
    }

    train_task = Chain(
        BinaryTreeTiTask(on_branch=True, **task_args),
        BinaryTreeTiTask(on_branch=False, fill_gaps=False, **task_args))
    
    rl_state = create_train_state(
        model=case.config.to_model(),
        params=case.state.params,
        lr=3e-5,
    )

    case.state, rl_hist = reinforce(rl_state, train_task, 
                                    action_fn=gen2, 
                                    reward_fn=bt_rew_fn, 
                                    rl_loss=bt_rl_loss,
                                    train_iters=4 * train_iters,
                                    test_every=1000,
                                    eval_fns=[gen_acc_rl]
                                    )
    
    case.info['etc']['rl_hist'] = rl_hist

    for test_len in test_lens:
        case.eval(
            BinaryTreeTiTask(depth=depth, samp_dist=test_len, on_branch=True, rl_prompt=True),
            eval_fns=[gen_acc_rl],
            prefix=f'on_rl_{test_len}'
        )

        case.eval(
            BinaryTreeTiTask(depth=depth, samp_dist=test_len, on_branch=False, rl_prompt=True),
            eval_fns=[gen_acc_rl],
            prefix=f'off_rl_{test_len}'
        )
    
    case.state = None
    case.train_args['eval_fns'] = None



df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
