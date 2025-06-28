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
from task.graph import StarfishTask, bt_rew_fn, bt_rl_loss

run_id = new_seed()
print('RUN ID', run_id)

run_split = 12

train_iter_set = [0, 25, 25_000]
train_iter_rl = 200_000

depth = 25
n_vocab = 2 * depth + 1 + StarfishTask.offset

n_hiddens = [32, 128, 512]
mup_scale = [True]
batch_size = 128

n_layers = [2]

trace_to_start = [False, True]
train_lens = [5]
test_lens = np.arange(1, depth-1)


### START TEST CONFIGS
# run_split = 1

# train_iter_set = [10]
# train_iter_rl = 20

# depth = 5
# n_vocab = 2 * depth + 1 + StarfishTask.offset

# n_hiddens = [32]
# mup_scale = [True]
# batch_size = 128

# n_layers = [2]

# trace_to_start = [False]
# train_lens = [1]
# test_lens = np.arange(1, depth-1)
### END TEST CONFIGS

all_cases = []

eval_fns = [loss_and_acc, gen_acc_cot]

for train_len_idx, mup, ttr, n_hidden, n_layer, train_iters \
    in itertools.product(range(len(train_lens)), mup_scale, trace_to_start, n_hiddens, n_layers, train_iter_set):

    model_args = {
        'n_vocab': n_vocab,
        'n_out': n_vocab,
        'n_hidden': n_hidden,
        'n_layers': n_layer,
        'use_bias': False,
        'freeze_emb': True,
        'mup_scale': mup
    }

    def make_train_args(loss='ce_mask'):
        return {
            'loss': loss,
            'test_every': 1000,
            'train_iters': train_iters,
            'eval_fns': eval_fns,
            'print_fn': print_gen,
            'lr': 5e-4
        }

    def make_chain(**kwargs):
        task_args = {
            'depth': depth,
            'samp_dist': (1, train_lens[train_len_idx]),
            'cot': True,
        }

        return StarfishTask(**task_args, **kwargs)
    

    all_cases.extend([
        Case('Transformer',
                TransformerConfig(n_heads=1, 
                                  pos_emb=False, 
                                  return_format=None, 
                                  n_mlp_layers=0,
                                  layer_norm=False,
                                  residual_connections=False,
                                  linear_att=True,
                                  **model_args),
                train_args=make_train_args('ce_mask'),
                train_task=make_chain(trace_to_start=ttr),
                info={'etc': {'train_len_rl': train_lens[i]}}
        ) for i in range(train_len_idx, len(train_lens))
    ])
    
all_cases = split_cases(all_cases, run_split)

print('CASES', all_cases)

for case in tqdm(all_cases):
    print('RUNNING', case.name)
    case.run()

    for n_hop in test_lens:
        tt = case.train_task
        test_task = StarfishTask(depth=tt.depth, samp_dist=n_hop, rl_prompt=True)

        case.eval(
            test_task,
            eval_fns=[gen_acc_rl],
            prefix=f'pr_{n_hop}'
        )
    
    # RL training
    task_args = {
        'depth': depth,
        'samp_dist': (1, case.info['etc']['train_len_rl']),
        'rl_prompt': True,
        'batch_size': 128
    }

    train_task = StarfishTask(**task_args)
    
    rl_state = create_train_state(
        model=case.config.to_model(),
        params=case.state.params,
        lr=3e-5,
    )

    case.state, rl_hist = reinforce(rl_state, train_task, 
                                    action_fn=gen2, 
                                    reward_fn=bt_rew_fn, 
                                    rl_loss=bt_rl_loss,
                                    train_iters=train_iter_rl,
                                    test_every=1000,
                                    eval_fns=[gen_acc_rl]
                                    )
    
    case.info['etc']['rl_hist'] = rl_hist

    for n_hop in test_lens:
        tt = case.train_task
        test_task = StarfishTask(depth=tt.depth, samp_dist=n_hop, rl_prompt=True)

        case.eval(
            test_task,
            eval_fns=[gen_acc_rl],
            prefix=f'rl_{n_hop}'
        )
    
    case.state = None
    case.train_args['eval_fns'] = None


df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')

# %%
