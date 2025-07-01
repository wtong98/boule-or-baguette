"""Proposition task"""

# <codecell>
from datasets import DatasetDict, concatenate_datasets
import jax
import jax.numpy as jnp
import numpy as np
from transformers import DataCollatorWithPadding
from tqdm import tqdm

import sys
sys.path.append('../')
from common import generate, new_seed

try:
    from .prop_gen.to_dataset import get_tokenizer, count_ops
except ImportError:
    from prop_gen.to_dataset import get_tokenizer, count_ops


ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf'

class PropTask:
    or_ops = {'apply Or', 'intro h', 'exact', 'apply True', 'efq'}
    n_vocab = 50257

    def __init__(self, depth, split='train', filter_ops=None, cot=False, batch_size=128) -> None:
        assert batch_size > 1, f'require batch_size={batch_size} > 1'
        assert type(filter_ops) is set, f'filter_ops must be a set, got {type(filter_ops)}'
        
        self.depth = depth
        self.split = split
        self.filter_ops = filter_ops
        self.cot = cot
        self.batch_size = batch_size

        self.ds = None

        self.true_ds = None
        self.max_true = None
        self.false_ds = None
        self.max_false = None

        self.tokenizer = get_tokenizer()
        self.collate = DataCollatorWithPadding(self.tokenizer, return_tensors='np')
    

    def load_ds(self):
        self.ds = DatasetDict.load_from_disk(ds_path)

        max_len = len(self.ds.keys()) // 2
        if self.split == 'train':
            self.true_ds = concatenate_datasets([self.ds[f'True_{i}'] for i in range(1, self.depth+1)])
            self.false_ds = concatenate_datasets([self.ds[f'False_{i}'] for i in range(1, self.depth+1)])
        elif self.split == 'test':
            self.true_ds = concatenate_datasets([self.ds[f'True_{i}'] for i in range(self.depth+1, max_len + 1)])
            self.false_ds = concatenate_datasets([self.ds[f'False_{i}'] for i in range(self.depth+1, max_len + 1)])
        elif self.split == 'single':
            self.true_ds = self.ds[f'True_{self.depth}']
            self.false_ds = self.ds[f'False_{self.depth}']
        else:
            raise ValueError(f'unrecognized split: {self.split}')
        
        if self.filter_ops is not None:
            def matches_ops(ex):
                return set(ex['ops']).issubset(self.filter_ops)

            self.true_ds = self.true_ds.filter(matches_ops)
            self.false_ds = self.false_ds.filter(matches_ops)
        
        self.max_true = len(self.true_ds)
        self.max_false = len(self.false_ds)

        if self.max_true == 0 or self.max_false == 0:
            raise ValueError(f'insufficient examples: max_true={self.max_true} and max_false={self.max_false}')

        if self.cot:
            self.true_ds = self.true_ds.remove_columns(['input_ids']) \
                               .rename_columns({'full_ids': 'input_ids'})
            self.false_ds = self.false_ds.remove_columns(['input_ids']) \
                                .rename_columns({'full_ids': 'input_ids'})
        else:
            self.true_ds = self.true_ds.remove_columns(['full_ids'])
            self.false_ds = self.false_ds.remove_columns(['full_ids'])
    

    def del_ds(self):
        self.ds = None
        self.true_ds = None
        self.max_true = None
        self.false_ds = None
        self.max_false = None

    
    def __next__(self):
        if self.true_ds is None or self.false_ds is None:
            self.load_ds()
        
        true_idxs = np.random.randint(1, self.max_true, size=self.batch_size // 2)
        false_idxs = np.random.randint(1, self.max_false, size=self.batch_size - len(true_idxs))

        true_batch = self.true_ds[true_idxs]
        false_batch = self.false_ds[false_idxs]

        batch = {k: true_batch[k] + false_batch[k] for k in true_batch.keys()}
        batch = self.collate(batch)
        
        xs = batch['input_ids']
        if self.cot:
            ys = xs[:,1:]
            xs = xs[:,:-1]
        else:
            ys = batch['is_true'].astype(int)

        return xs, ys


    def __iter__(self):
        return self

# <codecell>
# task = PropTask(depth=3, batch_size=5, split='train',filter_ops=PropTask.or_ops)
# task_test = PropTask(depth=3, batch_size=5, split='test', filter_ops=PropTask.or_ops)
# task.load_ds()
# task_test.load_ds()

# count_ops(task.true_ds)
# count_ops(task_test.true_ds)


# <codecell>

yes_id = 13138
no_id = 32165
state_id = 5219

# NOTE: un-optimized and very expensive to run
def gen_acc_cot_prop(state, batch, loss=None):
    tot_correct = 0
    all_exs = batch[0]

    for xs in tqdm(all_exs):
        start_idx = jnp.argmax((xs[2:] == state_id)) + 4
        preds = generate(state, xs, idx=start_idx)
        tot_correct += score(xs, preds)
    
    return {'gen_acc': tot_correct / len(all_exs)}
        

@jax.jit
def score(xs, preds):
    is_true = jnp.argmax(xs == yes_id) > 0
    pred_is_true = jnp.argmax(preds == yes_id) > 0
    pred_is_false = jnp.argmax(preds == no_id) > 0

    return is_true * pred_is_true + (1 - is_true) * pred_is_false


def generate(state, xs, idx, beta=1, seed=None):
    if seed is None:
        seed = new_seed()

    xs = xs[None]
    source = jax.random.key(seed)
    while idx < xs.shape[1] - 1:
        key, source = jax.random.split(source)
        xs = _gen_pass(key, state, xs, idx, beta)
        idx += 1
    
    return xs.squeeze()


@jax.jit
def _gen_pass(key, state, xs, idx, beta):
    logits = state.apply_fn({'params': state.params}, xs)
    pred = jax.random.categorical(key, beta * logits[0,idx])
    xs = xs.at[0,idx+1].set(pred)
    return xs

