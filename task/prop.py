"""Proposition task"""

# <codecell>
from datasets import DatasetDict, concatenate_datasets
import jax
import jax.numpy as jnp
import numpy as np
from transformers import DataCollatorWithPadding

import sys
sys.path.append('../')
from common import generate

try:
    from .prop_gen.to_dataset import get_tokenizer
except ImportError:
    from prop_gen.to_dataset import get_tokenizer


ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf'

class PropTask:
    def __init__(self, depth, split='train', cot=False, batch_size=128) -> None:
        self.depth = depth
        self.split = split
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
        else:
            raise ValueError(f'unrecognized split: {self.split}')
        
        self.max_true = len(self.true_ds)
        self.max_false = len(self.false_ds)

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
        false_idxs = np.random.randint(1, self.max_false, size=self.batch_size // 2)

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


# yes_id = 13138
# no_id = 32165
# state_id = 5219

# task = PropTask(depth=3, batch_size=4, cot=True)
# xs, _ = next(task)

# def gen_acc_cot_prop(batch):
#     pass

# yes_id = 13138
# no_id = 32165
# state_id = 5219

# def score(state, xs):
#     start_idx = jnp.argmax((xs[2:] == state_id)) + 4
#     is_true = jnp.argmax(xs == yes_id) > 0

#     preds = generate(state, xs, idx=start_idx)
#     pred_is_true = jnp.argmax(preds == yes_id) > 0

#     return is_true == pred_is_true
    

# score(xs)
# jnp.argmax(no_id == xs[0])

# <codecell>

# def extract_pred(traj):
    # # assumes no/yes classification offset by 1 for padding
    # no_occ = jnp.argmax(traj == 1, axis=1)
    # no_occ = jnp.where(no_occ == 0, jnp.inf, no_occ)
    # yes_occ = jnp.argmax(traj == 2, axis=1)
    # yes_occ = jnp.where(yes_occ == 0, jnp.inf, yes_occ)

    # preds = jnp.argmin(jnp.stack((no_occ, yes_occ), axis=1), axis=1) + 1
    # preds = jnp.where(no_occ != yes_occ, preds, jnp.inf)
    # return preds
