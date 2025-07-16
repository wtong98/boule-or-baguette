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


ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_full'


class PropTask:
    or_ops = {'apply Or', 'cases Or', 'intro h', 'exact', 'apply True', 'efq'}
    imply_ops = {'split Imply', 'intro h', 'exact', 'apply True', 'efq'}
    n_vocab = 50257

    def __init__(self, depth, split='train', filter_ops=None, cot=False, batch_size=128) -> None:
        assert batch_size > 1, f'require batch_size={batch_size} > 1'
        
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
    

    def _slice_dataset(self, prefix, start, stop):
        all_ds = []
        for i in range(start, stop):
            key = f'{prefix}_{i}'
            if key in self.ds:
                all_ds.append(self.ds[key])

        return concatenate_datasets(all_ds)
    

    def load_ds(self):
        self.ds = DatasetDict.load_from_disk(ds_path)

        max_len = len(self.ds.keys()) // 2
        if self.split == 'train':
            self.true_ds = self._slice_dataset(True, 1, self.depth + 1)
            self.false_ds = self._slice_dataset(False, 1, self.depth + 1)
        elif self.split == 'test':
            self.true_ds = self._slice_dataset(True, self.depth + 1, max_len + 1)
            self.false_ds = self._slice_dataset(False, self.depth + 1, max_len + 1)
        elif self.split == 'range':
            self.true_ds = self._slice_dataset(True, self.depth[0], self.depth[1])
            self.false_ds = self._slice_dataset(False, self.depth[0], self.depth[1])
        else:
            raise ValueError(f'unrecognized split: {self.split}')
        
        if self.filter_ops is not None:
            assert type(self.filter_ops) is set, f'filter_ops must be a set, got {type(self.filter_ops)}'

            def matches_ops(ex):
                return set(ex['ops']).issubset(self.filter_ops)

            self.true_ds = self.true_ds.filter(matches_ops)
            self.false_ds = self.false_ds.filter(matches_ops)
        
        self.max_true = len(self.true_ds)
        self.max_false = len(self.false_ds)

        if self.max_true == 0 or self.max_false == 0:
            print(f'warn: insufficient examples: max_true={self.max_true} and max_false={self.max_false}')

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
        
        true_idxs = []
        false_idxs = []
        if self.max_true == 0:
            false_idxs = np.random.randint(1, self.max_false, size=self.batch_size)
        elif self.max_false == 0:
            true_idxs = np.random.randint(1, self.max_true, size=self.batch_size)
        else:
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

# task_test = PropTask(depth=3, batch_size=5, split='test', cot=True, filter_ops=PropTask.imply_ops)
# xs, ys = next(task_test)

# # <codecell>
# tok = get_tokenizer()
# tok.decode(xs[0])

# # <codecell>
# # tok.decode(task_test.true_ds[10000]['input_ids'])
# task_test.true_ds[10000]['ops']

# # <codecell>
# count_ops(task_test.false_ds)

# # <codecell>
# def matches_ops(ex):
#     return set(ex['ops']).issubset(PropTask.imply_ops)

# filt_ds = task_test.ds.filter(matches_ops)

# # <codecell>
# t_vals = []
# f_vals = []

# for k, v in filt_ds.items():
#     name, num = k.split('_')

#     if name == 'True':
#         t_vals.append((int(num), len(v)))
#     else:
#         f_vals.append((int(num), len(v)))
        
# # <codecell>
# import matplotlib.pyplot as plt

# t_idx, t_lens = np.array(t_vals).T
# f_idx, f_lens = np.array(f_vals).T
# plt.scatter(t_idx, t_lens)
# plt.scatter(f_idx, f_lens)

# plt.xscale('log')
# plt.yscale('log')

# # <codecell>
# t_sort_idx = np.argsort(t_idx)
# t_cum = np.cumsum(t_lens[t_sort_idx]) / np.sum(t_lens)

# f_sort_idx = np.argsort(f_idx)
# f_cum = np.cumsum(f_lens[f_sort_idx]) / np.sum(f_lens)

# plt.plot(t_cum)
# plt.plot(f_cum)

# props = [0.25, 0.5, 0.75, 0.95]

# idxs = []
# for p in props:
#     # idx = np.sum(t_cum < p) - 1
#     # idxs.append(t_idx[t_sort_idx][idx].item())
#     idx = np.sum(f_cum < p) - 1
#     idxs.append(f_idx[f_sort_idx][idx].item())

# idxs

# final indices: [3, 5, 7, 12]

# <codecell>

yes_id = 13138
no_id = 32165
state_id = 5219

# NOTE: un-optimized and very expensive to run
# TODO: give extra buffer for x
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

