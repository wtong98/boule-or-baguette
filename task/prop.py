"""Proposition task"""

# <codecell>
import functools
import os

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
    from .prop_gen.to_dataset import get_tokenizer
except ImportError:
    from prop_gen.to_dataset import get_tokenizer


trunc_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_trunc'
full_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_full'
implies_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_implies'
or_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_or'

full_text_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_full_text'
or_text_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_or_text'
imply_text_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_implies_text'
php_text_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_php_text'
php_enum_text_ds_path = '~/workspace/imply/imply/task/prop_gen/data/hf_php_enum_text'
debug_ds_path = None

yes_id = 366
no_id = 352
state_id = 264


class PropTask:
    n_vocab = 512

    or_ops = {'apply Or', 'cases Or', 'intro h', 'exact', 'apply True', 'efq'}
    imply_ops = {'split Imply', 'intro h', 'exact', 'apply True', 'efq'}

    def __init__(self, depth, ds_path=None, split='train', filter_ops=None, max_len=None, padding='longest', cot=False, n_proc=None, batch_size=128) -> None:
        assert batch_size > 1, f'require batch_size={batch_size} > 1'
        
        self.depth = depth
        self.ds_path = ds_path if ds_path is not None else debug_ds_path
        self.split = split
        self.filter_ops = filter_ops
        self.max_len = max_len
        self.cot = cot
        self.n_proc = n_proc if n_proc is not None else os.cpu_count()
        self.batch_size = batch_size

        self.ds = None

        self.true_ds = None
        self.max_true = None
        self.false_ds = None
        self.max_false = None

        self.tokenizer = get_tokenizer()
        self.collate = DataCollatorWithPadding(self.tokenizer, return_tensors='np', max_length=max_len, padding=padding)
    

    def _slice_dataset(self, prefix, start, stop):
        all_ds = []
        
        for key in self.ds.keys():
            name, i = key.split('_')
            name = True if name == 'True' else False
            i = int(i)

            if start <= i < stop and name == prefix:
                all_ds.append(self.ds[key])
        
        if len(all_ds) == 0:
            return self.ds[key].select(range(0))  # empty dataset

        return concatenate_datasets(all_ds)
    

    def load_ds(self):
        self.ds = DatasetDict.load_from_disk(self.ds_path)

        if self.split == 'train':
            self.true_ds = self._slice_dataset(True, 1, self.depth + 1)
            self.false_ds = self._slice_dataset(False, 1, self.depth + 1)
        elif self.split == 'test':
            self.true_ds = self._slice_dataset(True, self.depth + 1, np.inf)
            self.false_ds = self._slice_dataset(False, self.depth + 1, np.inf)
        elif self.split == 'range':
            self.true_ds = self._slice_dataset(True, self.depth[0], self.depth[1])
            self.false_ds = self._slice_dataset(False, self.depth[0], self.depth[1])
        else:
            raise ValueError(f'unrecognized split: {self.split}')
        
        if self.filter_ops is not None:
            assert type(self.filter_ops) is set, f'filter_ops must be a set, got {type(self.filter_ops)}'

            def matches_ops(ex):
                return set(ex['ops']).issubset(self.filter_ops)

            self.true_ds = self.true_ds.filter(matches_ops, num_proc=self.n_proc)
            self.false_ds = self.false_ds.filter(matches_ops, num_proc=self.n_proc)
        
        if self.max_len is not None and self.ds_path != implies_ds_path:
            def filter_len(ex):
                return len(ex['full_ids']) <= self.max_len

            self.true_ds = self.true_ds.filter(filter_len, num_proc=self.n_proc)
            self.false_ds = self.false_ds.filter(filter_len, num_proc=self.n_proc)
        
        self.max_true = len(self.true_ds)
        self.max_false = len(self.false_ds)

        if self.max_true <= 1 or self.max_false <= 1:
            print(f'warn: insufficient examples: max_true={self.max_true} and max_false={self.max_false}')

        if self.cot != 'text':
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
            if self.cot == 'final':
                xs[(xs == yes_id) | (xs == no_id)] = 0
                ys = batch['is_true'].astype(int)
            else:
                ys = xs[:,1:]
                xs = xs[:,:-1]
        else:
            ys = batch['is_true'].astype(int)

        if xs.shape[1] % 2 == 1:
            xs = np.concatenate((xs, np.zeros((xs.shape[0], 1))), axis=1)

            if self.cot and self.cot != 'final':
                ys = np.concatenate((ys, np.zeros((ys.shape[0], 1))), axis=1)

        return xs.astype(np.int16), ys.astype(np.int16)


    def __iter__(self):
        return self

# xs.shape


# <codecell>

# xs.dtype

# # <codecell>
# all_lens

# # <codecell>

# # quantiles for depth 3:  array([ 428.,  552.,  680.,  813.,  939., 1518.])
# # quantiles for depth 20: array([ 779., 1073., 1659., 2269., 2599., 4025.])

# # <codecell>

# # <codecell>
# # tok.decode(task_test.true_ds[10000]['input_ids'])

# # <codecell>

# # <codecell>


# # <codecell>


        
# # <codecell>



# # <codecell>




#     # idx = np.sum(t_cum < p) - 1
#     # idxs.append(t_idx[t_sort_idx][idx].item())

# idxs

# final indices: [3, 5, 7, 12]



# <codecell>

# NOTE: un-optimized and very expensive to run
def _old_gen_acc_cot_prop(state, batch, loss=None):
    tot_correct = 0
    all_exs = batch[0]

    for xs in tqdm(all_exs):
        xs = jnp.array(xs)
        start_idx = jnp.argmax((xs[2:] == state_id)) + 4
        preds = generate(state, xs, idx=start_idx)
        tot_correct += score(xs, preds)
    
    return {'gen_acc': tot_correct / len(all_exs)}
    

def gen_acc_cot_prop(state, batch, loss=None, **kwargs):
    seed = new_seed()
    return fast_gen_acc_cot_prop(state, batch, seed, **kwargs)


@functools.partial(jax.jit, static_argnames=['return_preds'])
def fast_gen_acc_cot_prop(state, batch, seed, beta=1, return_preds=False):
    xs = jnp.array(batch[0])

    start_idxs = jnp.argmax((xs[:,2:] == state_id), axis=-1) + 4
    preds = fast_generate(state, xs, idx=start_idxs, beta=beta, seed=seed)
    true_pos, true_neg, false_pos, false_neg = score(xs, preds)

    res = {
        'gen_acc': jnp.mean(true_pos + true_neg),
        'true_pos': jnp.mean(true_pos),
        'true_neg': jnp.mean(true_neg),
        'false_pos': jnp.mean(false_pos),
        'false_neg': jnp.mean(false_neg),
    }

    if return_preds:
        res['preds'] = preds
        res['xs'] = xs
        res['start_idxs'] = start_idxs

    return res
        



#     # return is_true * pred_is_true + (1 - is_true) * pred_is_false


def score(xs, preds):
    is_true = jnp.argmax(xs == yes_id, axis=-1) > 0
    t = jnp.argmax(preds == yes_id, axis=-1)
    f = jnp.argmax(preds == no_id, axis=-1)

    pred_is_true = (t != 0) * ((f == 0) + (t < f))
    pred_is_false = (f != 0) * ((t == 0) + (f < t))

    true_pos = is_true * pred_is_true
    true_neg = (1 - is_true) * pred_is_false
    false_pos = (1 - is_true) * pred_is_true
    false_neg = is_true * pred_is_false

    return true_pos, true_neg, false_pos, false_neg


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


def fast_generate(state, xs, idx, beta=1, seed=0):
    max_len = xs.shape[1]

    def cond_fun(val):
        _, idx, _ = val
        # NOTE: technically off-by-one for a subset of runs
        return jnp.any((idx + 1) < max_len - 1)

    def body_fun(val):
        xs, idx, key = val
        key, subkey = jax.random.split(key)
        xs = _gen_pass(subkey, state, xs, idx, beta)
        next_idx = idx + 1 * ((idx + 1) < max_len - 1)
        return xs, next_idx, key

    key = jax.random.PRNGKey(seed)
    xs, idx, _ = jax.lax.while_loop(cond_fun, body_fun, (xs, idx, key))
    return xs.squeeze()


def _gen_pass(key, state, xs, idx, beta):
    batch_size = xs.shape[0]

    logits = state.apply_fn({'params': state.params}, xs)
    pred = jax.random.categorical(key, beta * logits[jnp.arange(batch_size), idx])
    xs = xs.at[jnp.arange(batch_size), idx + 1].set(pred)
    return xs


# # <codecell>




# state

# # <codecell>

# # <codecell>


# # <codecell>

