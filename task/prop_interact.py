"""Measuring different statistics of the prop task"""

# <codecell>
import datasets
import numpy as np
from transformers import AutoTokenizer
from prop import *

# <codecell>
# task = PropTask(depth=20, batch_size=5, split='test', cot=True, grow_fac=1, max_len=2000)
# # task = PropTask(depth=3, batch_size=5, split='test', cot=True, filter_ops=PropTask.imply_ops, grow_fac=1)
# xs, ys = next(task)
# xs.shape

# <codecell>
task = PropTask(100, 
                batch_size=2, 
                split='train', 
                cot='text', 
                ds_path='prop_gen/data/hf_php_text')
task.load_ds()
# next(task)

filt_ds = task.ds
filt_ds['True_9'][0]

# <codecell>
full_ds = datasets.concatenate_datasets(list(filt_ds.values()))

# <codecell>
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

def to_toks(ex):
    inp_and_proof_toks = tokenizer(ex['prompt'] + ex['completion'], return_attention_mask=False)

    return {
        # 'input_ids': inp_toks['input_ids'],
        # 'full_ids': inp_and_proof_toks['input_ids'],
        'len': len(inp_and_proof_toks['input_ids']),
    }

full_ds = full_ds.map(to_toks, batched=False, num_proc=16)
full_ds[0]


# <codecell>
from multiprocessing import Pool

def get_len(ex):
    return ex['len']

# all_lens = [ex['len'] for ex in tqdm(all_lens)]
all_lens = Pool(16).map(get_len, tqdm(full_ds))
all_lens

# <codecell>
import matplotlib.pyplot as plt
plt.hist(all_lens, bins=50)
np.quantile(all_lens, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# quantiles for depth 3:  array([ 428.,  552.,  680.,  813.,  939., 1518.])
# quantiles for depth 20: array([ 779., 1073., 1659., 2269., 2599., 4025.])


# <codecell>
max_len = 1000

def filter_len(ex):
    return len(ex['full_ids']) <= max_len

filt_ds = task.ds.filter(filter_len, num_proc=16)

# <codecell>
def matches_ops(ex):
    return set(ex['ops']).issubset(PropTask.imply_ops)

filt_ds = filt_ds.filter(matches_ops, num_proc=16)

# <codecell>
t_vals = []
f_vals = []

for k, v in filt_ds.items():
    name, num = k.split('_')

    if name == 'True':
        t_vals.append((int(num), len(v)))
    else:
        f_vals.append((int(num), len(v)))
        
# <codecell>
import matplotlib.pyplot as plt

t_idx, t_lens = np.array(t_vals).T
f_idx, f_lens = np.array(f_vals).T
plt.scatter(t_idx, t_lens)
plt.scatter(f_idx, f_lens)

plt.xscale('log')
plt.yscale('log')

# <codecell>
t_sort_idx = np.argsort(t_idx)
t_cum = np.cumsum(t_lens[t_sort_idx]) / np.sum(t_lens)

f_sort_idx = np.argsort(f_idx)
f_cum = np.cumsum(f_lens[f_sort_idx]) / np.sum(f_lens)

plt.plot(np.sort(t_idx), t_cum, 'o--')
plt.plot(np.sort(f_idx), f_cum, 'o--')

props = [0.2, 0.4, 0.5, 0.6, 0.8, 0.95]

idxs = []
for p in props:
    # idx = np.sum(t_cum < p) - 1
    # idxs.append(t_idx[t_sort_idx][idx].item())
    idx = np.sum(f_cum < p) - 1
    idxs.append(f_idx[f_sort_idx][idx].item())

idxs

# idxs for full, max_len=1000: [2, 3, 4, 5, 6, 10]
# idxs for full, max_len=1500: [2, 4, 6, 11]

# NOTE: 2, 3, 4, 5, 6, 10 may be globally a good bet
# TODO: integrate into training; implement more careful evaluation <-- STOPPED HERE
