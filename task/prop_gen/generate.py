"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
from concurrent.futures import ThreadPoolExecutor 
import itertools
import json
from pathlib import Path
import uuid
import sys

from tqdm import tqdm

from util.data import *
from util.proof import prove
from util.out import format_example 
from util.sample import gen_batch, n_combo


def split(it, run_split):
    run_idx = sys.argv[1]
    try:
        run_idx = int(run_idx) % run_split
        print(f'info: run_idx = {run_idx}')
    except ValueError:
        print(f'warn: unable to parse index {run_idx}, setting run_idx=0')
        run_idx = 0
    
    div = run_idx % run_split

    for i, item in enumerate(it):
        if i % run_split == div:
            print('yield', item)
            yield item
        else:
            print('skip', item)
    

run_id = uuid.uuid4()

n_atoms = 3
max_nodes = 5
n_cores = 16 * 16

run_split = 12

out_dir = Path('/scratch/prop')

### START TEST CONFIG
# out_dir = Path('test_data')
# n_cores = 1
# n_atoms = 2
# max_nodes = 2
### END TEST CONFIG

if not out_dir.exists():
    out_dir.mkdir(parents=True)

out_path = out_dir / f'{run_id}.json'

all_ex = itertools.chain(*[gen_batch(n_atoms, n) for n in range(1, max_nodes + 1)])
all_ex = split(all_ex, run_split)
total_ex = sum(n_combo(n_atoms, n) for n in range(1, max_nodes + 1))

pbar = tqdm(total=total_ex // run_split)

def write_example(prop):
    proof = prove(prop, keep='until_success')
    ex = format_example(n_atoms, prop, proof)
    pbar.update(1)

    with open(out_path, 'a') as fp:
        json.dump(ex, fp)
        fp.write('\n')

with ThreadPoolExecutor(max_workers=n_cores) as executor:
    itr = all_ex
    executor.map(write_example, itr)

pbar.close()
print('done')

