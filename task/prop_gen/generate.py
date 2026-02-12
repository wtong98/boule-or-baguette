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
from util.sample import *


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
            yield item


run_id = uuid.uuid4()
print('RUN_ID', run_id)

n_atoms = 3
max_nodes = 8
n_cores = 16 * 16
keep = 'until_success'

run_split = 72



run_idx = int(sys.argv[1])
all_ex = gen_php(seed=1130, do_start=True)

all_ex = list(split(all_ex, run_split))

pbar = tqdm(total=len(all_ex))

out_dir = Path('/n/netscratch/pehlevan_lab/Lab/wlt/prop/php_enum')


if not out_dir.exists():
    out_dir.mkdir(parents=True, exist_ok=False)

out_path = out_dir / f'{run_id}.json'

def write_example(prop):
    proof = prove(prop, keep=keep)
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

