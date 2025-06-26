"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import json
import xml.etree.ElementTree as et

from tqdm import tqdm

from util.data import *
from util.proof import prove
from util.out import format_example 
from util.sample import gen_batch, n_combo

n_atoms = 3
max_nodes = 5
n_cores = 112

out_path = '/scratch/data.json'

### START TEST CONFIG
# out_path = 'data.json'
# n_cores = 16
n_atoms = 1
max_nodes = 1
### END TEST CONFIG

all_ex = itertools.chain(*[gen_batch(n_atoms, n) for n in range(1, max_nodes + 1)])
total_ex = sum(n_combo(n_atoms, n) for n in range(1, max_nodes + 1))

pbar = tqdm(total=total_ex)

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

# TODO: fix paths and install lean <-- STOPPED HERE