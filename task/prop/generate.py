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

n_atoms = 1
max_nodes = 2

all_ex = itertools.chain(*[gen_batch(4, n) for n in range(1, max_nodes + 1)])
total_ex = sum(n_combo(4, n) for n in range(1, max_nodes + 1))

pbar = tqdm(total=total_ex)

def write_example(prop):
    proof = prove(prop, keep='until_success')
    ex = format_example(n_atoms, prop, proof)
    pbar.update(1)

    with open('tmp.jsonl', 'a') as fp:
        json.dump(ex, fp)
        fp.write('\n')
    
    

with ThreadPoolExecutor(max_workers=16) as executor:
    itr = all_ex
    executor.map(write_example, itr)


pbar.close()
print('done')

        
        
        # for prop in tqdm(all_ex, total=total_ex):
        #     proof = prove(prop, keep='until_success')
        #     start, res, is_true = format_example(lean_proc, n_atoms, prop, proof)

        #     ex = {
        #         'input': et.tostring(start, encoding='utf-8').decode('utf-8'),
        #         'is_true': is_true,
        #         'proof': ''.join([et.tostring(r, encoding='utf-8').decode('utf-8') for r in res])
        #     }

        #     json.dump(ex, fp)
        #     fp.write('\n')
