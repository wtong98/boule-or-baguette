"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
import itertools as it
import json
import xml.etree.ElementTree as et

import numpy as np
from transformers import AutoTokenizer
from typing import Optional, Sequence

from util.data import *
from util.proof import prove
from util.out import format_example, start_lean
from util.sample import gen_batch, n_combo, gen_php
from tqdm import tqdm
from functools import lru_cache
# <codecell>
prop = Implies(Atom('p'), Atom('p'))

print('PROP', prop)
# proof = prove(prop, keep='until_success')
proof = prove(prop, keep='simplest')
print('PROOF', proof)

ex = format_example(3, prop, proof, proof_to_string=False)

print(ex['input'])

all_proof_lines = []
for elem in ex['proof']:
    proof_line = et.tostring(elem, encoding='utf-8').decode('utf-8')
    all_proof_lines.append(proof_line)

    et.indent(elem)
    proof_line = et.tostring(elem, encoding='utf-8').decode('utf-8')
    print(proof_line)

# <codecell>
inp = et.fromstring(ex['input'])
et.indent(inp)
print(et.tostring(inp, encoding='utf-8').decode('utf-8'))

# <codecell>
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')
tok

# <codecell>
example = ex['input'] + ''.join(all_proof_lines)
ids = tok.encode(example)
len(ids)

# <codecell>
example = ex['input'] + ''.join(all_proof_lines)
out = '<example>' + example + '</example>'

full_ex = et.fromstring(out)
et.indent(full_ex)
print(et.tostring(full_ex, encoding='utf-8').decode('utf-8'))

