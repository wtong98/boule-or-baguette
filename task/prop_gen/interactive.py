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

from util.data import *
from util.proof import prove
from util.out import format_example, start_lean
from util.sample import gen_batch, n_combo, gen_php
from tqdm import tqdm

# # <codecell>
# tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')
# tok


# p1 ∨ (True → p3 → False) → p1

# def or_elem(is_true): return Or(PFalse(), PTrue() if is_true else PFalse())

# prop = Implies(Atom('p1'), Or(Atom('p1'), Atom('p2')))
# prop = Implies(Or(Atom('p1'), Implies(PTrue(), Implies(Atom('p3'), PFalse()))), Atom('p1'))

# def chi(x, y):
#     prop = Implies(Implies(Implies(Implies(x, y), x), x), y)
#     return prop

# xi1 = chi(Atom('p2'), Atom('p1'))
# xi2 = chi(Atom('p3'), xi1)
# prop = Implies(xi2, Atom('p1'))

# prop = Implies(Implies(Implies(Implies(Implies(Atom('p1'), Atom('p2')), Atom('p1')), Atom('p1')), Atom('p2')), Atom('p2'))

def prod(a, b):
    for x in a:
        for y in b:
            yield (x, y)


def catalan(nodes, op):
    if len(nodes) == 1:
        try:
            for n in nodes[0]:
                yield n
        except TypeError:
            yield nodes[0]
        return

    if len(nodes) == 2:
        left, right = nodes
        try:
            for l, r in prod(left, right):
                yield op(l, r)
        except TypeError:
            yield op(left, right)
        return

    for idx, _ in enumerate(nodes[:-1]):
        left_branch = nodes[:(idx+1)]
        right_branch = nodes[(idx+1):]
        
        for left, right in prod(catalan(left_branch, op), catalan(right_branch, op)):
            try:
                for l, r in prod(left, right):
                    yield op(l, r)
            except TypeError:
                yield op(left, right)

# atoms = [Atom(f'p{i+1}') for i in range(3)]
# list(catalan([atoms, atoms, atoms], Or))

# <codecell>
def pigeon(n_pigeons, n_holes, pigeon_occupation_ablation_prop=None, roommate_ablation_prop=None):
    # atoms = [[Atom(f'p{i * n_holes + j}') for j in range(n_holes)] for i in range(n_pigeons)]
    atoms = [[Atom(f'p{i}{j}') for j in range(n_holes)] for i in range(n_pigeons)]

    pigeons = []
    for i in range(n_pigeons):
        curr_p = atoms[i]
        if pigeon_occupation_ablation_prop is not None:
            keep_idxs = np.random.binomial(1, pigeon_occupation_ablation_prop, size=len(curr_p)).astype(bool)
            curr_p = [curr_p[j] for j, keep in enumerate(keep_idxs) if keep]

        pigeons.append(catalan(curr_p, Or))

    pigeons_in_a_hole = catalan(pigeons, And)

    all_pigeon_roommates = []
    for i1 in range(n_pigeons):
        for i2 in range(n_pigeons):
        # for i2 in range(i1 + 1):
            if i1 != i2:
                statements = [And(atoms[i1][j], atoms[i2][j]) for j in range(n_holes)]
                all_pigeon_roommates.extend(statements)

    if roommate_ablation_prop is not None:
        keep_idxs = np.random.binomial(1, roommate_ablation_prop, size=len(all_pigeon_roommates)).astype(bool)
        all_pigeon_roommates = [all_pigeon_roommates[i] for i, keep in enumerate(keep_idxs) if keep]

    all_pigeon_roommates = catalan(all_pigeon_roommates, Or)
    php = catalan([pigeons_in_a_hole, all_pigeon_roommates], Implies)
    return php

tot = 0
for _ in tqdm(pigeon(3, 4, 
                     pigeon_occupation_ablation_prop=0.5,
                     roommate_ablation_prop=0.5)):
    tot += 1

print(tot)

php = gen_php()
# <codecell>
# tot_pigeons = 4
# tot_holes = 4

# all_props = []
# for n in range(2, tot_pigeons + 1):
#     for m in range(1, tot_holes + 1):
#         if n + m <= 5:
#             props = list(pigeon(n, m))
#             print('n', n, m)
#             print(len(props))
#             all_props.extend(props)

# print(len(all_props))

atoms = pigeon(3, 4, pigeon_occupation_ablation_prop=0.5, roommate_ablation_prop=0.5)
prop = next(atoms)
print(prop)

# <codecell>
# p11 = Atom('p11')
# p21 = Atom('p21')
# p31 = Atom('p31')

# left = And(And(p11, p21), p31)
# right = Or(And(p11, p21), Or(And(p21, p31), And(p31, p11)))

# prop = Implies(left, right)
# prop = atoms[-1]

prop = php[5000]
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

