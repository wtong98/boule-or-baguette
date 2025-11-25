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


@lru_cache(maxsize=None)
def _count_dyck_suffixes(n_pairs: int, open_used: int, close_used: int) -> int:
    if open_used == n_pairs:
        return 1

    total = 0
    if open_used < n_pairs:
        total += _count_dyck_suffixes(n_pairs, open_used + 1, close_used)
    if close_used < open_used:
        total += _count_dyck_suffixes(n_pairs, open_used, close_used + 1)

    return total


def sample_dyck_word(n_pairs: int, seed=None) -> str:
    if n_pairs <= 0:
        return ''
    rng = np.random.default_rng(seed)

    word = []
    open_used = 0
    close_used = 0
    total_len = 2 * n_pairs

    while len(word) < total_len:
        n_left = 0
        n_right = 0

        if open_used < n_pairs:
            n_left = _count_dyck_suffixes(n_pairs, open_used + 1, close_used)
        if close_used < open_used:
            n_right = _count_dyck_suffixes(n_pairs, open_used, close_used + 1)

        p_left = n_left / (n_left + n_right) if (n_left + n_right) > 0 else 0
        is_left = rng.random() < p_left
        
        if is_left:
            word.append('(')
            open_used += 1
        else:
            word.append(')')
            close_used += 1

    return ''.join(word)


def _match_parentheses(word: str) -> dict[int, int]:
    stack = []
    matches: dict[int, int] = {}
    for idx, char in enumerate(word):
        if char == '(':
            stack.append(idx)
        elif char == ')':
            if not stack:
                raise ValueError('Invalid Dyck word: unmatched closing parenthesis.')
            start = stack.pop()
            matches[start] = idx
        else:
            raise ValueError('Dyck word must consist only of "(" and ")" characters.')
    if stack:
        raise ValueError('Invalid Dyck word: unmatched opening parenthesis.')
    return matches


def group_by_samp_dyck(op, leaves, word):
    leaves = tuple(leaves)
    if not leaves:
        raise ValueError('Cannot build expression with no atoms.')

    expected_pairs = len(leaves) - 1
    if expected_pairs == 0:
        if word:
            raise ValueError('Dyck word should be empty when only one atom is provided.')
        return leaves[0]
    if len(word) != 2 * expected_pairs:
        raise ValueError('Dyck word length does not match number of atoms.')

    matches = _match_parentheses(word)
    leaf_iter = iter(leaves)

    def consume(start: int, end: int):
        if start >= end:
            try:
                return next(leaf_iter)
            except StopIteration as exc:
                raise ValueError('Dyck word consumed more atoms than provided.') from exc

        if word[start] != '(':
            raise ValueError('Invalid Dyck word structure.')

        try:
            split = matches[start]
        except KeyError as exc:
            raise ValueError('Invalid Dyck word: missing matching parenthesis.') from exc

        if split >= end:
            raise ValueError('Invalid Dyck word segmentation.')

        left = consume(start + 1, split)
        right = consume(split + 1, end)
        return op(left, right)

    tree = consume(0, len(word))
    try:
        next(leaf_iter)
    except StopIteration:
        return tree
    raise ValueError('Unused atoms remain after constructing the OR tree.')


def random_group(op, atoms, seed=None):
    dyck_word = sample_dyck_word(len(atoms) - 1, seed=seed)
    out = group_by_samp_dyck(op, atoms, dyck_word)
    return out

def new_seed(rng):
    return rng.integers(0, np.iinfo(np.int32).max)

def prod(a, b):
    for x in a:
        for y in b:
            yield (x, y)


def catalan_adv(nodes, op):
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
        
        for left, right in prod(catalan_adv(left_branch, op), catalan_adv(right_branch, op)):
            try:
                for l, r in prod(left, right):
                    yield op(l, r)
            except TypeError:
                yield op(left, right)


def pigeon_set(repeats=1, *args, **kwargs):
    all_pigeons = []
    seed = kwargs.pop('seed', None)
    rng = np.random.default_rng(seed)
    
    all_pigeons = [pigeon(rng, *args, **kwargs) for _ in range(repeats)]
    return all_pigeons


def pigeon(rng, n_pigeons, n_holes, pigeon_occupation_ablation_prop=None, roommate_ablation_prop=None):
    atoms = [[Atom(f'p{i}{j}') for j in range(n_holes)] for i in range(n_pigeons)]

    pigeons = []
    for i in range(n_pigeons):
        curr_p = atoms[i]
        if pigeon_occupation_ablation_prop is not None:
            keep_idxs = rng.binomial(1, pigeon_occupation_ablation_prop, size=len(curr_p)).astype(bool)

            if np.sum(keep_idxs) == 0:
                keep_idxs[rng.integers(0, len(keep_idxs))] = True

            curr_p = [curr_p[j] for j, keep in enumerate(keep_idxs) if keep]

        pigeons.append(random_group(Or, curr_p, seed=new_seed(rng)))

    pigeons_in_a_hole = random_group(And, pigeons, seed=new_seed(rng))

    all_pigeon_roommates = []
    for i1 in range(n_pigeons):
        for i2 in range(n_pigeons):
            if i1 != i2:
                statements = [And(atoms[i1][j], atoms[i2][j]) for j in range(n_holes)]
                all_pigeon_roommates.extend(statements)

    if roommate_ablation_prop is not None:
        keep_idxs = rng.binomial(1, roommate_ablation_prop, size=len(all_pigeon_roommates)).astype(bool)

        if np.sum(keep_idxs) == 0:
            keep_idxs[rng.integers(0, len(keep_idxs))] = True

        all_pigeon_roommates = [all_pigeon_roommates[i] for i, keep in enumerate(keep_idxs) if keep]

    all_pigeon_roommates = random_group(Or, all_pigeon_roommates, seed=new_seed(rng))
    php = Implies(pigeons_in_a_hole, all_pigeon_roommates)
    return php


# rng = np.random.default_rng()
# prop = pigeon(rng, 4, 2, pigeon_occupation_ablation_prop=0.5, roommate_ablation_prop=0.5)
# print(prop)

def gen_php(seed=5):
    n_pigeons = 4
    n_holes = 4
    reps_per_six_case = 100

    param_sets = []
    for n_p in range(2, n_pigeons + 1):
        for n_h in range(1, n_holes + 1):
            pigeon_occupation_ablation_prop = None
            roommate_ablation_prop = None
            repeats = 1

            if n_p == 2 and n_h == 4:
                pigeon_occupation_ablation_prop = 0.75
                roommate_ablation_prop = 0.75
                repeats = reps_per_six_case
            
            elif n_p == 3 and n_h >= 3:
                pigeon_occupation_ablation_prop = 2 / n_h
                roommate_ablation_prop = 2 / n_h
                repeats = reps_per_six_case

            elif n_p == 4 and n_h >= 2:
                pigeon_occupation_ablation_prop = 1 / n_h
                roommate_ablation_prop = 1 / n_h
                repeats = reps_per_six_case
            
            param_sets.append({'n_pigeons': n_p,
                            'n_holes': n_h,
                            'pigeon_occupation_ablation_prop': pigeon_occupation_ablation_prop,
                            'roommate_ablation_prop': roommate_ablation_prop,
                            'repeats': repeats,
                            'seed': seed})

    all_pigeons = it.chain.from_iterable([pigeon_set(**params) for params in param_sets])
    return all_pigeons

# TODO: scale up to 500k examples and send to cluster <-- STOPPED HERE
all_pigeons = gen_php()
for p in all_pigeons:
    print(p)

# <codecell>
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

