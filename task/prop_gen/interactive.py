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


def or_tree_from_dyck(leaves: Sequence, word: str):
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
        return Or(left, right)

    tree = consume(0, len(word))
    try:
        next(leaf_iter)
    except StopIteration:
        return tree
    raise ValueError('Unused atoms remain after constructing the OR tree.')


def random_group(atoms, seed=None):
    dyck_word = sample_dyck_word(len(atoms) - 1, seed=seed)
    out = or_tree_from_dyck(atoms, dyck_word)
    return out


def gen_or(n_exs_per_set=10_000, seed=None):
    n_prop_set = np.arange(2, 30)
    switch = [False, True]

    max_pid = 100_000
    global_rng = np.random.default_rng(seed)

    ### START TEST CONFIG
    # max_pid = 10
    # n_prop_set = np.arange(2, 5)
    # switch = [False, True]
    # n_exs_per_set = 3
    ### END TEST CONFIG

    for n_props, is_true in it.product(n_prop_set, switch):
        for _ in range(n_exs_per_set):
            pids = global_rng.choice(max_pid, size=n_props + (not is_true), replace=False)

            if is_true:
                target_pid = global_rng.choice(pids)
                pids = np.append(pids, target_pid)

            atoms = [Atom(f'p{pid}') for pid in pids]
            target_atom = atoms[-1]
            all_atoms = atoms[:-1]

            seed = global_rng.integers(0, np.iinfo(np.int32).max)
            cons = random_group(all_atoms, seed=seed)
            prop = Implies(target_atom, cons)
            yield prop



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

