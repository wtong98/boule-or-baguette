"""Enumerate new propositions"""

# <codecell>
import itertools as it
import math

import numpy as np

try:
    from .data import *
except ImportError:
    from data import *


def catalan(nodes):
    if len(nodes) == 1:
        yield nodes[0]
        return

    if len(nodes) == 2:
        for op in ops:
            yield op(nodes[0], nodes[1])
        return

    for idx, _ in enumerate(nodes[:-1]):
        left_branch = nodes[:(idx+1)]
        right_branch = nodes[(idx+1):]

        for left, right in it.product(catalan(left_branch), catalan(right_branch)):
            for op in ops:
                yield op(left, right)

# ops = [And, Or, Implies]
# ops = [Implies]
ops = [Or]

def gen_batch_or(n_atoms, n_nodes):
    assert len(ops) == 1 and ops[0] == Or
    
    atoms = [Atom(f'p{i+1}') for i in range(n_atoms)]
    for target_atom in atoms:
        # rem_atoms = [a for a in atoms if a is not target_atom]
        for node_set in it.product(*it.repeat(atoms, n_nodes)):
            for example in catalan(node_set):
                formula = Implies(target_atom, example)
                yield formula


def n_combo_or(n_atoms, n_nodes):
    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = len(ops)**(n) * cat
    by_atoms = n_atoms**n_nodes * by_ops
    return n_atoms * by_atoms


def gen_batch(n_atoms, n_nodes):
    atoms = [PFalse(), PTrue()] + [Atom(f'p{i+1}') for i in range(n_atoms)]
    
    for node_set in it.product(*it.repeat(atoms, n_nodes)):
        for example in catalan(node_set):
            yield example


def n_combo(n_atoms, n_nodes):
    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = len(ops)**(n) * cat
    by_atoms = (n_atoms + 2)**n_nodes * by_ops
    return by_atoms
    

### pigeon mania
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
    count = 0
    seed = kwargs.pop('seed', None)
    rng = np.random.default_rng(seed)
    
    while count < repeats:
        pigeons = list(pigeon(rng, *args, **kwargs))
        if len(pigeons) != 0:
            all_pigeons.extend(pigeons)
            count += 1
    
    return all_pigeons


def pigeon(rng, n_pigeons, n_holes, pigeon_occupation_ablation_prop=None, roommate_ablation_prop=None):
    atoms = [[Atom(f'p{i}{j}') for j in range(n_holes)] for i in range(n_pigeons)]

    pigeons = []
    for i in range(n_pigeons):
        curr_p = atoms[i]
        if pigeon_occupation_ablation_prop is not None:
            keep_idxs = rng.binomial(1, pigeon_occupation_ablation_prop, size=len(curr_p)).astype(bool)
            curr_p = [curr_p[j] for j, keep in enumerate(keep_idxs) if keep]

        pigeons.append(catalan_adv(curr_p, Or))

    pigeons_in_a_hole = catalan_adv(pigeons, And)

    all_pigeon_roommates = []
    for i1 in range(n_pigeons):
        for i2 in range(n_pigeons):
            if i1 != i2:
                statements = [And(atoms[i1][j], atoms[i2][j]) for j in range(n_holes)]
                all_pigeon_roommates.extend(statements)

    if roommate_ablation_prop is not None:
        keep_idxs = rng.binomial(1, roommate_ablation_prop, size=len(all_pigeon_roommates)).astype(bool)
        all_pigeon_roommates = [all_pigeon_roommates[i] for i, keep in enumerate(keep_idxs) if keep]

    all_pigeon_roommates = catalan_adv(all_pigeon_roommates, Or)
    php = catalan_adv([pigeons_in_a_hole, all_pigeon_roommates], Implies)
    return php


def gen_php(seed=5):
    n_pigeons = 4
    n_holes = 4

    param_sets = []
    for n_p in range(2, n_pigeons + 1):
        for n_h in range(1, n_holes + 1):
            pigeon_occupation_ablation_prop = None
            roommate_ablation_prop = None
            repeats = 1

            if n_p == 2 and n_h == 4:
                pigeon_occupation_ablation_prop = 0.75
                roommate_ablation_prop = 0.75
                repeats = 5
            
            elif n_p == 3 and n_h >= 3:
                pigeon_occupation_ablation_prop = 2 / n_h
                roommate_ablation_prop = 2 / n_h
                repeats = 5

            elif n_p == 4 and n_h >= 2:
                pigeon_occupation_ablation_prop = 1 / n_h
                roommate_ablation_prop = 1 / n_h
                repeats = 5
            
            param_sets.append({'n_pigeons': n_p,
                            'n_holes': n_h,
                            'pigeon_occupation_ablation_prop': pigeon_occupation_ablation_prop,
                            'roommate_ablation_prop': roommate_ablation_prop,
                            'repeats': repeats,
                            'seed': seed})

    from tqdm import tqdm
    all_pigeons = list(it.chain.from_iterable([pigeon_set(**params) for params in tqdm(param_sets)]))
    return all_pigeons

# php = gen_php(seed=3011)
# print('TOTAL PHP', len(php))
# print('EX 5000', php[5000])