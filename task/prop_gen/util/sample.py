"""Enumerate new propositions"""

# <codecell>
import itertools as it
import math

try:
    from .data import *
except ImportError:
    from data import *

# ops = [And, Or, Implies]
ops = [Implies]


def gen_batch(n_atoms, n_nodes):
    atoms = [PFalse(), PTrue()] + [Atom(f'p{i+1}') for i in range(n_atoms)]
    
    for node_set in it.product(*it.repeat(atoms, n_nodes)):
        for example in catalan(node_set):
            yield example

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


def n_combo(n_atoms, n_nodes):
    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = len(ops)**(n) * cat
    by_atoms = (n_atoms + 2)**n_nodes * by_ops
    return by_atoms
    
