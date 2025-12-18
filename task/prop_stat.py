"""
Computing basic statistics for PITA splits
"""

# <codecell>
import math

import numpy as np
import matplotlib.pyplot as plt

def full_width(n_nodes):
    n_ops = 3
    n_atoms = 3
    n_perms = math.perm(n_atoms, min(n_nodes, n_atoms))

    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = n_ops**n * cat
    by_atoms = (n_atoms + 2)**n_nodes * by_ops
    return by_atoms / n_perms


def imply_width(n_nodes):
    n_ops = 1
    n_atoms = 3
    n_perms = math.perm(n_atoms, min(n_nodes, n_atoms))

    n = n_nodes - 1
    fac1 = 1 / (n + 1)
    fac2 = math.comb(2 * n, n)
    cat = fac1 * fac2

    by_ops = n_ops**n * cat
    by_atoms = (n_atoms + 2)**n_nodes * by_ops
    return by_atoms / n_perms


def or_width(n_nodes):
    return np.sum(np.arange(2, n_nodes + 2))


ls = np.arange(1, 11)
full_ws = [full_width(n) for n in ls]
imply_ws = [imply_width(n) for n in ls]
or_ws = [or_width(n) for n in ls]

plt.plot(ls, full_ws, '--o')
plt.plot(ls, imply_ws, '--o')
plt.plot(ls, or_ws, '--o')
plt.yscale('log')

plt.axhline(1e6, color='C3', linestyle='--')
