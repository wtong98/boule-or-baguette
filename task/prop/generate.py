"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
from util.data import *
from util.proof import prove
from util.out import format_example


# prop = decode_prop(1001, 3)
prop = Implies(Atom('p2'), Or(Atom('p1'), Atom('p2')))

print('PROP', prop)
proof = prove(prop, keep='simplest')
print('PROOF', proof)

# TODO: keep lean repl online
ex = format_example(2, prop, proof)
print(ex)

