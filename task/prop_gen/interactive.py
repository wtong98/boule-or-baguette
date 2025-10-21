"""
Generate propositional logic dataset

Parts of this code are adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
import json
import xml.etree.ElementTree as et

from util.data import *
from util.proof import prove
from util.out import format_example, start_lean
from util.sample import gen_batch, n_combo

# p1 ∨ (True → p3 → False) → p1

# def or_elem(is_true): return Or(PFalse(), PTrue() if is_true else PFalse())

# prop = Implies(Atom('p1'), Or(Atom('p1'), Atom('p2')))
# prop = Implies(Or(Atom('p1'), Implies(PTrue(), Implies(Atom('p3'), PFalse()))), Atom('p1'))
prop = Implies(Implies(Implies(Implies(Implies(Atom('p1'), Atom('p2')), Atom('p1')), Atom('p1')), Atom('p2')), Atom('p2'))

print('PROP', prop)
# proof = prove(prop, keep='until_success')
proof = prove(prop, keep='simplest')
print('PROOF', proof)

ex = format_example(3, prop, proof, proof_to_string=False)

print(ex['input'])

for elem in ex['proof']:
    et.indent(elem)
    print(et.tostring(elem, encoding='utf-8').decode('utf-8'))

# <codecell>
inp = et.fromstring(ex['input'])
et.indent(inp)
print(et.tostring(inp, encoding='utf-8').decode('utf-8'))
