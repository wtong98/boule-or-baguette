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


# <codecell>
# def or_elem(is_true): return Or(PFalse(), PTrue() if is_true else PFalse())

prop = Implies(Atom('p1'), Or(Or(Atom('p2'), Atom('p2')), Or(Atom('p3'), Atom('p1'))))


print('PROP', prop)
proof = prove(prop, keep='until_success')
print('PROOF', proof)

ex = format_example(3, prop, proof, proof_to_string=False)

for elem in ex['proof']:
    et.indent(elem)
    print(et.tostring(elem, encoding='utf-8').decode('utf-8'))
