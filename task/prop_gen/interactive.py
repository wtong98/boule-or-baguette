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

n_combo(3, 5)

# <codecell>
prop = Implies(Atom('p1'), Or(PFalse(), PTrue()))


print('PROP', prop)
proof = prove(prop, keep='until_success')
print('PROOF', proof)

ex = format_example(3, prop, proof, proof_to_string=False)

for elem in ex['proof']:
    et.indent(elem)
    print(et.tostring(elem, encoding='utf-8').decode('utf-8'))
