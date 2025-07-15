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


# prop = decode_prop(1001, 3)
# prop = Implies(Atom('p3'), Implies(Or(Atom('p1'), Or(Atom('p2'), Atom('p3'))), Or(Atom('p1'), Atom('p3'))))
# prop = Implies(Atom('p3'), Or(Or(Atom('p1'), Atom('p2')), Atom('p3')))

# prop = Implies(Implies(Atom('p1'), And(Atom('p2'), Atom('p1'))), Implies(Atom('p1'), And(Atom('p2'), Atom('p2'))))
prop = Implies(Atom('p1'), Implies(PTrue(), PTrue()))
# prop = Implies(And(Atom('p1'), And(Atom('p2'), Atom('p2'))), And(Atom('p2'), And(Atom('p1'), Atom('p1'))))
# prop = Implies(Or(Atom('p1'), Atom('p1')), Atom('p1'))


print('PROP', prop)
proof = prove(prop, keep='until_success')
print('PROOF', proof)

ex = format_example(3, prop, proof)

ex
