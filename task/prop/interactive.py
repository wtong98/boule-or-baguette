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
prop = Implies(Atom('p3'), Or(Or(Atom('p1'), Atom('p2')), Atom('p3')))
# prop = Implies(Implies(Atom('p1'), And(Atom('p2'), Atom('p1'))), Implies(Atom('p1'), And(Atom('p2'), Atom('p2'))))
# prop = Implies(And(Atom('p1'), And(Atom('p2'), Atom('p2'))), And(Atom('p2'), And(Atom('p1'), Atom('p1'))))
# prop = Implies(Or(Atom('p1'), Atom('p1')), Atom('p1'))


print('PROP', prop)
proof = prove(prop, keep='until_success')
print('PROOF', proof)

with start_lean() as lean_proc:
    start, res, is_true = format_example(lean_proc, 3, prop, proof)

print(et.tostring(start, encoding='utf-8').decode('utf-8'))
for elem in res:
    # et.indent(elem)
    print(et.tostring(elem, encoding='utf-8').decode('utf-8'))

print('is true:', is_true)

with open('tmp.jsonl', 'a') as fp:
    for _ in range(3):
        ex = {
            'input': et.tostring(start, encoding='utf-8').decode('utf-8'),
            'proof': ''.join([et.tostring(r, encoding='utf-8').decode('utf-8') for r in res]),
            'is_true': is_true
        }

        json.dump(ex, fp)
        fp.write('\n')