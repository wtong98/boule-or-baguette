"""
Generate propositional logic dataset

Code adapted from https://arxiv.org/abs/2404.07382
"""

# <codecell>
# from propositional_logic.random_gen.theorem_encoding import decode_prop
from util.data import *
from util.proof import prove
from util.out import format_example

# from propositional_logic.random_gen.consolidate_choices import commit_to_least_complex_success_choice, commit_drop_additional_choices_after_first_success
# from propositional_logic.random_gen.assumption_renaming import rename_assumption_top_level
# from propositional_logic.random_gen.to_training import get_thm_initial_state, get_thm_proof_text_top_level


# prop = decode_prop(1001, 3)
prop = Implies(Atom('p2'), Or(Atom('p1'), Atom('p2')))
# prop = Implies(Implies(Atom('p1'), Atom('p1')), Atom('p1'))
# prop = Implies(Atom('p1'), Implies(Atom('p1'), Atom('p1')))

print('PROP', prop)
proof = prove(prop, keep='simplest')
print('PROOF', proof)

# TODO: keep lean repl online
ex = format_example(2, prop, proof)
print(ex)

