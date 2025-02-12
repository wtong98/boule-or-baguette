"""
A simple transitive inference task

Paper implementation: https://github.com/sflippl/relational-generalization-in-ti/tree/main

author: William Tong (wtong@g.harvard.edu)
"""

'''
NOTE: Lippl's implementation concats one-hots directly
- may be solved by using a *learnable* embedding layer
'''

# <codecell>
import itertools
import numpy as np

class TiTask:
    """Adhere's closely to Lippl's implementation"""
    def __init__(self, n_symbols, sep_dists=None, one_hot_encode=True, add_pos_emb=False) -> None:
        self.n_symbols = n_symbols
        self.sep_dists = sep_dists
        self.add_pos_emb = add_pos_emb

        self.xs = itertools.product(
                            range(n_symbols), 
                            range(n_symbols))

        self.xs = [(a, b) for a, b in self.xs if a != b]
        if sep_dists is not None:
            self.xs = [(a, b) for a, b in self.xs if np.abs(a - b) in sep_dists]
        self.ys = [1 if a > b else 0 for a, b in self.xs]

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)

        if one_hot_encode:
            # slick code golf from 
            # https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
            self.xs = (np.arange(self.xs.max() + 1) == self.xs[...,None]).astype(float)

            if add_pos_emb:
                pos_enc = np.tile(np.eye(2), (len(self.xs), 1, 1))
                self.xs = np.concat((self.xs, pos_enc), axis=-1)
    
    def __next__(self):
        return self.xs, self.ys
    
    def __iter__(self):
        return self

# TODO: test pos emb <-- STOPPED HERE
# task = TiTask(3, sep_dists=[1], one_hot_encode=True, add_pos_emb=True)
# xs, ys = next(task)
# xs

# %%
